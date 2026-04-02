#!/usr/bin/env python3
"""
scripts/batch_upload_papers.py
=============================
Command-line script for batch processing multiple papers.

Features:
- Process multiple PDFs from a directory
- Auto-approve high-confidence entities
- Optionally auto-merge to graph
- Progress tracking and logging
- Error handling and recovery
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pdf_processor import process_pdf, clean_pdf_text
from src.pmc_service import (
    DEFAULT_HTTP_RETRIES,
    process_pmc_url_advanced,
    process_pmc_url_html_fallback,
)
from src.paper_entity_extractor import (
    extract_entities_from_text,
    validate_extracted_entities,
    format_extraction_for_review
)
from src.paper_db_manager import (
    merge_paper_entities_to_graph,
    merge_multiple_papers,
    print_merge_report
)
import deb_data_papers as papers_db
import deb_data
from logger_config import setup_logger

logger = setup_logger(__name__)


def process_single_pmc_url(
    pmc_url: str,
    auto_approve_threshold: float = 0.85,
    auto_merge: bool = False,
    dry_run: bool = False,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> bool:
    """Process a single PMC URL using API-first pipeline and fallback to HTML extraction."""
    try:
        logger.info("Processing PMC URL: %s", pmc_url)

        advanced = process_pmc_url_advanced(pmc_url, retries=retries)

        metadata = {
            "title": "",
            "authors": [],
            "pmid": "",
            "doi": "",
            "abstract": "",
            "sections": [],
            "pubdate": "",
        }
        full_text = None

        if advanced:
            metadata.update({
                "title": advanced.get("title", ""),
                "authors": advanced.get("authors", []),
                "abstract": "",
                "sections": advanced.get("sections", []),
                "pubdate": advanced.get("pubdate", ""),
                "pmcid": advanced.get("pmcid", ""),
            })
            full_text = advanced.get("text")

        if not full_text:
            fallback = process_pmc_url_html_fallback(pmc_url, retries=retries)
            metadata.update({
                "title": fallback.get("title", metadata.get("title", "")),
                "authors": fallback.get("authors", metadata.get("authors", [])),
                "pmid": fallback.get("pmid", metadata.get("pmid", "")),
                "doi": fallback.get("doi", metadata.get("doi", "")),
                "sections": fallback.get("sections", metadata.get("sections", [])),
                "pubdate": fallback.get("pubdate", metadata.get("pubdate", "")),
            })
            full_text = fallback.get("text", "")

        clean_text = clean_pdf_text(full_text)

        extracted = extract_entities_from_text(
            clean_text,
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", "")
        )

        if "error" in extracted:
            logger.error("Extraction error for %s: %s", pmc_url, extracted["error"])
            return False

        validated = validate_extracted_entities(
            extracted,
            min_confidence=auto_approve_threshold
        )

        if dry_run:
            total = sum(len(validated.get(t, [])) for t in ["genes", "proteins", "diseases", "pathways"])
            logger.info("[DRY RUN] Would process PMC URL %s with %s entities", pmc_url, total)
            return True

        import uuid
        paper_id = metadata.get("pmid", "") or str(uuid.uuid4())

        papers_db.add_paper(
            paper_id=paper_id,
            title=metadata.get("title", pmc_url),
            authors=metadata.get("authors", []),
            pmid=metadata.get("pmid", ""),
            doi=metadata.get("doi", ""),
            abstract=metadata.get("abstract", ""),
            pdf_path="",
            publication_date=metadata.get("pubdate", "") or "2024-01-01",
            source="pmc_link",
            source_url=pmc_url,
            sections=metadata.get("sections", []),
        )

        for entity_type in ["genes", "proteins", "diseases", "pathways"]:
            entities = validated.get(entity_type, [])
            for entity in entities:
                entity["approved"] = entity.get("confidence", 0) >= auto_approve_threshold
            if entities:
                papers_db.add_entities(paper_id, entity_type, entities)

        relationships = validated.get("relationships", [])
        for rel in relationships:
            rel["approved"] = rel.get("confidence", 0) >= auto_approve_threshold
        if relationships:
            papers_db.add_edges(paper_id, relationships)

        if auto_merge:
            papers_db.update_paper_status(paper_id, "approved")
            merge_stats = merge_paper_entities_to_graph(paper_id, dry_run=False)
            print_merge_report(merge_stats)
        else:
            papers_db.update_paper_status(paper_id, "extracted")

        logger.info("✓ Successfully processed PMC URL: %s", pmc_url)
        return True

    except Exception as exc:
        logger.error("Error processing PMC URL %s: %s", pmc_url, exc, exc_info=True)
        return False


def batch_process_pmc_urls(
    pmc_urls: List[str],
    auto_approve_threshold: float = 0.85,
    auto_merge: bool = False,
    dry_run: bool = False,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> Tuple[int, int]:
    """Batch process PMC URLs."""
    success_count = 0
    error_count = 0

    for pmc_url in pmc_urls:
        ok = process_single_pmc_url(
            pmc_url,
            auto_approve_threshold=auto_approve_threshold,
            auto_merge=auto_merge,
            dry_run=dry_run,
            retries=retries,
        )
        if ok:
            success_count += 1
        else:
            error_count += 1

    return success_count, error_count


def batch_process_pdfs(
    input_dir: str,
    auto_approve_threshold: float = 0.85,
    auto_merge: bool = False,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        auto_approve_threshold: Confidence threshold for auto-approval (0.0-1.0)
        auto_merge: Whether to automatically merge approved entities to graph
        dry_run: If True, don't actually save anything
    
    Returns:
        tuple: (success_count, error_count, total_entities_processed)
    """
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return 0, 0, 0
    
    # Find all PDF files
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    if not pdf_files:
        logger.warning("No PDF files found in directory")
        return 0, 0, 0
    
    success_count = 0
    error_count = 0
    total_entities = 0
    paper_ids = []
    
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing: {pdf_path.name}")
            
            # Process PDF
            metadata, full_text = process_pdf(str(pdf_path), max_pages=10)
            clean_text = clean_pdf_text(full_text)
            
            # Extract entities
            logger.info("Extracting entities...")
            extracted = extract_entities_from_text(
                clean_text,
                title=metadata.get("title", ""),
                abstract=metadata.get("abstract", "")
            )
            
            if "error" in extracted:
                logger.error(f"Extraction error: {extracted['error']}")
                error_count += 1
                continue
            
            # Validate entities
            validated = validate_extracted_entities(
                extracted,
                min_confidence=auto_approve_threshold
            )
            
            # Count entities
            entity_count = sum(
                len(validated.get(etype, []))
                for etype in ["genes", "proteins", "diseases", "pathways"]
            )
            total_entities += entity_count
            
            if dry_run:
                logger.info(f"[DRY RUN] Would extract {entity_count} entities from {pdf_path.name}")
            else:
                # Generate paper ID
                import uuid
                paper_id = metadata.get("pmid", str(uuid.uuid4()))
                
                # Save paper
                paper = papers_db.add_paper(
                    paper_id=paper_id,
                    title=metadata.get("title", pdf_path.stem),
                    authors=metadata.get("authors", []),
                    pmid=metadata.get("pmid", ""),
                    doi=metadata.get("doi", ""),
                    abstract=metadata.get("abstract", ""),
                    pdf_path=str(pdf_path),
                    publication_date=metadata.get("publication_date", "2024-01-01")
                )
                
                # Save entities (auto-approve if above threshold)
                for entity_type in ["genes", "proteins", "diseases", "pathways"]:
                    entities = validated.get(entity_type, [])
                    for entity in entities:
                        entity["approved"] = entity.get("confidence", 0) >= auto_approve_threshold
                    if entities:
                        papers_db.add_entities(paper_id, entity_type, entities)
                
                # Save relationships (auto-approve if above threshold)
                relationships = validated.get("relationships", [])
                for rel in relationships:
                    rel["approved"] = rel.get("confidence", 0) >= auto_approve_threshold
                if relationships:
                    papers_db.add_edges(paper_id, relationships)
                
                # Update status
                if auto_approve_threshold <= 0.5:
                    # Auto-approve everything
                    papers_db.update_paper_status(paper_id, "approved")
                    paper_ids.append(paper_id)
                else:
                    papers_db.update_paper_status(paper_id, "extracted")
                
                logger.info(f"✓ Saved paper: {paper_id} with {entity_count} entities")
            
            success_count += 1
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
            error_count += 1
    
    # Auto-merge if requested
    if not dry_run and auto_merge and paper_ids:
        logger.info(f"Auto-merging {len(paper_ids)} papers to graph...")
        merge_stats = merge_multiple_papers(paper_ids, dry_run=False)
        print_merge_report(merge_stats)
    
    return success_count, error_count, total_entities


def process_single_paper(
    pdf_path: str,
    auto_approve_threshold: float = 0.85,
    auto_merge: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        auto_approve_threshold: Confidence threshold for auto-approval
        auto_merge: Whether to auto-merge to graph
        dry_run: If True, don't save
    
    Returns:
        bool: True if successful
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return False
        
        logger.info(f"Processing single paper: {pdf_path}")
        
        # Process PDF
        metadata, full_text = process_pdf(pdf_path, max_pages=10)
        clean_text = clean_pdf_text(full_text)
        
        # Extract entities
        extracted = extract_entities_from_text(
            clean_text,
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", "")
        )
        
        if "error" in extracted:
            logger.error(f"Extraction error: {extracted['error']}")
            return False
        
        # Validate
        validated = validate_extracted_entities(
            extracted,
            min_confidence=auto_approve_threshold
        )
        
        entity_count = sum(
            len(validated.get(etype, []))
            for etype in ["genes", "proteins", "diseases", "pathways"]
        )
        
        if dry_run:
            logger.info(f"[DRY RUN] Would process paper with {entity_count} entities")
            return True
        
        # Save
        import uuid
        paper_id = metadata.get("pmid", str(uuid.uuid4()))
        
        paper = papers_db.add_paper(
            paper_id=paper_id,
            title=metadata.get("title", Path(pdf_path).stem),
            authors=metadata.get("authors", []),
            pmid=metadata.get("pmid", ""),
            doi=metadata.get("doi", ""),
            abstract=metadata.get("abstract", ""),
            pdf_path=pdf_path,
            publication_date=metadata.get("publication_date", "2024-01-01")
        )
        
        # Save entities
        for entity_type in ["genes", "proteins", "diseases", "pathways"]:
            entities = validated.get(entity_type, [])
            for entity in entities:
                entity["approved"] = entity.get("confidence", 0) >= auto_approve_threshold
            if entities:
                papers_db.add_entities(paper_id, entity_type, entities)
        
        # Save relationships
        relationships = validated.get("relationships", [])
        for rel in relationships:
            rel["approved"] = rel.get("confidence", 0) >= auto_approve_threshold
        if relationships:
            papers_db.add_edges(paper_id, relationships)
        
        # Update status and merge if needed
        if auto_merge:
            papers_db.update_paper_status(paper_id, "approved")
            merge_stats = merge_paper_entities_to_graph(paper_id, dry_run=False)
            print_merge_report(merge_stats)
        else:
            papers_db.update_paper_status(paper_id, "extracted")
        
        logger.info(f"✓ Successfully processed: {paper_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing paper: {e}", exc_info=True)
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process research papers for entity extraction"
    )
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Directory containing PDF files to process"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Single PDF file to process"
    )

    parser.add_argument(
        "--pmc-url",
        action="append",
        help="Single PMC URL (can be passed multiple times)"
    )

    parser.add_argument(
        "--pmc-file",
        type=str,
        help="Path to text file with one PMC URL per line"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for auto-approval (0.0-1.0, default: 0.85)"
    )
    
    parser.add_argument(
        "-m", "--merge",
        action="store_true",
        help="Automatically merge approved entities to graph"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually save anything, just show what would happen"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.directory and not args.file and not args.pmc_url and not args.pmc_file:
        parser.print_help()
        print("\nError: Must specify --directory, --file, --pmc-url, or --pmc-file")
        sys.exit(1)
    
    selected_modes = sum(
        1 for flag in [bool(args.directory), bool(args.file), bool(args.pmc_url), bool(args.pmc_file)] if flag
    )
    if selected_modes > 1:
        print("Error: Specify only one input mode at a time")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("BATCH PAPER UPLOAD & EXTRACTION")
    print("="*60 + "\n")
    
    if args.file:
        # Process single file
        success = process_single_paper(
            args.file,
            auto_approve_threshold=args.threshold,
            auto_merge=args.merge,
            dry_run=args.dry_run
        )
        sys.exit(0 if success else 1)

    elif args.pmc_url or args.pmc_file:
        pmc_urls = list(args.pmc_url or [])
        if args.pmc_file:
            with open(args.pmc_file, "r", encoding="utf-8") as handle:
                pmc_urls.extend([line.strip() for line in handle.readlines() if line.strip()])

        success, errors = batch_process_pmc_urls(
            pmc_urls,
            auto_approve_threshold=args.threshold,
            auto_merge=args.merge,
            dry_run=args.dry_run,
        )

        print("\n" + "="*60)
        print("PMC BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Successfully processed: {success}")
        print(f"Errors encountered:     {errors}")
        print(f"Mode:                    {'DRY RUN' if args.dry_run else 'REAL'}")
        print("="*60 + "\n")

        sys.exit(0 if errors == 0 else 1)
    
    else:
        # Process directory
        success, errors, entities = batch_process_pdfs(
            args.directory,
            auto_approve_threshold=args.threshold,
            auto_merge=args.merge,
            dry_run=args.dry_run
        )
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Successfully processed: {success}")
        print(f"Errors encountered:     {errors}")
        print(f"Total entities:         {entities}")
        print(f"Mode:                    {'DRY RUN' if args.dry_run else 'REAL'}")
        print("="*60 + "\n")
        
        sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
