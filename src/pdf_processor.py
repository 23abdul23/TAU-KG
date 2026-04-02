"""
src/pdf_processor.py
====================
PDF text extraction and metadata parsing utilities for paper upload system.

Handles:
- PDF text extraction (robust with fallback for scanned PDFs)
- Basic metadata extraction (title, authors, abstract)
- PDF validation and error handling
"""

import os
import re
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import logging
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_PMC_BOILERPLATE_PATTERNS = [
    r"^Skip to main content$",
    r"^An official website of the United States government$",
    r"^Here's how you know$",
    r"^Official websites use \.gov$",
    r"^A \.gov website belongs to an official government organization in the United States\.$",
    r"^Secure \.gov websites use HTTPS$",
    r"^A lock \( Lock Locked padlock icon \) or https:// means you've safely connected to the \.gov website\.$",
    r"^Share sensitive information only on official, secure websites\.$",
    r"^Search PMC Full-Text Archive$",
    r"^Search in PMC$",
    r"^Journal List$",
    r"^User Guide$",
    r"^As a library, NLM provides access to scientific literature\.$",
    r"^Inclusion in an NLM database does not imply endorsement, or agreement with, the contents by NLM or the National Institutes of Health\.$",
    r"^Learn more: PMC Disclaimer \| PMC Copyright Notice$",
    r"^Open in a new tab$",
    r"^ACTIONS$",
    r"^View on publisher site$",
    r"^Cite$",
    r"^Collections$",
    r"^Permalink$",
    r"^RESOURCES$",
    r"^Similar articles$",
    r"^Cited by other articles$",
    r"^Links to NCBI Databases$",
    r"^On this page$",
    r"^Follow NCBI$",
    r"^NLM on .*$",
    r"^Connect with NLM$",
    r"^National Library of Medicine$",
    r"^Tell us what you think!$",
    r"^Articles from .*$",
]

_PMC_SECTION_END_MARKERS = {
    "references",
    "associated data",
    "supplementary material",
}


def _normalize_pmc_text(text: str) -> str:
    """Remove PMC navigation/boilerplate and keep the article body."""
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    cleaned_lines = []
    in_references = False

    for line in lines:
        if not line:
            continue

        lowered = line.lower()
        if lowered in _PMC_SECTION_END_MARKERS:
            in_references = True
            continue
        if in_references:
            continue

        if any(re.match(pattern, line) for pattern in _PMC_BOILERPLATE_PATTERNS):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return cleaned_text.strip()


class _TextExtractor(HTMLParser):
    """Lightweight HTML to text converter for PMC pages."""

    block_tags = {
        "article", "section", "header", "footer", "div", "p", "br", "li",
        "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "table",
        "thead", "tbody", "sup", "sub"
    }

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
            return
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self):
        text = " ".join(self.parts)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def is_pmc_url(source: str) -> bool:
    """Return True when the source looks like a PMC article URL."""
    if not source:
        return False
    parsed = urlparse(source)
    host = parsed.netloc.lower()
    return "pmc.ncbi.nlm.nih.gov" in host and "/articles/pmc" in parsed.path.lower()


def extract_text_from_pmc_url(url: str, max_chars: Optional[int] = None) -> str:
    """Fetch a PMC article page and convert the HTML body into plain text."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")

    parser = _TextExtractor()
    parser.feed(html)
    text = _normalize_pmc_text(parser.get_text())

    if max_chars:
        text = text[:max_chars]

    if not text.strip():
        raise IOError(f"Could not extract text from PMC page: {url}")

    return text


def extract_metadata_from_pmc_url(url: str, text: Optional[str] = None) -> Dict[str, Any]:
    """Extract basic metadata from a PMC page URL."""
    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": [],
    }

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")

    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()
        title = re.sub(r"\s*\|\s*PMC.*$", "", title)
        metadata["title"] = title

    if text:
        extracted = extract_metadata_from_text(text)
        metadata.update({k: extracted.get(k, metadata.get(k)) for k in ["title", "authors", "abstract", "keywords"]})

    pmcid_match = re.search(r"(PMC\d+)", url, re.IGNORECASE)
    if pmcid_match:
        metadata["pmcid"] = pmcid_match.group(1).upper()

    pmid_match = re.search(r"PMID:\s*\[?(\d{6,8})\]?", html, re.IGNORECASE)
    if pmid_match:
        metadata["pmid"] = pmid_match.group(1)

    doi_match = re.search(r"doi:\s*[^\n<]*?(10\.\d{4,9}/[^\s<\"]+)", html, re.IGNORECASE)
    if doi_match:
        metadata["doi"] = doi_match.group(1).rstrip(".,;)")

    return metadata


def process_pmc_url(url: str, max_chars: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
    """Process a PMC URL through the same metadata/text extraction pipeline as PDFs."""
    logger.info(f"Processing PMC URL: {url}")
    text = extract_text_from_pmc_url(url, max_chars=max_chars)
    metadata = extract_metadata_from_pmc_url(url, text=text)
    return metadata, text

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")


def validate_pdf(pdf_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate PDF file exists and meets size requirements.
    
    Args:
        pdf_path (str): Path to PDF file
        max_size_mb (int): Maximum allowed file size in MB
    
    Returns:
        bool: True if valid, raises exception if invalid
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file size exceeds limit or not a PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"File must be PDF format: {pdf_path}")
    
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size {file_size_mb:.1f}MB exceeds limit {max_size_mb}MB")
    
    return True


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from PDF file.
    
    Uses pdfplumber for best text extraction, falls back to PyPDF2 if needed.
    
    Args:
        pdf_path (str): Path to PDF file
        max_pages (int): Maximum pages to extract (None = all)
    
    Returns:
        str: Extracted text
    
    Raises:
        IOError: If text extraction fails completely
    """
    validate_pdf(pdf_path)
    
    text = ""
    
    # Try pdfplumber first (best for standard PDFs)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_limit = min(max_pages, len(pdf.pages)) if max_pages else len(pdf.pages)
                for i, page in enumerate(pdf.pages[:page_limit]):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} chars from {pdf_path} using pdfplumber")
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}. Trying PyPDF2...")
    
    # Fall back to PyPDF2
    if PYPDF_AVAILABLE:
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                page_limit = min(max_pages, len(pdf_reader.pages)) if max_pages else len(pdf_reader.pages)
                
                for page_num in range(page_limit):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} chars from {pdf_path} using PyPDF2")
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
    
    if not text.strip():
        raise IOError(f"Could not extract text from PDF: {pdf_path}. File may be scanned image or corrupted.")
    
    return text


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Extract basic metadata from PDF text using regex and heuristics.
    
    This is a heuristic approach. For best results, use PubMed API with PMID.
    
    Args:
        text (str): Extracted PDF text
    
    Returns:
        dict: Metadata dictionary with title, authors, abstract
    """
    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": []
    }
    
    # Get first 5000 chars (usually contains title, authors, abstract)
    header_text = text[:5000]
    lines = header_text.split('\n')
    
    # Extract title (usually first non-empty line or after certain keywords)
    title_candidates = []
    for i, line in enumerate(lines[:30]):
        line_stripped = line.strip()
        if len(line_stripped) > 10 and len(line_stripped) < 300 and line_stripped[0].isupper():
            # Look for typical title characteristics
            if not any(keyword in line_stripped for keyword in ['doi:', 'http', 'email', '@']):
                title_candidates.append(line_stripped)
    
    if title_candidates:
        # Pick longest reasonable title (usually the paper title)
        metadata["title"] = max(title_candidates, key=len)
    
    # Extract authors (look for "Author(s):", "by:", etc.)
    author_pattern = r'(?:Author[s]?:|by:?)\s*([^,\n]+(?:,\s*[^,\n]+)*)'
    author_match = re.search(author_pattern, header_text, re.IGNORECASE)
    if author_match:
        author_text = author_match.group(1)
        # Split by comma and clean
        authors = [a.strip() for a in author_text.split(',')]
        metadata["authors"] = [a for a in authors if a and len(a) > 2][:10]  # Max 10 authors
    
    # Extract abstract (look for "Abstract" section)
    abstract_pattern = r'(?:abstract|summary)\s*[:\n]+\s*([^(†‡§¶*)]+?)(?:\n\s*(?:introduction|keywords|methods|intro)|\Z)'
    abstract_match = re.search(abstract_pattern, text, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        # Limit to reasonable length
        metadata["abstract"] = abstract_text[:1500]
    
    # Extract keywords
    keywords_pattern = r'(?:keywords?|index terms)\s*[:\n]+\s*([^\n]+)'
    keywords_match = re.search(keywords_pattern, text, re.IGNORECASE)
    if keywords_match:
        keywords_text = keywords_match.group(1)
        keywords = [k.strip() for k in keywords_text.split(',')]
        metadata["keywords"] = keywords[:15]  # Max 15 keywords
    
    return metadata


def extract_pmid_from_text(text: str) -> Optional[str]:
    """
    Extract PubMed ID from PDF text.
    
    Args:
        text (str): Extracted PDF text
    
    Returns:
        str: PMID (8 digits) if found, None otherwise
    """
    # PMID pattern: 8 digits, usually appears as "PMID: 12345678" or similar
    pmid_pattern = r'(?:PMID|pmid|PubMed ID)\s*[:\s=]*(\d{8})'
    match = re.search(pmid_pattern, text[:3000])  # Search in header
    
    if match:
        return match.group(1)
    
    # Fallback: look for 8-digit number after common patterns
    alt_pattern = r'(?:PMID|pmid)\s+(\d{8})'
    match = re.search(alt_pattern, text)
    if match:
        return match.group(1)
    
    return None


def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract DOI from PDF text.
    
    Args:
        text (str): Extracted PDF text
    
    Returns:
        str: DOI if found, None otherwise
    """
    # DOI pattern: 10.XXXX/XXXXX
    doi_pattern = r'(?:DOI|doi)\s*[:\s=]*\s*(10\.\S+?)(?:\s|$|\n|\))'
    match = re.search(doi_pattern, text[:2000])
    
    if match:
        doi = match.group(1).rstrip('.,;:')
        return doi
    
    return None


def process_pdf(pdf_path: str, max_pages: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
    """
    Complete PDF processing pipeline: validation, text extraction, metadata extraction.
    
    Args:
        pdf_path (str): Path to PDF file
        max_pages (int): Maximum pages to process
    
    Returns:
        tuple: (metadata_dict, full_text_string)
    
    Raises:
        Exception: If processing fails at any stage
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Validate
    validate_pdf(pdf_path)
    
    # Extract text
    text = extract_text_from_pdf(pdf_path, max_pages)
    
    # Extract metadata
    metadata = extract_metadata_from_text(text)
    
    # Try to find PMID and DOI
    pmid = extract_pmid_from_text(text)
    doi = extract_doi_from_text(text)
    
    if pmid:
        metadata["pmid"] = pmid
    if doi:
        metadata["doi"] = doi
    
    logger.info(f"Extracted metadata - Title: {metadata.get('title', 'N/A')[:50]}...")
    
    return metadata, text


def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text: remove excess whitespace, fix common OCR errors, etc.
    
    Args:
        text (str): Raw extracted text
    
    Returns:
        str: Cleaned text
    """
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Remove excess spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Fix common OCR errors (optional - can be extended)
    # E.g., replace "rn" with "m" if it looks wrong (careful with this)
    
    # Remove page numbers and headers/footers (optional heuristic)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just page numbers
        if not re.match(r'^-?\s*\d+\s*-?$', line.strip()):
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text.strip()
