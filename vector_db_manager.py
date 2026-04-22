import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.llm_provider import LLMClient
from src.paper_schema import normalize_entities_for_paper, normalize_relationships_for_paper, slugify_identifier

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

try:
    from citations import Citation, fetch_pubmed_citations

    CITATIONS_AVAILABLE = True
except ImportError:
    CITATIONS_AVAILABLE = False
    print("Warning: Citations module not available. Literature search disabled.")

try:
    import deb_data_papers as papers_db

    PAPERS_DB_AVAILABLE = True
except ImportError:
    PAPERS_DB_AVAILABLE = False
    papers_db = None


class VectorDBManager:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "knowledge"):
        self.db_path = self._resolve_db_path(db_path)
        self.collection_name = collection_name or "knowledge"

        self.client = chromadb.PersistentClient(path=self.db_path)

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token=hf_token)
            except TypeError:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=hf_token)
        else:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.llm_client = LLMClient()
        if self.llm_client.is_available():
            print(f"{self.llm_client.get_provider_label()} client initialized for enhanced responses.")
        else:
            print(f"Warning: LLM client unavailable ({self.llm_client.unavailable_reason}). Enhanced responses disabled.")

        self.create_knowledge_collection(self.collection_name)

    @staticmethod
    def _is_readonly_error(exc: Exception) -> bool:
        message = str(exc or "").lower()
        return "readonly" in message or "read-only" in message

    @classmethod
    def _path_is_writable(cls, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            probe_path = os.path.join(path, ".write_probe")
            with open(probe_path, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe_path)
            return True
        except Exception:
            return False

    @classmethod
    def _resolve_db_path(cls, requested_path: str) -> str:
        """
        Resolve a writable Chroma persistence path.
        Priority:
        1) CHROMA_DB_PATH env var
        2) requested path (default ./chroma_db)
        3) /tmp/chroma_db fallback
        """
        env_path = str(os.getenv("CHROMA_DB_PATH", "")).strip()
        candidates: List[str] = []
        if env_path:
            candidates.append(env_path)
        candidates.append(str(requested_path or "./chroma_db"))
        candidates.append("/tmp/chroma_db")

        for candidate in candidates:
            if cls._path_is_writable(candidate):
                if candidate != candidates[0]:
                    print(f"Chroma path fallback selected: {candidate}")
                return candidate

        # Last-resort return requested path; downstream error will include details.
        return str(requested_path or "./chroma_db")

    def refresh_llm_client(self, provider: Optional[str] = None) -> None:
        """Refresh the active LLM client after runtime provider changes."""
        if provider:
            os.environ["LLM_PROVIDER"] = str(provider).strip()

        self.llm_client = LLMClient()
        if self.llm_client.is_available():
            print(f"{self.llm_client.get_provider_label()} client initialized for enhanced responses.")
        else:
            print(f"Warning: LLM client unavailable ({self.llm_client.unavailable_reason}). Enhanced responses disabled.")

    @staticmethod
    def _distance_to_similarity(distance: Any) -> float:
        try:
            numeric_distance = max(float(distance), 0.0)
        except (TypeError, ValueError):
            return 0.0
        return 100.0 / (1.0 + numeric_distance)

    @staticmethod
    def _stable_numeric_id(seed: str) -> int:
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return int(digest, 16)

    @staticmethod
    def _normalize_label_filter(values: Optional[List[str]]) -> Optional[set]:
        if not values:
            return None
        normalized = {str(value).strip().lower() for value in values if str(value).strip()}
        return normalized or None

    @staticmethod
    def _normalize_entity_label(value: Any) -> str:
        type_key = VectorDBManager._normalize_entity_type_key(value)
        if type_key == "gene/protein":
            return "Gene/Protein"
        if type_key == "disease":
            return "Disease"
        if type_key == "pathway":
            return "Pathway"
        if type_key == "paper":
            return "Paper"
        raw = re.sub(r"\s+", " ", str(value or "").strip())
        if not raw:
            return "Entity"
        return raw

    @staticmethod
    def _normalize_entity_type_key(value: Any) -> str:
        raw = re.sub(r"[\s_\-]+", "", str(value or "").strip().lower())
        if raw in {"gene", "genes", "protein", "proteins", "geneprotein", "gene/protein"}:
            return "gene/protein"
        if raw in {"disease", "diseases"}:
            return "disease"
        if raw in {"pathway", "pathways"}:
            return "pathway"
        if raw == "paper":
            return "paper"
        if raw == "entity":
            return "entity"
        return re.sub(r"\s+", "", str(value or "").strip().lower())

    @staticmethod
    def _paper_sections_text(paper: Dict[str, Any], limit: int = 700) -> str:
        sections = paper.get("sections", [])
        if not isinstance(sections, list):
            return ""

        parts: List[str] = []
        for section in sections[:6]:
            if not isinstance(section, dict):
                continue
            heading = str(section.get("heading", "")).strip()
            text = str(section.get("text", "")).strip()
            if heading and text:
                parts.append(f"{heading}: {text}")
            elif text:
                parts.append(text)

        return " ".join(parts)[:limit]

    def _hydrate_paper_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        hydrated = dict(metadata or {})
        authors_raw = hydrated.get("authors", "")
        if isinstance(authors_raw, str):
            hydrated["authors"] = [author for author in authors_raw.split("|") if author]
        elif not isinstance(authors_raw, list):
            hydrated["authors"] = []

        paper_id = str(hydrated.get("paper_id", "")).strip()
        if PAPERS_DB_AVAILABLE and paper_id:
            paper_record = papers_db.get_paper_by_id(paper_id)
            if paper_record:
                hydrated["title"] = paper_record.get("title") or hydrated.get("title", "")
                hydrated["authors"] = paper_record.get("authors") or hydrated.get("authors", [])
                hydrated["abstract"] = paper_record.get("abstract", "")
                hydrated["doi"] = paper_record.get("doi") or hydrated.get("doi", "")
                hydrated["pmid"] = paper_record.get("pmid") or hydrated.get("pmid", "")
                hydrated["publication_date"] = paper_record.get("publication_date") or hydrated.get("publication_date", "")
                hydrated["source_url"] = paper_record.get("source_url", "")

        publication_date = str(hydrated.get("publication_date", "") or "")
        hydrated["publication_year"] = publication_date[:4] if len(publication_date) >= 4 else ""
        return hydrated

    def _build_paper_document_text(self, paper: Dict[str, Any]) -> str:
        parts = [
            f"Paper: {str(paper.get('title', '')).strip()}",
            f"Abstract: {str(paper.get('abstract', '')).strip()[:1200]}",
            f"Authors: {', '.join(paper.get('authors', []))[:400]}",
            f"PMID: {str(paper.get('pmid', '')).strip()}",
            f"DOI: {str(paper.get('doi', '')).strip()}",
        ]
        sections_text = self._paper_sections_text(paper)
        if sections_text:
            parts.append(f"Sections: {sections_text}")
        return " | ".join(part for part in parts if part and not part.endswith(": "))

    def _build_curated_entity_document_text(self, row: Dict[str, Any]) -> str:
        return " | ".join(
            [
                f"Entity: {row.get('node_name', '')}",
                f"Type: {row.get('node_type', 'entity')}",
                f"Source: {row.get('node_source', 'CURATED_GRAPH')}",
                f"Node ID: {row.get('node_id', '')}",
            ]
        )

    def _build_paper_entity_document_text(self, paper: Dict[str, Any], entity_bucket: str, entity: Dict[str, Any]) -> str:
        entity_name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("entity_type", entity_bucket)).strip()
        context = str(entity.get("context", "")).strip()
        chromosome = str(entity.get("chromosome", "")).strip()
        paper_title = str(paper.get("title", "")).strip()
        paper_abstract = str(paper.get("abstract", "")).strip()

        parts = [
            f"Entity: {entity_name}",
            f"Type: {entity_type}",
            f"Paper: {paper_title}",
        ]
        if chromosome:
            parts.append(f"Chromosome: {chromosome}")
        if context:
            parts.append(f"Context: {context[:400]}")
        elif paper_abstract:
            parts.append(f"Paper abstract: {paper_abstract[:300]}")
        return " | ".join(part for part in parts if part and not part.endswith(": "))

    def _build_relationship_document_text(self, paper: Dict[str, Any], relationship: Dict[str, Any]) -> str:
        parts = [
            (
                f"Relationship: {relationship.get('source_name', '')} "
                f"[{relationship.get('edge_type', 'ASSOCIATES')}] "
                f"{relationship.get('target_name', '')}"
            ),
            f"Source type: {relationship.get('source_type', 'Entity')}",
            f"Target type: {relationship.get('target_type', 'Entity')}",
            f"Paper: {str(paper.get('title', '')).strip()}",
        ]
        evidence = str(relationship.get("evidence", "")).strip()
        if evidence:
            parts.append(f"Evidence: {evidence[:500]}")
        return " | ".join(part for part in parts if part and not part.endswith(": "))

    def _build_curated_entity_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        node_name = str(row.get("node_name", "")).strip()
        node_source = str(row.get("node_source", "CURATED_GRAPH")).strip() or "CURATED_GRAPH"
        node_type = self._normalize_entity_type_key(row.get("node_type", "entity")) or "entity"
        record_id = f"curated_{slugify_identifier(str(row.get('node_index', node_name)))}"
        return {
            "record_kind": "entity",
            "record_id": record_id,
            "approved": True,
            "source_system": "CURATED_GRAPH",
            "entity_id": str(row.get("node_id", node_name)),
            "entity_name": node_name,
            "entity_type": self._normalize_entity_label(node_type),
            "confidence": 1.0,
            "chromosome": "",
            "paper_id": "",
            "node_index": int(row.get("node_index", self._stable_numeric_id(record_id))),
            "node_id": str(row.get("node_id", node_name)),
            "node_type": node_type,
            "node_name": node_name,
            "node_source": node_source,
            "source_paper_id": "",
        }

    def _build_paper_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        paper_id = str(paper.get("paper_id", "")).strip()
        return {
            "record_kind": "paper",
            "record_id": f"paper_{paper_id}",
            "approved": True,
            "source_system": "PAPER_INGEST",
            "paper_id": paper_id,
            "title": str(paper.get("title", "")),
            "authors": "|".join(paper.get("authors", []))[:500],
            "pmid": str(paper.get("pmid", "")),
            "doi": str(paper.get("doi", "")),
            "publication_date": str(paper.get("publication_date", "")),
            "upload_date": str(paper.get("upload_date", "")),
            "extraction_status": str(paper.get("extraction_status", "pending")),
            "source_url": str(paper.get("source_url", "")),
        }

    def _build_paper_entity_metadata(self, paper_id: str, entity_bucket: str, entity: Dict[str, Any]) -> Dict[str, Any]:
        entity_name = str(entity.get("name", "")).strip()
        entity_id = str(entity.get("id", "")).strip() or f"entity_{slugify_identifier(entity_name)}"
        legacy_node_type = self._normalize_entity_type_key(entity.get("entity_type", entity_bucket))
        entity_type = self._normalize_entity_label(legacy_node_type)
        record_id = entity_id
        return {
            "record_kind": "entity",
            "record_id": record_id,
            "approved": bool(entity.get("approved", False)),
            "source_system": "PAPER_INGEST",
            "paper_id": str(paper_id),
            "entity_id": entity_id,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "confidence": float(entity.get("confidence", 0.0)),
            "chromosome": str(entity.get("chromosome", "")),
            "node_index": self._stable_numeric_id(record_id),
            "node_id": entity_id,
            "node_type": legacy_node_type,
            "node_name": entity_name,
            "node_source": "PAPER_INGEST",
            "source_paper_id": str(paper_id),
            "mapped_to_existing": str(entity.get("mapped_to_existing", "")),
        }

    def _build_relationship_metadata(self, paper_id: str, relationship: Dict[str, Any]) -> Dict[str, Any]:
        edge_id = str(relationship.get("id", "")).strip() or f"edge_{paper_id}_{slugify_identifier(str(relationship.get('source_name', '')))}_{slugify_identifier(str(relationship.get('target_name', '')))}"
        return {
            "record_kind": "relationship",
            "record_id": edge_id,
            "approved": bool(relationship.get("approved", False)),
            "source_system": "PAPER_INGEST",
            "paper_id": str(paper_id),
            "edge_id": edge_id,
            "source_id": str(relationship.get("source_id", "")),
            "source_name": str(relationship.get("source_name", "")),
            "source_type": self._normalize_entity_label(relationship.get("source_type", "Entity")),
            "target_id": str(relationship.get("target_id", "")),
            "target_name": str(relationship.get("target_name", "")),
            "target_type": self._normalize_entity_label(relationship.get("target_type", "Entity")),
            "edge_type": str(relationship.get("edge_type", "ASSOCIATES")),
            "edge_weight": float(relationship.get("edge_weight", 0.0)),
            "evidence": str(relationship.get("evidence", "")),
            "original_relation": str(relationship.get("original_relation", "")),
            "extraction_method": str(relationship.get("extraction_method", "")),
            "source_chromosome": str(relationship.get("source_chromosome", "")),
        }

    def _upsert_records(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0

        documents = [record["document"] for record in records]
        metadatas = [record["metadata"] for record in records]
        ids = [record["id"] for record in records]
        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return len(records)

    def _clear_collection(self) -> None:
        existing = self.collection.get()
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def _grouped_paper_entities(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        if not PAPERS_DB_AVAILABLE:
            return {}
        return {
            str(paper_id): normalize_entities_for_paper(str(paper_id), entities)
            for paper_id, entities in getattr(papers_db, "paper_entities", {}).items()
        }

    def create_knowledge_collection(self, collection_name: str = "knowledge") -> None:
        self.collection_name = collection_name or "knowledge"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            try:
                self.collection = self.client.create_collection(name=self.collection_name)
                print(f"Created new collection: {self.collection_name}")
            except Exception as exc:
                if self._is_readonly_error(exc):
                    fallback_path = self._resolve_db_path("/tmp/chroma_db")
                    if fallback_path != self.db_path:
                        print(
                            f"Primary Chroma path '{self.db_path}' is read-only; "
                            f"retrying with writable fallback '{fallback_path}'."
                        )
                        self.db_path = fallback_path
                        self.client = chromadb.PersistentClient(path=self.db_path)
                        try:
                            self.collection = self.client.get_collection(name=self.collection_name)
                            print(f"Loaded existing collection from fallback path: {self.collection_name}")
                            return
                        except Exception:
                            self.collection = self.client.create_collection(name=self.collection_name)
                            print(f"Created new collection at fallback path: {self.collection_name}")
                            return
                raise

    def upsert_curated_nodes_to_knowledge(self, csv_path: str = "nodes_main.csv", batch_size: int = 500) -> Dict[str, int]:
        if not os.path.exists(csv_path):
            return {"upserted": 0}

        df = pd.read_csv(csv_path)
        if df.empty:
            return {"upserted": 0}

        try:
            self.collection.delete(where={"source_system": "CURATED_GRAPH"})
        except Exception:
            pass

        processed = 0
        with tqdm(total=len(df), desc="Indexing curated nodes", unit="rows") as pbar:
            for start_idx in range(0, len(df), batch_size):
                batch_df = df.iloc[start_idx : start_idx + batch_size]
                records: List[Dict[str, Any]] = []
                for _, row in batch_df.iterrows():
                    row_dict = row.to_dict()
                    metadata = self._build_curated_entity_metadata(row_dict)
                    records.append(
                        {
                            "id": metadata["record_id"],
                            "document": self._build_curated_entity_document_text(row_dict),
                            "metadata": metadata,
                        }
                    )
                processed += self._upsert_records(records)
                pbar.update(len(batch_df))
                time.sleep(0.01)

        return {"upserted": processed}

    def delete_paper_records_from_knowledge(self, paper_id: str) -> Dict[str, int]:
        normalized_paper_id = str(paper_id or "").strip()
        if not normalized_paper_id:
            return {"deleted": 0}

        try:
            self.collection.delete(where={"paper_id": normalized_paper_id})
        except Exception:
            matching_ids = [
                item_id
                for item_id, metadata in zip(
                    self.collection.get().get("ids", []),
                    self.collection.get().get("metadatas", []),
                )
                if str((metadata or {}).get("paper_id", "")).strip() == normalized_paper_id
            ]
            if matching_ids:
                self.collection.delete(ids=matching_ids)
                return {"deleted": len(matching_ids)}
            return {"deleted": 0}

        return {"deleted": 1}

    def upsert_paper_records_to_knowledge(self, paper_id: str, include_pending: bool = False) -> Dict[str, int]:
        if not PAPERS_DB_AVAILABLE:
            return {"papers": 0, "entities": 0, "relationships": 0}

        normalized_paper_id = str(paper_id or "").strip()
        if not normalized_paper_id:
            return {"papers": 0, "entities": 0, "relationships": 0}

        paper = papers_db.get_paper_by_id(normalized_paper_id)
        if not paper:
            self.delete_paper_records_from_knowledge(normalized_paper_id)
            return {"papers": 0, "entities": 0, "relationships": 0}

        self.delete_paper_records_from_knowledge(normalized_paper_id)

        normalized_entities = normalize_entities_for_paper(
            normalized_paper_id,
            papers_db.get_paper_entities(normalized_paper_id) or {},
        )
        normalized_relationships = normalize_relationships_for_paper(
            normalized_paper_id,
            papers_db.get_paper_relationships(normalized_paper_id) or [],
            normalized_entities,
        )

        records: List[Dict[str, Any]] = []
        paper_metadata = self._build_paper_metadata(paper)
        records.append(
            {
                "id": paper_metadata["record_id"],
                "document": self._build_paper_document_text(paper),
                "metadata": paper_metadata,
            }
        )

        entity_count = 0
        for entity_bucket, entities in normalized_entities.items():
            if not isinstance(entities, list):
                continue
            for entity in entities:
                if not include_pending and not entity.get("approved", False):
                    continue
                metadata = self._build_paper_entity_metadata(normalized_paper_id, entity_bucket, entity)
                records.append(
                    {
                        "id": metadata["record_id"],
                        "document": self._build_paper_entity_document_text(paper, entity_bucket, entity),
                        "metadata": metadata,
                    }
                )
                entity_count += 1

        relationship_count = 0
        for relationship in normalized_relationships:
            if not include_pending and not relationship.get("approved", False):
                continue
            metadata = self._build_relationship_metadata(normalized_paper_id, relationship)
            records.append(
                {
                    "id": metadata["record_id"],
                    "document": self._build_relationship_document_text(paper, relationship),
                    "metadata": metadata,
                }
            )
            relationship_count += 1

        self._upsert_records(records)
        return {"papers": 1, "entities": entity_count, "relationships": relationship_count}

    def rebuild_knowledge_index_from_store(self, include_curated: bool = True) -> Dict[str, int]:
        self.create_knowledge_collection(self.collection_name)
        self._clear_collection()

        stats = {"curated_entities": 0, "papers": 0, "entities": 0, "relationships": 0}
        if include_curated:
            curated_stats = self.upsert_curated_nodes_to_knowledge()
            stats["curated_entities"] = curated_stats.get("upserted", 0)

        if not PAPERS_DB_AVAILABLE:
            return stats

        for paper in getattr(papers_db, "papers_data", []):
            paper_id = str(paper.get("paper_id", "")).strip()
            if not paper_id:
                continue
            result = self.upsert_paper_records_to_knowledge(paper_id, include_pending=False)
            stats["papers"] += result.get("papers", 0)
            stats["entities"] += result.get("entities", 0)
            stats["relationships"] += result.get("relationships", 0)

        return stats

    def load_csv_to_vectordb(self, csv_path: str, batch_size: int = 100) -> None:
        result = self.upsert_curated_nodes_to_knowledge(csv_path=csv_path, batch_size=max(batch_size, 1))
        print(f"Successfully loaded {result.get('upserted', 0)} curated nodes into the vector database.")

    def search_knowledge(
        self,
        query: str,
        n_results: int = 20,
        record_kinds: Optional[List[str]] = None,
        approved_only: bool = True,
        entity_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        source_systems: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=max(n_results * 3, n_results))

        record_kind_filter = self._normalize_label_filter(record_kinds)
        entity_type_filter = self._normalize_label_filter(entity_types)
        edge_type_filter = self._normalize_label_filter(edge_types)
        source_system_filter = self._normalize_label_filter(source_systems)

        formatted_results: List[Dict[str, Any]] = []
        if results.get("documents") and results["documents"][0]:
            for index in range(len(results["documents"][0])):
                metadata = dict(results["metadatas"][0][index])
                record_kind = str(metadata.get("record_kind", "")).strip().lower()
                source_system = str(metadata.get("source_system", "")).strip().lower()
                entity_type = str(metadata.get("entity_type", metadata.get("node_type", ""))).strip().lower()
                edge_type = str(metadata.get("edge_type", "")).strip().lower()

                if record_kind_filter and record_kind not in record_kind_filter:
                    continue
                if source_system_filter and source_system not in source_system_filter:
                    continue
                if entity_type_filter and entity_type not in entity_type_filter:
                    continue
                if edge_type_filter and edge_type not in edge_type_filter:
                    continue

                if approved_only and record_kind in {"entity", "relationship"} and source_system == "paper_ingest":
                    if not bool(metadata.get("approved", False)):
                        continue

                distance = results["distances"][0][index] if results.get("distances") else None
                formatted_results.append(
                    {
                        "document": results["documents"][0][index],
                        "metadata": metadata,
                        "distance": distance,
                        "similarity_score": self._distance_to_similarity(distance),
                    }
                )
                if len(formatted_results) >= n_results:
                    break

        return formatted_results

    def search_similar(self, query: str, n_results: int = 15) -> List[Dict[str, Any]]:
        return self.search_knowledge(query, n_results=n_results, record_kinds=["entity"], approved_only=True)

    def search_entities_by_type(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        node_sources: Optional[List[str]] = None,
        n_results: int = 15,
    ) -> List[Dict[str, Any]]:
        return self.search_knowledge(
            query,
            n_results=n_results,
            record_kinds=["entity"],
            approved_only=True,
            entity_types=entity_types,
            source_systems=node_sources,
        )

    def search_relationships(
        self,
        query: str,
        n_results: int = 10,
        edge_types: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        target_types: Optional[List[str]] = None,
        approved_only: bool = True,
    ) -> List[Dict[str, Any]]:
        source_type_filter = self._normalize_label_filter(source_types)
        target_type_filter = self._normalize_label_filter(target_types)

        results = self.search_knowledge(
            query,
            n_results=max(n_results * 2, n_results),
            record_kinds=["relationship"],
            approved_only=approved_only,
            edge_types=edge_types,
        )

        formatted_results: List[Dict[str, Any]] = []
        for result in results:
            metadata = dict(result.get("metadata", {}))
            source_type = str(metadata.get("source_type", "")).strip().lower()
            target_type = str(metadata.get("target_type", "")).strip().lower()
            if source_type_filter and source_type not in source_type_filter:
                continue
            if target_type_filter and target_type not in target_type_filter:
                continue

            formatted_results.append(
                {
                    "document": result.get("document", ""),
                    "metadata": metadata,
                    "distance": result.get("distance"),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "source_name": metadata.get("source_name", ""),
                    "source_type": metadata.get("source_type", ""),
                    "target_name": metadata.get("target_name", ""),
                    "target_type": metadata.get("target_type", ""),
                    "edge_type": metadata.get("edge_type", ""),
                    "edge_weight": metadata.get("edge_weight", 0.0),
                    "evidence": metadata.get("evidence", ""),
                    "paper_id": metadata.get("paper_id", ""),
                }
            )
            if len(formatted_results) >= n_results:
                break

        return formatted_results

    def search_papers(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.search_knowledge(query, n_results=n_results, record_kinds=["paper"], approved_only=False)
        except Exception as exc:
            print(f"Error searching papers: {exc}")
            return []

        formatted_results: List[Dict[str, Any]] = []
        for result in results:
            metadata = self._hydrate_paper_metadata(result.get("metadata", {}))
            formatted_results.append(
                {
                    "document": result.get("document", ""),
                    "metadata": metadata,
                    "distance": result.get("distance"),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "paper_id": metadata.get("paper_id", ""),
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", []),
                    "pmid": metadata.get("pmid", ""),
                    "doi": metadata.get("doi", ""),
                    "publication_date": metadata.get("publication_date", ""),
                    "publication_year": metadata.get("publication_year", ""),
                    "abstract": metadata.get("abstract", ""),
                    "source_url": metadata.get("source_url", ""),
                }
            )
        return formatted_results

    def get_all_gene_names(self) -> List[str]:
        all_results = self.collection.get()
        gene_names = [
            str(metadata.get("node_name", metadata.get("entity_name", ""))).strip()
            for metadata in all_results.get("metadatas", [])
            if str((metadata or {}).get("record_kind", "")).strip().lower() == "entity"
        ]
        return sorted({name for name in gene_names if name})

    def get_gene_info(self, gene_name: str) -> Optional[Dict[str, Any]]:
        results = self.search_knowledge(gene_name, n_results=1, record_kinds=["entity"], approved_only=True)
        if results:
            return {"document": results[0]["document"], "metadata": results[0]["metadata"]}
        return None

    def get_database_stats(self) -> Dict[str, Any]:
        all_results = self.collection.get()
        metadatas = all_results.get("metadatas", [])
        stats = {
            "total_documents": len(all_results.get("documents", [])),
            "sources": {},
            "node_types": {},
            "record_kinds": {},
            "papers_documents": 0,
            "paper_entities_documents": 0,
            "paper_edges_documents": 0,
            "curated_entities_documents": 0,
            "paper_ingest_nodes_total": 0,
            "paper_ingest_edges_total": 0,
            "paper_ingest_node_types": {},
            "paper_ingest_edge_types": {},
        }

        for metadata in metadatas:
            metadata = metadata or {}
            source = str(metadata.get("source_system", metadata.get("node_source", "UNKNOWN")))
            normalized_source = source.strip().upper()
            node_type = str(metadata.get("node_type", metadata.get("entity_type", "entity")))
            record_kind = str(metadata.get("record_kind", "unknown"))
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            stats["record_kinds"][record_kind] = stats["record_kinds"].get(record_kind, 0) + 1

            if record_kind == "paper":
                stats["papers_documents"] += 1
            elif record_kind == "relationship":
                stats["paper_edges_documents"] += 1
            elif record_kind == "entity" and source == "CURATED_GRAPH":
                stats["curated_entities_documents"] += 1
            elif record_kind == "entity":
                stats["paper_entities_documents"] += 1

            if normalized_source == "PAPER_INGEST":
                if record_kind == "entity":
                    stats["paper_ingest_nodes_total"] += 1
                    stats["paper_ingest_node_types"][node_type] = (
                        stats["paper_ingest_node_types"].get(node_type, 0) + 1
                    )
                elif record_kind == "relationship":
                    stats["paper_ingest_edges_total"] += 1
                    edge_type = str(metadata.get("edge_type", "ASSOCIATES")).strip() or "ASSOCIATES"
                    stats["paper_ingest_edge_types"][edge_type] = (
                        stats["paper_ingest_edge_types"].get(edge_type, 0) + 1
                    )

        stats["total_documents_all_collections"] = stats["total_documents"]
        return stats

    def sync_paper_entities_to_main_collection(
        self,
        paper_entities: Dict[str, Any],
        source_name: str = "PAPER_INGEST",
        only_approved: bool = False,
    ) -> Dict[str, int]:
        total_candidates = 0
        added_by_type: Dict[str, int] = {}

        for paper_id, entities in (paper_entities or {}).items():
            normalized_entities = normalize_entities_for_paper(str(paper_id), entities)
            for entity_bucket, values in normalized_entities.items():
                for entity in values:
                    if only_approved and not entity.get("approved", False):
                        continue
                    total_candidates += 1
                    entity_type_key = str(entity.get("entity_type", entity_bucket)).strip().lower() or entity_bucket
                    added_by_type[entity_type_key] = added_by_type.get(entity_type_key, 0) + 1
            self.upsert_paper_records_to_knowledge(str(paper_id), include_pending=not only_approved)

        return {
            "total_candidates": total_candidates,
            "added": total_candidates,
            "skipped": 0,
            "added_by_type": added_by_type,
            "candidates_by_type": dict(added_by_type),
            "source_name": source_name,
        }

    def create_papers_collection(self, collection_name: str = "knowledge") -> None:
        self.create_knowledge_collection(collection_name)

    def create_paper_edges_collection(self, collection_name: str = "knowledge") -> None:
        self.create_knowledge_collection(collection_name)

    def _get_or_create_collection(self, collection_name: str):
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return self.client.create_collection(name=collection_name)

    def upsert_paper_to_vectordb(self, paper: Dict[str, Any], collection_name: str = "knowledge") -> Dict[str, int]:
        del collection_name
        paper_id = str((paper or {}).get("paper_id", "")).strip()
        if not paper_id:
            return {"upserted": 0}
        result = self.upsert_paper_records_to_knowledge(paper_id, include_pending=False)
        return {"upserted": result.get("papers", 0)}

    def load_papers_to_vectordb(self, papers_data: List[Dict[str, Any]], batch_size: int = 50) -> None:
        del batch_size
        if not papers_data:
            print("No papers to load.")
            return
        processed = 0
        for paper in papers_data:
            paper_id = str(paper.get("paper_id", "")).strip()
            if not paper_id:
                continue
            processed += self.upsert_paper_records_to_knowledge(paper_id, include_pending=False).get("papers", 0)
        print(f"Successfully loaded {processed} papers into the vector database.")

    def upsert_paper_entities_to_vectordb(
        self,
        paper_id: str,
        entities: Dict[str, Any],
        collection_name: str = "knowledge",
    ) -> Dict[str, int]:
        del entities, collection_name
        result = self.upsert_paper_records_to_knowledge(paper_id, include_pending=True)
        return {"upserted": result.get("entities", 0)}

    def load_paper_entities_to_vectordb(self, paper_entities: Dict[str, Any], collection_name: str = "knowledge") -> None:
        del collection_name
        processed = 0
        for paper_id in (paper_entities or {}).keys():
            processed += self.upsert_paper_records_to_knowledge(str(paper_id), include_pending=True).get("entities", 0)
        print(f"Successfully loaded {processed} entities into the vector database.")

    def sync_paper_edges_to_vectordb(
        self,
        paper_edges: List[Dict[str, Any]],
        only_approved: bool = True,
        collection_name: str = "knowledge",
    ) -> Dict[str, int]:
        del collection_name
        paper_ids = {
            str(edge.get("paper_id", "")).strip()
            for edge in (paper_edges or [])
            if isinstance(edge, dict) and str(edge.get("paper_id", "")).strip()
        }
        upserted = 0
        for paper_id in paper_ids:
            upserted += self.upsert_paper_records_to_knowledge(paper_id, include_pending=not only_approved).get("relationships", 0)
        return {"total_candidates": len(paper_edges or []), "upserted": upserted}

    def load_paper_edges_to_vectordb(
        self,
        paper_edges: List[Dict[str, Any]],
        collection_name: str = "knowledge",
        only_approved: bool = True,
    ) -> None:
        result = self.sync_paper_edges_to_vectordb(
            paper_edges=paper_edges,
            only_approved=only_approved,
            collection_name=collection_name,
        )
        print(f"Successfully loaded {result.get('upserted', 0)} paper relationships into the vector database.")

    def _format_context_for_gpt(self, search_results: List[Dict[str, Any]]) -> str:
        if not search_results:
            return "No relevant knowledge records found in the database."

        context_parts = []
        for index, result in enumerate(search_results, 1):
            metadata = result.get("metadata", {})
            record_kind = metadata.get("record_kind", "entity")
            relevance = result.get("similarity_score", self._distance_to_similarity(result.get("distance")))
            if record_kind == "paper":
                context_parts.append(
                    (
                        f"Paper {index}:\n"
                        f"- Title: {metadata.get('title', '')}\n"
                        f"- PMID: {metadata.get('pmid', '')}\n"
                        f"- Publication Date: {metadata.get('publication_date', '')}\n"
                        f"- Relevance: {relevance:.1f}%"
                    )
                )
            elif record_kind == "relationship":
                context_parts.append(
                    (
                        f"Relationship {index}:\n"
                        f"- Source: {metadata.get('source_name', '')} ({metadata.get('source_type', '')})\n"
                        f"- Edge: {metadata.get('edge_type', '')}\n"
                        f"- Target: {metadata.get('target_name', '')} ({metadata.get('target_type', '')})\n"
                        f"- Evidence: {str(metadata.get('evidence', ''))[:240]}\n"
                        f"- Relevance: {relevance:.1f}%"
                    )
                )
            else:
                context_parts.append(
                    (
                        f"Entity {index}:\n"
                        f"- Name: {metadata.get('node_name', metadata.get('entity_name', ''))}\n"
                        f"- ID: {metadata.get('node_id', metadata.get('entity_id', ''))}\n"
                        f"- Type: {metadata.get('node_type', metadata.get('entity_type', ''))}\n"
                        f"- Source: {metadata.get('node_source', metadata.get('source_system', ''))}\n"
                        f"- Relevance: {relevance:.1f}%"
                    )
                )

        return "\n\n".join(context_parts)

    def _get_gpt_response(self, query: str, context: str, citations_context: str = "", max_tokens: int = 1024) -> str:
        if not self.llm_client.is_available():
            return "Enhanced responses not available (LLM client not initialized)."

        user_prompt = f"""You are a concise biomedical AI assistant. Answer the user's question using the supplied knowledge context.

RULES:
1. Be concise and specific.
2. Prefer retrieved evidence over unsupported claims.
3. Distinguish entities, relationships, and supporting papers when useful.
4. Cite literature context when relevant.

KNOWLEDGE CONTEXT:
{context}

LITERATURE CONTEXT:
{citations_context}

USER QUERY: {query}
"""

        try:
            response = self.llm_client.generate_text(
                system_prompt="You are a concise biomedical AI assistant.",
                user_prompt=user_prompt,
                model=os.getenv("CHAT_LLM_MODEL", ""),
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.text.strip()
        except Exception as exc:
            return f"Error generating enhanced response: {exc}"

    def _get_pubmed_citations(self, query: str, max_citations: int = 5) -> List[Citation]:
        if not CITATIONS_AVAILABLE:
            return []

        try:
            return fetch_pubmed_citations(
                query=query,
                max_citations=max_citations,
                prioritize_reviews=True,
                max_age_years=5,
            )
        except Exception as exc:
            print(f"Error fetching citations: {exc}")
            return []

    def _format_citations_for_gpt(self, citations: List[Citation]) -> str:
        if not citations:
            return "No recent literature found."

        citations_parts = []
        for index, citation in enumerate(citations, 1):
            citations_parts.append(
                (
                    f"Citation {index}:\n"
                    f"- Title: {citation.title}\n"
                    f"- Authors: {citation.authors if citation.authors else 'N/A'}\n"
                    f"- Journal: {citation.journal} ({citation.year if citation.year else 'N/A'})\n"
                    f"- PMID: {citation.pmid if citation.pmid else 'N/A'}"
                )
            )
        return "\n\n".join(citations_parts)

    def generate_enhanced_response(self, query: str, search_results: List[Dict[str, Any]], max_tokens: int = 1024) -> Dict[str, Any]:
        context = self._format_context_for_gpt(search_results)
        citations = self._get_pubmed_citations(query, max_citations=5)
        citations_context = self._format_citations_for_gpt(citations)
        gpt_response = self._get_gpt_response(query, context, citations_context, max_tokens)

        return {
            "gpt_response": gpt_response,
            "citations": citations,
            "search_results": search_results,
            "context_used": context,
            "citations_context_used": citations_context,
            "has_openai": self.llm_client.is_available(),
            "has_llm": self.llm_client.is_available(),
            "llm_provider": self.llm_client.provider,
            "has_citations": CITATIONS_AVAILABLE,
        }


if __name__ == "__main__":
    db_manager = VectorDBManager()
    db_manager.rebuild_knowledge_index_from_store(include_curated=True)
    results = db_manager.search_knowledge("protein kinase", n_results=3)
    print("\nSearch results for 'protein kinase':")
    for result in results:
        print(f"- {result['document']}")
        print(f"  Distance: {result['distance']:.4f}" if result.get("distance") is not None else "  Distance: N/A")

    stats = db_manager.get_database_stats()
    print("\nDatabase Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Sources: {stats['sources']}")
    print(f"Node types: {stats['node_types']}")
