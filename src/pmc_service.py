"""
src/pmc_service.py
==================
Robust PMC ingestion service built on official NCBI APIs.

Pipeline:
PMCID -> E-utilities metadata -> OA check -> BioC full text -> section parsing -> normalized text
"""

import json
import re
import time
import logging
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

EUTILS_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
BIOC_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

SECTION_SKIP_KEYWORDS = {"references", "acknowledgements", "acknowledgments"}
SECTION_PRIORITY = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]


class _PMCHTMLMainExtractor(HTMLParser):
    """Extract core article content from PMC HTML main/article/body tags."""

    def __init__(self):
        super().__init__()
        self.in_main = False
        self.main_depth = 0
        self.skip_depth = 0

        self.in_title = False
        self.in_heading = False
        self.in_paragraph = False

        self.title_buffer: List[str] = []
        self.heading_buffer: List[str] = []
        self.paragraph_buffer: List[str] = []

        self.current_section = "unknown"
        self.sections_map: Dict[str, List[str]] = {}

    @staticmethod
    def _attrs_to_dict(attrs) -> Dict[str, str]:
        return {k: (v or "") for k, v in attrs}

    @staticmethod
    def _should_skip_section(attrs_dict: Dict[str, str]) -> bool:
        combined = " ".join([
            attrs_dict.get("id", ""),
            attrs_dict.get("class", ""),
            attrs_dict.get("aria-label", ""),
        ]).lower()
        markers = [
            "ref-list", "references", "associated-data", "supplementary",
            "acknowledg", "copyright", "author information", "article notes",
        ]
        return any(marker in combined for marker in markers)

    def handle_starttag(self, tag, attrs):
        attrs_dict = self._attrs_to_dict(attrs)

        if tag == "main" and attrs_dict.get("id", "").lower() == "main-content":
            self.in_main = True
            self.main_depth = 1
            return

        if not self.in_main:
            return

        if tag == "main":
            self.main_depth += 1

        if self.skip_depth > 0:
            self.skip_depth += 1
            return

        if tag == "section" and self._should_skip_section(attrs_dict):
            self.skip_depth = 1
            return

        if tag == "h1":
            self.in_title = True
            self.title_buffer = []
            return

        classes = attrs_dict.get("class", "").lower()
        if tag in {"h2", "h3", "h4"} and "pmc_sec_title" in classes:
            self.in_heading = True
            self.heading_buffer = []
            return

        if tag == "p":
            self.in_paragraph = True
            self.paragraph_buffer = []

    def handle_endtag(self, tag):
        if self.in_main and tag == "main":
            self.main_depth -= 1
            if self.main_depth <= 0:
                self.in_main = False
                self.main_depth = 0
            return

        if not self.in_main:
            return

        if self.skip_depth > 0:
            self.skip_depth -= 1
            return

        if tag == "h1" and self.in_title:
            self.in_title = False
            return

        if tag in {"h2", "h3", "h4"} and self.in_heading:
            self.in_heading = False
            heading_text = " ".join(self.heading_buffer).strip()
            if heading_text:
                self.current_section = _normalize_section_type(heading_text)
            return

        if tag == "p" and self.in_paragraph:
            self.in_paragraph = False
            paragraph_text = " ".join(self.paragraph_buffer).strip()
            if paragraph_text:
                self.sections_map.setdefault(self.current_section, []).append(paragraph_text)

    def handle_data(self, data):
        if not self.in_main or self.skip_depth > 0:
            return

        clean = data.strip()
        if not clean:
            return

        if self.in_title:
            self.title_buffer.append(clean)
        elif self.in_heading:
            self.heading_buffer.append(clean)
        elif self.in_paragraph:
            self.paragraph_buffer.append(clean)

    def to_sections(self) -> List[Dict[str, str]]:
        sections: List[Dict[str, str]] = []
        for section_type, paragraphs in self.sections_map.items():
            for para in paragraphs:
                sections.append({"type": section_type, "text": para})
        return sections

    def get_title(self) -> str:
        return " ".join(self.title_buffer).strip()


def _http_get_text(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> str:
    """Fetch URL contents as text using urllib to avoid extra dependencies."""
    if params:
        query = urlencode(params)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{query}"

    request = Request(url, headers={"User-Agent": "TAU-KG/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_html_with_curl(url: str, timeout: int = 30) -> str:
    """Fetch full web page HTML with curl, with urllib fallback."""
    try:
        result = subprocess.run(
            ["curl", url],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout
    except Exception as exc:
        logger.warning("curl fetch failed for %s (%s). Falling back to urllib.", url, exc)
        return _http_get_text(url, timeout=timeout)


def extract_pmcid(source: str) -> str:
    """Extract PMCID from a PMC URL or raw PMCID text."""
    if not source:
        raise ValueError("PMC source is empty")

    match = re.search(r"PMC(\d+)", source, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract PMCID from input: {source}")

    return f"PMC{match.group(1)}"


def fetch_metadata_eutils(pmcid: str) -> Dict[str, Any]:
    """Fetch article metadata using NCBI E-utilities esummary endpoint."""
    numeric_id = pmcid.replace("PMC", "")
    body = _http_get_text(
        EUTILS_SUMMARY_URL,
        params={"db": "pmc", "id": numeric_id, "retmode": "json"},
        timeout=20,
    )

    data = json.loads(body)
    result = data.get("result", {})
    item = result.get(numeric_id, {})

    authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]

    return {
        "pmcid": pmcid,
        "title": item.get("title", ""),
        "authors": authors,
        "journal": item.get("fulljournalname", "") or item.get("source", ""),
        "pubdate": item.get("pubdate", ""),
    }


def check_open_access(pmcid: str) -> bool:
    """Check if PMCID is in Open Access / author manuscript set via OA API."""
    body = _http_get_text(PMC_OA_URL, params={"id": pmcid}, timeout=20)
    if 'status="ok"' in body:
        return True

    try:
        root = ET.fromstring(body)
        record = root.find("record")
        if record is not None and record.attrib.get("status") == "ok":
            return True
    except ET.ParseError:
        logger.warning("Could not parse OA XML for %s", pmcid)

    return False


def fetch_bioc_full_text(pmcid: str) -> Dict[str, Any]:
    """Fetch structured full text from PMC BioC API in JSON format."""
    url = BIOC_URL_TEMPLATE.format(pmcid=pmcid)
    body = _http_get_text(url, timeout=30)
    return json.loads(body)


def _normalize_section_type(raw_section: str) -> str:
    text = (raw_section or "").strip().lower()
    if not text:
        return "unknown"

    if "abstract" in text:
        return "abstract"
    if "intro" in text:
        return "introduction"
    if "method" in text or "materials" in text:
        return "methods"
    if "result" in text:
        return "results"
    if "discussion" in text:
        return "discussion"
    if "conclusion" in text:
        return "conclusion"
    if "acknowledg" in text:
        return "acknowledgements"
    if "reference" in text or text == "ref":
        return "references"

    return text


def parse_bioc_sections(bioc_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Parse BioC JSON into section list with type/text."""
    sections: List[Dict[str, str]] = []

    for document in bioc_payload.get("documents", []):
        for passage in document.get("passages", []):
            text = (passage.get("text") or "").strip()
            if not text:
                continue

            infons = passage.get("infons", {}) or {}
            raw_section = (
                infons.get("section_type")
                or infons.get("type")
                or infons.get("section")
                or "unknown"
            )
            section_type = _normalize_section_type(raw_section)

            sections.append({"type": section_type, "text": text})

    return sections


def normalize_sections_to_text(sections: List[Dict[str, str]], max_chars: Optional[int] = None) -> str:
    """Create normalized plain text while skipping low-value sections."""
    buckets: Dict[str, List[str]] = {k: [] for k in SECTION_PRIORITY}
    buckets["other"] = []

    for section in sections:
        section_type = (section.get("type") or "unknown").lower()
        text = (section.get("text") or "").strip()
        if not text:
            continue

        if any(skip in section_type for skip in SECTION_SKIP_KEYWORDS):
            continue

        if section_type in buckets:
            buckets[section_type].append(text)
        else:
            buckets["other"].append(text)

    ordered_parts: List[str] = []
    for key in SECTION_PRIORITY:
        ordered_parts.extend(buckets[key])
    ordered_parts.extend(buckets["other"])

    merged = "\n\n".join(ordered_parts)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    merged = re.sub(r"[ \t]+", " ", merged).strip()

    if max_chars:
        merged = merged[:max_chars]

    return merged


def process_pmc_url_html_fallback(url: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
    """Fallback PMC processing by downloading full HTML and extracting main article content."""
    pmcid = extract_pmcid(url)
    html = fetch_html_with_curl(url)

    extractor = _PMCHTMLMainExtractor()
    extractor.feed(html)

    sections = extractor.to_sections()
    text = normalize_sections_to_text(sections, max_chars=max_chars)

    pmid_match = re.search(r"PMID:\s*<a[^>]*>(\d{6,8})</a>", html, re.IGNORECASE)
    doi_match = re.search(r"doi:\s*<a[^>]*>(10\.\d{4,9}/[^<\s]+)</a>", html, re.IGNORECASE)

    metadata = {
        "pmcid": pmcid,
        "title": extractor.get_title(),
        "authors": [],
        "journal": "",
        "pubdate": "",
        "pmid": pmid_match.group(1) if pmid_match else "",
        "doi": doi_match.group(1).rstrip(".,;)") if doi_match else "",
        "text": text,
        "sections": sections,
        "source_url": url,
    }

    return metadata


def process_pmc_url_advanced(url: str, max_chars: Optional[int] = None, retries: int = 2) -> Dict[str, Any]:
    """Unified PMC ingestion using official APIs with OA-aware full-text retrieval."""
    pmcid = extract_pmcid(url)
    metadata = fetch_metadata_eutils(pmcid)

    oa_available = False
    sections: List[Dict[str, str]] = []
    text: Optional[str] = None

    try:
        oa_available = check_open_access(pmcid)
    except Exception as exc:
        logger.warning("OA check failed for %s: %s", pmcid, exc)

    if oa_available:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                bioc_payload = fetch_bioc_full_text(pmcid)
                sections = parse_bioc_sections(bioc_payload)
                text = normalize_sections_to_text(sections, max_chars=max_chars)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    sleep_seconds = (2 ** attempt) + 0.2
                    time.sleep(sleep_seconds)

        if last_error:
            logger.warning("BioC fetch failed for %s after retries: %s", pmcid, last_error)

    return {
        "pmcid": pmcid,
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "journal": metadata.get("journal", ""),
        "pubdate": metadata.get("pubdate", ""),
        "text": text,
        "sections": sections,
        "oa_available": oa_available,
        "source_url": url,
    }
