"""
src/pmc_service.py
==================
Robust PMC ingestion service built on official NCBI APIs.

Pipeline:
PMCID -> E-utilities metadata -> OA check -> BioC full text -> section parsing -> normalized text
"""

import gzip
import json
import logging
import random
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Optional
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

EUTILS_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
BIOC_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

SECTION_SKIP_KEYWORDS = {"references", "acknowledgements", "acknowledgments"}
SECTION_PRIORITY = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
PMC_MAX_REQUESTS_PER_SECOND = 3
PMC_FETCH_WORKER_RECOMMENDATION = 3
DEFAULT_HTTP_RETRIES = 3
DEFAULT_HTTP_TIMEOUT = 30
_NCBI_HOST_MARKERS = ("ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov")


class _PerSecondRateLimiter:
    """Thread-safe fixed-window rate limiter for NCBI/PMC calls."""

    def __init__(self, max_requests: int, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._request_times: Deque[float] = deque()

    def acquire(self) -> None:
        while True:
            sleep_seconds = 0.0
            with self._lock:
                now = time.monotonic()
                while self._request_times and (now - self._request_times[0]) >= self.window_seconds:
                    self._request_times.popleft()

                if len(self._request_times) < self.max_requests:
                    self._request_times.append(now)
                    return

                sleep_seconds = self.window_seconds - (now - self._request_times[0]) + 0.01

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


_PMC_RATE_LIMITER = _PerSecondRateLimiter(PMC_MAX_REQUESTS_PER_SECOND)


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


def _http_get_text(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> str:
    """Fetch URL contents as text with retries, gzip support, and PMC rate limiting."""
    body = _http_get_bytes(url, params=params, timeout=timeout, retries=retries)
    return body.decode("utf-8", errors="ignore")


def _build_url(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    if params:
        query = urlencode(params)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{query}"
    return url


def _should_rate_limit(url: str) -> bool:
    lowered = (url or "").lower()
    return any(marker in lowered for marker in _NCBI_HOST_MARKERS)


def _retry_delay_seconds(attempt: int, exc: Exception, base_delay: float = 0.6) -> float:
    if isinstance(exc, HTTPError):
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), base_delay)
            except ValueError:
                pass

    return (base_delay * (2 ** attempt)) + random.uniform(0.05, 0.25)


def _is_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or exc.code >= 500
    if isinstance(exc, (URLError, TimeoutError)):
        return True
    return isinstance(exc, OSError)


def _run_with_retries(
    operation: Callable[[], Any],
    description: str,
    retries: int,
    retryable: Optional[Callable[[Exception], bool]] = None,
) -> Any:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            should_retry = attempt < retries and (retryable(exc) if retryable else True)
            if not should_retry:
                raise

            delay = _retry_delay_seconds(attempt, exc)
            logger.warning(
                "%s failed (attempt %s/%s): %s. Retrying in %.2fs.",
                description,
                attempt + 1,
                retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"{description} failed without raising an explicit error")


def _http_get_bytes(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> bytes:
    final_url = _build_url(url, params=params)

    def _fetch() -> bytes:
        if _should_rate_limit(final_url):
            _PMC_RATE_LIMITER.acquire()

        request = Request(
            final_url,
            headers={
                "User-Agent": "TAU-KG/1.0",
                "Accept-Encoding": "gzip",
                "Connection": "keep-alive",
            },
        )

        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if (response.headers.get("Content-Encoding") or "").lower() == "gzip":
                payload = gzip.decompress(payload)
            return payload

    return _run_with_retries(
        _fetch,
        description=f"HTTP GET {final_url}",
        retries=retries,
        retryable=_is_retryable_http_error,
    )


def fetch_html_with_curl(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT, retries: int = DEFAULT_HTTP_RETRIES) -> str:
    """Fetch PMC HTML with urllib first and use curl only as a last-resort fallback."""
    try:
        return _http_get_text(url, timeout=timeout, retries=retries)
    except Exception as primary_exc:
        logger.warning("Direct HTML fetch failed for %s (%s). Trying curl fallback.", url, primary_exc)

    def _curl_fetch() -> str:
        if _should_rate_limit(url):
            _PMC_RATE_LIMITER.acquire()

        result = subprocess.run(
            [
                "curl",
                "-L",
                "-sS",
                "--compressed",
                "--connect-timeout",
                str(timeout),
                "--max-time",
                str(timeout),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=True,
        )
        return result.stdout

    return _run_with_retries(
        _curl_fetch,
        description=f"curl fetch {url}",
        retries=max(1, min(retries, 2)),
        retryable=lambda exc: isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError)),
    )


def extract_pmcid(source: str) -> str:
    """Extract PMCID from a PMC URL or raw PMCID text."""
    if not source:
        raise ValueError("PMC source is empty")

    match = re.search(r"PMC(\d+)", source, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract PMCID from input: {source}")

    return f"PMC{match.group(1)}"


def fetch_metadata_eutils(pmcid: str, retries: int = DEFAULT_HTTP_RETRIES) -> Dict[str, Any]:
    """Fetch article metadata using NCBI E-utilities esummary endpoint."""
    numeric_id = pmcid.replace("PMC", "")

    def _fetch() -> Dict[str, Any]:
        body = _http_get_text(
            EUTILS_SUMMARY_URL,
            params={"db": "pmc", "id": numeric_id, "retmode": "json"},
            timeout=20,
            retries=0,
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

    return _run_with_retries(
        _fetch,
        description=f"E-utilities metadata fetch {pmcid}",
        retries=retries,
        retryable=lambda exc: _is_retryable_http_error(exc) or isinstance(exc, json.JSONDecodeError),
    )


def check_open_access(pmcid: str, retries: int = DEFAULT_HTTP_RETRIES) -> bool:
    """Check if PMCID is in Open Access / author manuscript set via OA API."""

    def _fetch() -> bool:
        body = _http_get_text(PMC_OA_URL, params={"id": pmcid}, timeout=20, retries=0)
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

    return _run_with_retries(
        _fetch,
        description=f"OA check {pmcid}",
        retries=retries,
        retryable=_is_retryable_http_error,
    )


def fetch_bioc_full_text(pmcid: str, retries: int = DEFAULT_HTTP_RETRIES) -> Dict[str, Any]:
    """Fetch structured full text from PMC BioC API in JSON format."""
    url = BIOC_URL_TEMPLATE.format(pmcid=pmcid)

    def _fetch() -> Dict[str, Any]:
        body = _http_get_text(url, timeout=30, retries=0)
        stripped = body.lstrip()
        if not stripped.startswith("{") and not stripped.startswith("["):
            snippet = re.sub(r"\s+", " ", stripped[:160]).strip()
            raise ValueError(f"BioC API returned non-JSON payload for {pmcid}: {snippet!r}")
        return json.loads(body)

    return _run_with_retries(
        _fetch,
        description=f"BioC full-text fetch {pmcid}",
        retries=retries,
        retryable=lambda exc: _is_retryable_http_error(exc) or isinstance(exc, json.JSONDecodeError),
    )


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


def process_pmc_url_html_fallback(
    url: str,
    max_chars: Optional[int] = None,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> Dict[str, Any]:
    """Fallback PMC processing by downloading full HTML and extracting main article content."""
    pmcid = extract_pmcid(url)
    html = fetch_html_with_curl(url, retries=retries)

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


def process_pmc_url_advanced(
    url: str,
    max_chars: Optional[int] = None,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> Dict[str, Any]:
    """Unified PMC ingestion using official APIs with retry and PMC-aware rate limiting."""
    pmcid = extract_pmcid(url)
    metadata: Dict[str, Any] = {
        "pmcid": pmcid,
        "title": "",
        "authors": [],
        "journal": "",
        "pubdate": "",
    }

    oa_available = False
    sections: List[Dict[str, str]] = []
    text: Optional[str] = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        metadata_future = executor.submit(fetch_metadata_eutils, pmcid, retries)
        bioc_future = executor.submit(fetch_bioc_full_text, pmcid, retries)

        try:
            metadata = metadata_future.result()
        except Exception as exc:
            logger.warning("Metadata fetch failed for %s after retries: %s", pmcid, exc)

        try:
            bioc_payload = bioc_future.result()
            sections = parse_bioc_sections(bioc_payload)
            text = normalize_sections_to_text(sections, max_chars=max_chars)
            oa_available = bool(text)
        except Exception as exc:
            logger.warning("BioC fetch failed for %s after retries: %s", pmcid, exc)

    if text is None:
        try:
            oa_available = check_open_access(pmcid, retries=retries)
        except Exception as exc:
            logger.warning("OA check failed for %s after retries: %s", pmcid, exc)

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
