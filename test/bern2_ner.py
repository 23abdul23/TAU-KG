import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from lxml import etree
from dotenv import load_dotenv


METHOD = "bern2"
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

PMC_XML_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?format=xml"
HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT", "TAU-KG-NER-Benchmark/1.0").strip()
PMC_REQUEST_TIMEOUT = int(os.getenv("PMC_REQUEST_TIMEOUT", "60"))
BERN2_REQUEST_TIMEOUT = int(os.getenv("BERN2_REQUEST_TIMEOUT", "180"))
REMOTE_BERN2_URL = os.getenv("BERN2_REMOTE_URL", "http://bern2.korea.ac.kr/plain").strip()
TARGET_LABELS = {"gene", "disease", "chemical"}


def normalize_pmcid(source: str) -> str:
    value = (source or "").strip()
    if not value:
        raise ValueError("PMC source is required.")

    match = re.search(r"PMC(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"PMC{match.group(1)}"

    if not value.lower().startswith("http") and value.isdigit():
        return f"PMC{value}"

    raise ValueError(f"Could not extract PMCID from input: {source}")


def fetch_article_xml(pmcid: str) -> bytes:
    response = requests.get(
        PMC_XML_URL.format(pmcid=pmcid),
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=PMC_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.content


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_section_text(root: etree._Element, xpath: str) -> List[str]:
    sections: List[str] = []
    for node in root.xpath(xpath):
        paragraphs = node.xpath(".//*[local-name()='p']")
        if paragraphs:
            for paragraph in paragraphs:
                text = normalize_text(" ".join(paragraph.itertext()))
                if text:
                    sections.append(text)
            continue

        text = normalize_text(" ".join(node.itertext()))
        if text:
            sections.append(text)
    return sections


def extract_article_text(xml_content: bytes) -> str:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml_content, parser=parser)

    abstract_parts = extract_section_text(root, ".//*[local-name()='abstract']")
    body_parts = [
        normalize_text(" ".join(paragraph.itertext()))
        for paragraph in root.xpath(".//*[local-name()='body']//*[local-name()='p']")
    ]
    body_parts = [paragraph for paragraph in body_parts if paragraph]

    combined_parts = abstract_parts + body_parts
    if not combined_parts:
        raise ValueError("No abstract or body text found in PMC XML.")

    return "\n\n".join(combined_parts)


def bern2_endpoints() -> List[str]:
    endpoints = []
    local_bern2_url = os.getenv("BERN2_LOCAL_URL", "").strip()
    if local_bern2_url:
        endpoints.append(local_bern2_url)

    endpoints.extend(
        [
            "http://127.0.0.1:8888/plain",
            "http://localhost:8888/plain",
        ]
    )

    deduped_endpoints: List[str] = []
    for endpoint in endpoints:
        if endpoint and endpoint not in deduped_endpoints:
            deduped_endpoints.append(endpoint)

    endpoints = deduped_endpoints
    endpoints.append(REMOTE_BERN2_URL)
    return endpoints


def normalize_bern_label(raw_label: Optional[str]) -> Optional[str]:
    value = (raw_label or "").strip().lower()
    if not value:
        return None

    if value in {"gene", "gene/protein", "protein", "dna", "rna"}:
        return "gene"
    if value == "disease":
        return "disease"
    if value in {"drug", "chemical", "compound"}:
        return "chemical"
    return None


def extract_span(annotation: Dict[str, object]) -> Optional[Dict[str, int]]:
    span = annotation.get("span")
    if isinstance(span, dict):
        begin = span.get("begin")
        end = span.get("end")
        if begin is not None and end is not None:
            return {"start": int(begin), "end": int(end)}

    spans = annotation.get("spans")
    if isinstance(spans, list) and spans:
        first = spans[0]
        if isinstance(first, dict):
            begin = first.get("begin")
            end = first.get("end")
            if begin is not None and end is not None:
                return {"start": int(begin), "end": int(end)}

    if "start" in annotation and "end" in annotation:
        return {"start": int(annotation["start"]), "end": int(annotation["end"])}

    return None


def parse_bern_payload(payload: Dict[str, object], source_text: str) -> List[Dict[str, object]]:
    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("Unexpected BERN2 response format: missing annotations list.")

    entities: List[Dict[str, object]] = []
    seen = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        label = normalize_bern_label(
            str(annotation.get("obj") or annotation.get("type") or annotation.get("entity_type") or "")
        )
        if label not in TARGET_LABELS:
            continue

        span = extract_span(annotation)
        if not span:
            continue

        start = span["start"]
        end = span["end"]
        mention = str(annotation.get("mention") or source_text[start:end])
        entity = {
            "text": mention,
            "label": label,
            "start": start,
            "end": end,
        }
        key = (entity["text"], entity["label"], entity["start"], entity["end"])
        if key in seen:
            continue
        seen.add(key)
        entities.append(entity)

    entities.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return entities


def call_bern2(text: str) -> Dict[str, object]:
    errors: List[str] = []
    for endpoint in bern2_endpoints():
        try:
            response = requests.post(
                endpoint,
                data={"text": text},
                headers={"User-Agent": HTTP_USER_AGENT},
                timeout=BERN2_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("BERN2 response is not a JSON object.")
            payload["_endpoint"] = endpoint
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{endpoint}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"All BERN2 endpoints failed.\n{joined_errors}")


def save_output(payload: Dict[str, object]) -> Path:
    output_path = Path(__file__).with_name(f"output_{METHOD}.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BERN2 NER on a PMC article.")
    parser.add_argument("source", help="PMCID (e.g. PMC12512994) or full PMC URL")
    args = parser.parse_args()

    started_at = time.time()
    pmcid = normalize_pmcid(args.source)
    xml_content = fetch_article_xml(pmcid)
    article_text = extract_article_text(xml_content)
    bern_payload = call_bern2(article_text)
    entities = parse_bern_payload(bern_payload, article_text)
    runtime_seconds = round(time.time() - started_at, 4)

    payload = {
        "pmcid": pmcid,
        "method": METHOD,
        "entities": entities,
        "runtime_seconds": runtime_seconds,
    }
    output_path = save_output(payload)

    print(f"Saved {output_path}")
    print(f"Total entities: {len(entities)}")
    print(f"Runtime: {runtime_seconds:.4f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
