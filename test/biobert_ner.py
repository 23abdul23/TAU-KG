import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
from lxml import etree
from transformers import AutoTokenizer, pipeline


METHOD = "biobert"
MODEL_NAME = "d4data/biomedical-ner-all"
PMC_XML_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?format=xml"
MAX_TOKENS_PER_CHUNK = 900
TOKEN_OVERLAP = 100


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
        headers={"User-Agent": "TAU-KG-NER-Benchmark/1.0"},
        timeout=60,
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


def build_chunks(text: str, tokenizer: AutoTokenizer) -> List[Dict[str, object]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = encoded["offset_mapping"]
    if not offsets:
        return []

    chunks: List[Dict[str, object]] = []
    start_index = 0
    total_tokens = len(offsets)
    while start_index < total_tokens:
        end_index = min(start_index + MAX_TOKENS_PER_CHUNK, total_tokens)
        start_char = offsets[start_index][0]
        end_char = offsets[end_index - 1][1]
        chunks.append(
            {
                "start": start_char,
                "end": end_char,
                "text": text[start_char:end_char],
            }
        )
        if end_index >= total_tokens:
            break
        next_index = max(end_index - TOKEN_OVERLAP, start_index + 1)
        start_index = next_index
    return chunks


def run_biobert(text: str) -> List[Dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ner_pipeline = pipeline("ner", model=MODEL_NAME, tokenizer=tokenizer, aggregation_strategy="simple")
    chunks = build_chunks(text, tokenizer)
    if not chunks:
        return []

    predictions = ner_pipeline([chunk["text"] for chunk in chunks], batch_size=min(4, len(chunks)))
    entities: List[Dict[str, object]] = []
    seen = set()

    for chunk, chunk_predictions in zip(chunks, predictions):
        chunk_start = int(chunk["start"])
        for item in chunk_predictions:
            start = chunk_start + int(item["start"])
            end = chunk_start + int(item["end"])
            entity = {
                "text": text[start:end],
                "label": str(item.get("entity_group", item.get("entity", "UNKNOWN"))),
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


def save_output(payload: Dict[str, object]) -> Path:
    output_path = Path(__file__).with_name(f"output_{METHOD}.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BioBERT NER on a PMC article.")
    parser.add_argument("source", help="PMCID (e.g. PMC12512994) or full PMC URL")
    args = parser.parse_args()

    started_at = time.time()
    pmcid = normalize_pmcid(args.source)
    xml_content = fetch_article_xml(pmcid)
    article_text = extract_article_text(xml_content)
    entities = run_biobert(article_text)
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
