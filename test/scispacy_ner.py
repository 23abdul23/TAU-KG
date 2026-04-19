import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
import spacy
from lxml import etree


METHOD = "scispacy"
MODEL_NAME = "en_ner_bionlp13cg_md"
PMC_XML_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?format=xml"
TARGET_LABELS = {"GENE_OR_GENE_PRODUCT", "DISEASE", "SIMPLE_CHEMICAL"}


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


def run_scispacy(text: str) -> List[Dict[str, object]]:
    nlp = spacy.load(MODEL_NAME)
    nlp.max_length = max(nlp.max_length, len(text) + 1)

    entities: List[Dict[str, object]] = []
    for doc in nlp.pipe([text], batch_size=1):
        for ent in doc.ents:
            if ent.label_ not in TARGET_LABELS:
                continue
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": int(ent.start_char),
                    "end": int(ent.end_char),
                }
            )
    return entities


def save_output(payload: Dict[str, object]) -> Path:
    output_path = Path(__file__).with_name(f"output_{METHOD}.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SciSpaCy NER on a PMC article.")
    parser.add_argument("source", help="PMCID (e.g. PMC12512994) or full PMC URL")
    args = parser.parse_args()

    started_at = time.time()
    pmcid = normalize_pmcid(args.source)
    xml_content = fetch_article_xml(pmcid)
    article_text = extract_article_text(xml_content)
    entities = run_scispacy(article_text)
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
