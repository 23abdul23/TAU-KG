import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
import spacy
from lxml import etree, html

# ---------------- CONFIG ---------------- #
MODEL_NAME = "en_ner_bionlp13cg_md"
TARGET_LABELS = {"GENE_OR_GENE_PRODUCT", "DISEASE", "SIMPLE_CHEMICAL"}
PMC_XML_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?format=xml"

CHUNK_SIZE = 50000  # safe size for spacy
USER_AGENT = "Mozilla/5.0 (compatible; PMC-NER/1.0)"

# ---------------- LOAD MODEL ONCE ---------------- #
nlp = spacy.load(MODEL_NAME)


# ---------------- UTIL ---------------- #
def normalize_pmcid(source: str) -> str:
    value = (source or "").strip()

    match = re.search(r"PMC(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"PMC{match.group(1)}"

    if value.isdigit():
        return f"PMC{value}"

    raise ValueError(f"Invalid PMCID: {source}")


# ---------------- FETCH ---------------- #
def fetch_article_xml(pmcid: str) -> bytes:
    pmcid_num = pmcid.replace("PMC", "")

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pmc",
        "id": pmcid_num,
        "retmode": "xml",
    }

    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )

    response.raise_for_status()
    content = response.content

    # ✅ Much safer validation
    if b"<article" not in content and b"<pmc-articleset" not in content:
        print(content[:500])  # debug
        raise ValueError("Invalid XML response from PMC")

    return content

# ---------------- PARSE ---------------- #
def parse_xml(content: bytes):
    try:
        return etree.fromstring(content)
    except Exception:
        return html.fromstring(content)


# ---------------- EXTRACTION ---------------- #
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(root) -> List[Dict]:
    sections = []

    sec_nodes = root.xpath(".//*[local-name()='sec']")

    for sec in sec_nodes:
        title_nodes = sec.xpath(".//*[local-name()='title']")
        title = normalize_text(" ".join(
            " ".join(node.itertext()) for node in title_nodes
        )) if title_nodes else ""

        paragraphs = sec.xpath(".//*[local-name()='p']")
        texts = []

        for p in paragraphs:
            txt = normalize_text(" ".join(p.itertext()))
            if txt:
                texts.append(txt)

        if texts:
            sections.append({
                "title": title,
                "text": "\n".join(texts)
            })

    return sections


def extract_fallback(root) -> List[Dict]:
    texts = []

    nodes = root.xpath("//*[local-name()='p']")
    for n in nodes:
        txt = normalize_text(" ".join(n.itertext()))
        if txt:
            texts.append(txt)

    return [{"title": "full_text", "text": "\n".join(texts)}]


def extract_article_text(content: bytes) -> List[Dict]:
    root = parse_xml(content)

    sections = extract_sections(root)

    if not sections:
        sections = extract_fallback(root)

    return sections


# ---------------- NER ---------------- #
def chunk_text(text: str, size: int = CHUNK_SIZE):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def run_scispacy(sections: List[Dict]) -> List[Dict]:
    results = []

    for sec in sections:
        title = sec["title"]
        text = sec["text"]

        for chunk in chunk_text(text):
            doc = nlp(chunk)

            for ent in doc.ents:
                if ent.label_ not in TARGET_LABELS:
                    continue

                results.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "section": title,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

    return results

# ---------------- Duplication ---------------- #
def deduplicate_global(entities):
    seen = set()
    unique = []

    for e in entities:
        key = (e["text"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique

# ---------------- SAVE ---------------- #
def save_output(data: Dict, pmcid: str) -> Path:
    path = Path(f"output_{pmcid}.json")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="PMCID or PMC URL")
    args = parser.parse_args()

    start = time.time()

    pmcid = normalize_pmcid(args.source)

    print(f"[INFO] Fetching {pmcid}...")
    xml_content = fetch_article_xml(pmcid)

    print("[INFO] Extracting sections...")
    sections = extract_article_text(xml_content)

    print(f"[INFO] Sections found: {len(sections)}")

    print("[INFO] Running NER...")
    entities = deduplicate_global(run_scispacy(sections))

    runtime = round(time.time() - start, 3)

    output = {
        "pmcid": pmcid,
        "sections": sections,
        "entities": entities,
        "total_entities": len(entities),
        "runtime_seconds": runtime,
    }

    path = save_output(output, pmcid)

    print(f"[DONE] Saved: {path}")
    print(f"[DONE] Entities: {len(entities)}")
    print(f"[DONE] Time: {runtime}s")


if __name__ == "__main__":
    main()