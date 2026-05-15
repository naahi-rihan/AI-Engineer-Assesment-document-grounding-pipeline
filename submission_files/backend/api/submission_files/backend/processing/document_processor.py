"""
document_processor.py
---------------------
Ingests messy legal documents (text, PDF, scanned) and produces:
- clean extracted text
- structured metadata fields
- chunked passages ready for retrieval
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

# ── helpers ──────────────────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    r"\[unclear section.*?\]",
    r"\[illegible.*?\]",
    r"\[SCAN.*?\]",
]

def _strip_noise_markers(text: str) -> str:
    """Replace illegibility markers with a placeholder so retrieval is honest."""
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, "[TEXT UNCLEAR]", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _clean_text(raw: str) -> str:
    """Basic cleaning: normalise whitespace, remove stray control chars."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse excessive blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)           # collapse repeated spaces/tabs
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)  # drop ctrl chars
    return text.strip()


def _extract_structured_fields(text: str, filename: str) -> dict:
    """
    Heuristic extraction of common legal document fields.
    Returns a dict; missing fields are None.
    """

    def find(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    case_id   = find(r"Case ID[:\s]+([A-Z0-9\-]+)")
    matter    = find(r"Matter[:\s]+(.+?)(?:\n|$)")
    date_filed = find(r"Date Filed[:\s]+(.+?)(?:\n|$)")
    attorney  = find(r"Assigned Attorney[:\s]+(.+?)(?:\n|$)")
    client    = find(r"Client[:\s]+(.+?)(?:\n|$)")
    opposing  = find(r"Opposing Party[:\s]+(.+?)(?:\n|$)")
    status    = find(r"STATUS[:\s]+(.+?)(?:\n|$)")

    # financial figures
    amounts = re.findall(r"\$[\d,]+(?:\.\d+)?[MKmk]?", text)

    # dates
    dates = re.findall(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
        text,
    )

    return {
        "source_file": filename,
        "case_id":     case_id,
        "matter":      matter,
        "date_filed":  date_filed,
        "attorney":    attorney,
        "client":      client,
        "opposing_party": opposing,
        "status":      status,
        "financial_figures": list(dict.fromkeys(amounts)),   # dedupe, keep order
        "dates_mentioned":   list(dict.fromkeys(dates)),
        "word_count":   len(text.split()),
        "has_unclear_sections": "[TEXT UNCLEAR]" in text,
    }


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """
    Split text into overlapping word-level chunks.
    Each chunk carries its position so retrieval can cite it.
    """
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "chunk_id":  idx,
            "start_word": start,
            "end_word":   end,
            "text":       " ".join(chunk_words),
        })
        idx += 1
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── main extraction functions ─────────────────────────────────────────────────

def extract_from_text_file(path: str) -> dict:
    """Read a plain-text legal document and return a processed doc dict."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return _process_raw(raw, Path(path).name)


def extract_from_pdf(path: str) -> dict:
    """
    Extract text from a PDF (including scanned pages via embedded text layer).
    Falls back to a page-level concatenation with page markers.
    """
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if not page_text.strip():
            # Blank text layer — flag it (OCR would go here in production)
            page_text = f"[PAGE {page_num}: NO TEXT LAYER — OCR REQUIRED]"
        pages.append(f"--- PAGE {page_num} ---\n{page_text}")
    raw = "\n\n".join(pages)
    doc.close()
    return _process_raw(raw, Path(path).name)


def extract_from_bytes(file_bytes: bytes, filename: str) -> dict:
    """Entry point for API uploads: auto-detect format from extension."""
    ext = Path(filename).suffix.lower()
    tmp = Path(f"/tmp/{filename}")
    tmp.write_bytes(file_bytes)
    try:
        if ext == ".pdf":
            return extract_from_pdf(str(tmp))
        else:
            # treat as text (txt, md, etc.)
            raw = file_bytes.decode("utf-8", errors="replace")
            return _process_raw(raw, filename)
    finally:
        tmp.unlink(missing_ok=True)


def _process_raw(raw: str, filename: str) -> dict:
    """Common pipeline: clean → structure → chunk → return."""
    cleaned = _clean_text(_strip_noise_markers(raw))
    fields  = _extract_structured_fields(cleaned, filename)
    chunks  = _chunk_text(cleaned)
    doc_id  = hashlib.md5(cleaned.encode()).hexdigest()[:12]

    return {
        "doc_id":    doc_id,
        "metadata":  fields,
        "full_text": cleaned,
        "chunks":    chunks,
    }


# ── batch helper ──────────────────────────────────────────────────────────────

def process_directory(directory: str) -> list[dict]:
    """Process all supported files in a directory."""
    results = []
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() in {".txt", ".pdf", ".md"}:
            try:
                if p.suffix.lower() == ".pdf":
                    doc = extract_from_pdf(str(p))
                else:
                    doc = extract_from_text_file(str(p))
                results.append(doc)
            except Exception as exc:
                results.append({"error": str(exc), "file": p.name})
    return results


if __name__ == "__main__":
    docs = process_directory("data/sample_docs")
    for d in docs:
        if "error" in d:
            print(f"ERROR: {d}")
            continue
        m = d["metadata"]
        print(f"\n{'='*60}")
        print(f"File      : {m['source_file']}")
        print(f"Case ID   : {m['case_id']}")
        print(f"Matter    : {m['matter']}")
        print(f"Chunks    : {len(d['chunks'])}")
        print(f"Unclear?  : {m['has_unclear_sections']}")
        print(f"Financials: {m['financial_figures']}")
