"""
main.py — FastAPI application for Legal AI Document System
Pearson Specter Litt Internal Workflow
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Path setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.processing.document_processor import (
    extract_from_bytes,
    extract_from_text_file,
    process_directory,
)
from backend.retrieval.retrieval_engine import TFIDFIndex, add_and_save, retrieve, get_index
from backend.generation.draft_generator import generate_draft
from backend.improvement.improvement_engine import (
    capture_edit,
    extract_patterns_from_edit,
    load_patterns,
    get_all_edits,
    get_all_patterns,
)

# ── app setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Legal AI — Pearson Specter Litt",
    description="Document ingestion, retrieval, and grounded draft generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── in-memory state (persisted via JSON files) ────────────────────────────────

_DOCS: dict[str, dict] = {}   # doc_id -> processed doc


def _load_sample_docs():
    """Pre-load sample documents from data/sample_docs at startup."""
    sample_dir = ROOT / "data" / "sample_docs"
    if sample_dir.exists():
        docs = process_directory(str(sample_dir))
        idx  = get_index()
        for doc in docs:
            if "error" not in doc:
                _DOCS[doc["doc_id"]] = doc
                idx.add_document(doc)
        # Save index
        out_dir = ROOT / "data" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        idx.save(str(out_dir / "index.json"))


@app.on_event("startup")
async def startup():
    _load_sample_docs()


# ── request/response models ───────────────────────────────────────────────────

class DraftRequest(BaseModel):
    doc_id: str
    draft_type: str = "Internal Case Fact Summary Memo"
    query: Optional[str] = None
    top_k: int = 5


class EditRequest(BaseModel):
    doc_id: str
    draft_type: str
    original_draft: str
    edited_draft: str


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = ROOT / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Legal AI API — see /docs</h1>"


@app.get("/health")
async def health():
    return {"status": "ok", "indexed_chunks": len(get_index())}


# -- Document endpoints --

@app.get("/documents")
async def list_documents():
    return [
        {
            "doc_id":  d["doc_id"],
            "case_id": d["metadata"].get("case_id"),
            "matter":  d["metadata"].get("matter"),
            "source":  d["metadata"].get("source_file"),
            "chunks":  len(d["chunks"]),
            "has_unclear": d["metadata"].get("has_unclear_sections", False),
        }
        for d in _DOCS.values()
    ]


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF or text file and index it."""
    content = await file.read()
    try:
        doc = extract_from_bytes(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}")

    _DOCS[doc["doc_id"]] = doc
    add_and_save(doc)
    return {
        "doc_id":   doc["doc_id"],
        "metadata": doc["metadata"],
        "chunks":   len(doc["chunks"]),
        "message":  "Document indexed successfully",
    }


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    if doc_id not in _DOCS:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = _DOCS[doc_id]
    return {
        "doc_id":    doc["doc_id"],
        "metadata":  doc["metadata"],
        "full_text": doc["full_text"],
        "chunks":    len(doc["chunks"]),
    }


# -- Retrieval endpoint --

@app.get("/retrieve")
async def retrieve_passages(
    query: str,
    doc_id: Optional[str] = None,
    top_k: int = 5,
):
    results = retrieve(query, top_k=top_k, doc_id=doc_id)
    return {"query": query, "results": results}


# -- Draft generation endpoint --

@app.post("/draft")
async def create_draft(req: DraftRequest):
    if req.doc_id not in _DOCS:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = _DOCS[req.doc_id]
    query = req.query or f"{req.draft_type} key facts evidence timeline"

    chunks  = retrieve(query, top_k=req.top_k, doc_id=req.doc_id)
    if not chunks:
        # Fallback: no filter
        chunks = retrieve(query, top_k=req.top_k)

    patterns = load_patterns(draft_type=req.draft_type)

    try:
        result = generate_draft(
            doc_metadata=doc["metadata"],
            retrieved_chunks=chunks,
            draft_type=req.draft_type,
            learned_patterns=patterns,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return result


# -- Improvement endpoints --

@app.post("/edits/submit")
async def submit_edit(req: EditRequest):
    if req.doc_id not in _DOCS:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = _DOCS[req.doc_id]
    edit_id  = capture_edit(
        original_draft=req.original_draft,
        edited_draft=req.edited_draft,
        doc_metadata=doc["metadata"],
        draft_type=req.draft_type,
    )

    # Immediately extract patterns
    try:
        patterns = extract_patterns_from_edit(edit_id)
    except Exception as e:
        patterns = []
        print(f"Pattern extraction failed: {e}")

    return {
        "edit_id":  edit_id,
        "patterns_extracted": patterns,
        "message":  f"Edit captured. {len(patterns)} pattern(s) learned.",
    }


@app.get("/edits")
async def list_edits():
    return get_all_edits()


@app.get("/patterns")
async def list_patterns():
    return get_all_patterns()


@app.get("/patterns/active")
async def active_patterns(draft_type: Optional[str] = None):
    return {"patterns": load_patterns(draft_type=draft_type)}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
