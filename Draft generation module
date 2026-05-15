"""
draft_generator.py
------------------
Generates grounded legal draft memos using Claude.

Every draft is anchored to retrieved passages — the model is explicitly
instructed not to add unsupported facts. Each draft includes:
  - a case fact summary / internal memo
  - provenance: which chunks were used
  - a confidence note where text was unclear
"""

import os
import json
import anthropic
from typing import Optional

_CLIENT = None

def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _CLIENT


# ── prompt assembly ──────────────────────────────────────────────────────────

def _build_prompt(
    doc_metadata: dict,
    retrieved_chunks: list[dict],
    draft_type: str,
    learned_patterns: list[str],
    extra_instructions: str = "",
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    """
    chunks_text = "\n\n".join(
        f"[PASSAGE {i+1} | Source: {c['source']} | Chunk {c['chunk_id']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )

    pattern_block = ""
    if learned_patterns:
        pattern_block = (
            "\n\nIMPROVEMENT PATTERNS (from prior operator edits — apply these):\n"
            + "\n".join(f"- {p}" for p in learned_patterns)
        )

    system = f"""You are a senior paralegal at Pearson Specter Litt producing internal legal memos.

RULES — READ CAREFULLY:
1. Every factual claim in your output MUST be directly supported by the provided passages.
2. If a passage contains unclear text (marked [TEXT UNCLEAR]), note the uncertainty explicitly.
3. Do NOT infer, assume, or add facts not present in the passages.
4. Where evidence is weak or absent, say so plainly ("evidence not available in provided documents").
5. Structure your output cleanly: use sections, be concise, be precise.
6. Always include a SOURCES section at the end listing which passages you relied on.
{pattern_block}
"""

    user = f"""Draft Type: {draft_type}

Case Metadata:
- Case ID: {doc_metadata.get('case_id', 'N/A')}
- Matter: {doc_metadata.get('matter', 'N/A')}
- Date Filed: {doc_metadata.get('date_filed', 'N/A')}
- Assigned Attorney: {doc_metadata.get('attorney', 'N/A')}
- Client: {doc_metadata.get('client', 'N/A')}
- Opposing Party: {doc_metadata.get('opposing_party', 'N/A')}

Retrieved Evidence Passages:
{chunks_text}

{extra_instructions}

Generate a {draft_type} grounded strictly in the evidence above. Be precise and cite passage numbers."""

    return system, user


# ── main generation function ──────────────────────────────────────────────────

def generate_draft(
    doc_metadata: dict,
    retrieved_chunks: list[dict],
    draft_type: str = "Internal Case Fact Summary Memo",
    learned_patterns: list[str] = None,
    extra_instructions: str = "",
) -> dict:
    """
    Generate a grounded draft and return a result dict with:
      - draft_text: the generated memo
      - sources_used: list of chunk references
      - metadata: pass-through
      - model: which Claude model was used
    """
    if learned_patterns is None:
        learned_patterns = []

    system, user = _build_prompt(
        doc_metadata, retrieved_chunks, draft_type, learned_patterns, extra_instructions
    )

    response = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    draft_text = response.content[0].text

    sources_used = [
        {
            "passage": i + 1,
            "source":    c["source"],
            "chunk_id":  c["chunk_id"],
            "score":     c["score"],
        }
        for i, c in enumerate(retrieved_chunks)
    ]

    return {
        "draft_text":   draft_text,
        "draft_type":   draft_type,
        "sources_used": sources_used,
        "metadata":     doc_metadata,
        "model":        response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from backend.processing.document_processor import process_directory
    from backend.retrieval.retrieval_engine import TFIDFIndex

    docs = process_directory("data/sample_docs")
    idx  = TFIDFIndex()
    for d in docs:
        if "error" not in d:
            idx.add_document(d)

    # Use first doc
    doc = docs[0]
    query = "contract breach termination performance standards damages"
    chunks = idx.retrieve(query, top_k=5, doc_id_filter=doc["doc_id"])

    result = generate_draft(
        doc_metadata=doc["metadata"],
        retrieved_chunks=chunks,
        draft_type="Internal Case Fact Summary Memo",
    )

    print(result["draft_text"])
    print("\n--- SOURCES ---")
    for s in result["sources_used"]:
        print(f"  Passage {s['passage']}: {s['source']} chunk {s['chunk_id']} (score {s['score']})")
