"""
improvement_engine.py
---------------------
Learns from operator edits to improve future drafts.

Pipeline:
  1. Capture: store (original_draft, edited_draft, doc_metadata) 
  2. Extract: use Claude to identify what changed and why (reusable patterns)
  3. Apply: inject learned patterns into future generation prompts

Patterns are stored as short, generalisable instructions —
e.g. "Always state the liability cap value in dollar terms when present."

This is a genuine improvement loop, not a diff viewer.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import anthropic

_CLIENT = None

def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _CLIENT

EDITS_PATH    = "data/outputs/operator_edits.json"
PATTERNS_PATH = "data/outputs/learned_patterns.json"


# ── storage helpers ───────────────────────────────────────────────────────────

def _load_json(path: str, default) -> any:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return default

def _save_json(path: str, data: any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


# ── step 1: capture ───────────────────────────────────────────────────────────

def capture_edit(
    original_draft: str,
    edited_draft: str,
    doc_metadata: dict,
    draft_type: str,
) -> str:
    """
    Store an operator edit. Returns the edit_id.
    """
    edits = _load_json(EDITS_PATH, [])
    edit_id = hashlib.md5(
        (original_draft + edited_draft + datetime.now(timezone.utc).isoformat()).encode()
    ).hexdigest()[:10]

    edits.append({
        "edit_id":        edit_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "draft_type":     draft_type,
        "case_id":        doc_metadata.get("case_id"),
        "matter":         doc_metadata.get("matter"),
        "original_draft": original_draft,
        "edited_draft":   edited_draft,
    })
    _save_json(EDITS_PATH, edits)
    return edit_id


# ── step 2: extract patterns ──────────────────────────────────────────────────

def extract_patterns_from_edit(edit_id: str) -> list[str]:
    """
    Use Claude to analyse one edit and extract reusable instructions.
    Returns a list of pattern strings.
    """
    edits = _load_json(EDITS_PATH, [])
    edit  = next((e for e in edits if e["edit_id"] == edit_id), None)
    if not edit:
        raise ValueError(f"Edit {edit_id} not found")

    system = """You are an AI system that analyses how a human operator edits AI-generated legal memos.
Your job: extract REUSABLE, GENERALIZABLE instructions from the differences.

Rules:
- Focus on structural, stylistic, or factual presentation patterns — not case-specific facts.
- Each pattern must be actionable: future AI can apply it without seeing this specific case.
- Be concise. Each pattern is one sentence, starting with an action verb (e.g. "Always...", "Include...", "Avoid...", "When X is present, ...").
- Output ONLY a JSON array of strings. No preamble. No markdown.
- If there are no meaningful reusable patterns (e.g. operator only fixed a typo), return [].
- Maximum 5 patterns per edit.

Example output: ["Always include the exact dollar value of the liability cap when mentioned in documents.", "Lead with client name and case ID on the first line of every memo."]"""

    user = f"""Draft Type: {edit['draft_type']}

ORIGINAL DRAFT:
{edit['original_draft']}

OPERATOR'S EDITED DRAFT:
{edit['edited_draft']}

Extract reusable patterns from this edit."""

    response = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        patterns = json.loads(raw)
        if not isinstance(patterns, list):
            patterns = []
    except json.JSONDecodeError:
        patterns = []

    # Persist patterns
    _save_patterns(patterns, edit)
    return patterns


# ── step 3: persist and load patterns ────────────────────────────────────────

def _save_patterns(new_patterns: list[str], edit: dict):
    """Merge new patterns into the global pattern store, deduplicating."""
    all_patterns = _load_json(PATTERNS_PATH, [])

    for p in new_patterns:
        # Simple text-level dedup
        if not any(p.lower() == existing["pattern"].lower() for existing in all_patterns):
            all_patterns.append({
                "pattern":    p,
                "edit_id":    edit["edit_id"],
                "draft_type": edit["draft_type"],
                "added_at":   datetime.now(timezone.utc).isoformat(),
                "use_count":  0,
            })

    _save_json(PATTERNS_PATH, all_patterns)


def load_patterns(draft_type: Optional[str] = None, limit: int = 8) -> list[str]:
    """
    Load learned patterns relevant to a draft type.
    Sorted by use_count descending so most-validated patterns come first.
    """
    all_patterns = _load_json(PATTERNS_PATH, [])
    if draft_type:
        # Include patterns with matching type or generic patterns
        filtered = [
            p for p in all_patterns
            if p["draft_type"] == draft_type or p["draft_type"] is None
        ]
    else:
        filtered = all_patterns

    filtered.sort(key=lambda p: p["use_count"], reverse=True)
    result = [p["pattern"] for p in filtered[:limit]]

    # Increment use counts
    pattern_texts = set(result)
    for p in all_patterns:
        if p["pattern"] in pattern_texts:
            p["use_count"] += 1
    _save_json(PATTERNS_PATH, all_patterns)

    return result


def get_all_edits() -> list[dict]:
    return _load_json(EDITS_PATH, [])

def get_all_patterns() -> list[dict]:
    return _load_json(PATTERNS_PATH, [])


# ── optional: missing type hint fix ──────────────────────────────────────────
from typing import Optional


if __name__ == "__main__":
    # Simulate capturing and learning from an operator edit
    original = """INTERNAL MEMO — Wyatt Industries v. Norwood Consulting Group

Summary:
The case involves a contract dispute. Norwood failed performance thresholds.
Termination was issued. Financial figures are significant."""

    edited = """INTERNAL MEMO
Client: Wyatt Industries, Inc. | Case: PSL-2024-0847

MATTER OVERVIEW
Wyatt Industries v. Norwood Consulting Group involves a breach of a $4.2M MSA.
Norwood failed the contractually required 80% satisfaction threshold for two
consecutive quarters (Q2: 71%, Q3: 68%), triggering termination under Section 7.1.

FINANCIAL EXPOSURE
- Fees paid to Norwood: $1,847,000
- Wyatt refund claim: $920,000
- Norwood counterclaim: $2,353,000 (likely capped at $1.8M under Section 9.3)

NOTE: Liability cap analysis under Section 9.3 is a priority action item."""

    meta = {
        "case_id": "PSL-2024-0847",
        "matter":  "Wyatt Industries v. Norwood Consulting Group",
    }

    edit_id  = capture_edit(original, edited, meta, "Internal Case Fact Summary Memo")
    patterns = extract_patterns_from_edit(edit_id)
    print("Extracted patterns:")
    for p in patterns:
        print(f"  - {p}")
