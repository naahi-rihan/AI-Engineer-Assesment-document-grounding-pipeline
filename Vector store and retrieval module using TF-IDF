"""
retrieval_engine.py
-------------------
Lightweight retrieval over processed legal document chunks.

Uses TF-IDF + cosine similarity so there are zero heavy ML dependencies.
In production, swap _score() for an embedding model call (e.g. text-embedding-3-small).

Key design:
  - Each chunk is indexed with its source doc_id and chunk_id
  - retrieve() returns ranked chunks + provenance metadata
  - The caller can see exactly which passage supported which part of the output
"""

import re
import math
import json
from collections import Counter, defaultdict
from typing import Optional


# ── TF-IDF implementation ─────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, remove 1-char tokens and stopwords."""
    _STOP = {
        "the","a","an","and","or","of","to","in","is","was","be","are",
        "for","on","at","by","it","as","this","that","with","from","not",
        "have","has","had","been","will","would","could","should","may",
        "its","we","he","she","they","their","our","his","her","said",
    }
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOP]


class TFIDFIndex:
    """
    In-memory TF-IDF index over document chunks.
    Supports incremental addition of documents.
    """

    def __init__(self):
        self._chunks: list[dict] = []           # {doc_id, chunk_id, text, metadata}
        self._tf: list[Counter] = []            # per-chunk term frequencies
        self._df: Counter = Counter()           # document (chunk) frequency per term
        self._idf: dict[str, float] = {}        # cached IDF scores
        self._dirty = False                     # recompute IDF on next query

    # ── indexing ──────────────────────────────────────────────────────────────

    def add_document(self, doc: dict):
        """
        Index all chunks from a processed document dict
        (as returned by document_processor).
        """
        doc_id   = doc["doc_id"]
        metadata = doc["metadata"]
        for chunk in doc["chunks"]:
            tokens  = _tokenize(chunk["text"])
            tf      = Counter(tokens)
            self._chunks.append({
                "doc_id":   doc_id,
                "chunk_id": chunk["chunk_id"],
                "text":     chunk["text"],
                "metadata": metadata,
            })
            self._tf.append(tf)
            for term in set(tokens):
                self._df[term] += 1
        self._dirty = True

    def _recompute_idf(self):
        N = len(self._chunks) or 1
        self._idf = {
            term: math.log((N + 1) / (df + 1)) + 1   # smoothed
            for term, df in self._df.items()
        }
        self._dirty = False

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _score_chunk(self, query_tokens: list[str], chunk_idx: int) -> float:
        """Cosine similarity between query TF-IDF and chunk TF-IDF."""
        tf   = self._tf[chunk_idx]
        total = sum(tf.values()) or 1
        score = 0.0
        for term in query_tokens:
            if term in tf and term in self._idf:
                tfidf = (tf[term] / total) * self._idf[term]
                score += tfidf
        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top-k chunks most relevant to the query.
        Each result includes: doc_id, chunk_id, text, score, metadata.
        """
        if self._dirty:
            self._recompute_idf()

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = []
        for idx, chunk in enumerate(self._chunks):
            if doc_id_filter and chunk["doc_id"] != doc_id_filter:
                continue
            s = self._score_chunk(q_tokens, idx)
            if s > 0:
                scores.append((s, idx))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            c = self._chunks[idx]
            results.append({
                "doc_id":   c["doc_id"],
                "chunk_id": c["chunk_id"],
                "score":    round(score, 4),
                "text":     c["text"],
                "source":   c["metadata"].get("source_file", "unknown"),
                "case_id":  c["metadata"].get("case_id"),
                "matter":   c["metadata"].get("matter"),
            })
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "chunks": self._chunks,
            "tf":     [dict(t) for t in self._tf],
            "df":     dict(self._df),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TFIDFIndex":
        idx = cls()
        idx._chunks = data["chunks"]
        idx._tf     = [Counter(t) for t in data["tf"]]
        idx._df     = Counter(data["df"])
        idx._dirty  = True
        return idx

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "TFIDFIndex":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __len__(self):
        return len(self._chunks)


# ── singleton store ───────────────────────────────────────────────────────────

_INDEX: Optional[TFIDFIndex] = None
_INDEX_PATH = "data/outputs/index.json"


def get_index() -> TFIDFIndex:
    global _INDEX
    if _INDEX is None:
        import os
        if os.path.exists(_INDEX_PATH):
            _INDEX = TFIDFIndex.load(_INDEX_PATH)
        else:
            _INDEX = TFIDFIndex()
    return _INDEX


def add_and_save(doc: dict):
    idx = get_index()
    idx.add_document(doc)
    import os; os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
    idx.save(_INDEX_PATH)


def retrieve(query: str, top_k: int = 5, doc_id: Optional[str] = None) -> list[dict]:
    return get_index().retrieve(query, top_k=top_k, doc_id_filter=doc_id)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from processing.document_processor import process_directory

    docs = process_directory("data/sample_docs")
    idx = TFIDFIndex()
    for d in docs:
        if "error" not in d:
            idx.add_document(d)
            print(f"Indexed: {d['metadata']['source_file']} — {len(d['chunks'])} chunks")

    query = "termination contract breach performance"
    results = idx.retrieve(query, top_k=3)
    print(f"\nQuery: '{query}'")
    for r in results:
        print(f"\n  [{r['score']}] {r['source']} / chunk {r['chunk_id']}")
        print(f"  {r['text'][:200]}...")
