# AI-Engineer-Assesment-document-grounding-pipeline
# AI Legal Document Grounding System

## Features
- OCR ingestion
- Retrieval grounded drafting
- Evidence tracing
- Operator edit learning

## Architecture
(image)

## Setup
(commands)

## Example Workflow
(step-by-step)

## Evaluation
(metrics/results)

## Tradeoffs
(design decisions)
# Legal AI — Document Grounding Pipeline
**Pearson Specter Litt · Internal Workflow**

An end-to-end pipeline that ingests messy legal documents, extracts structured content, retrieves grounded evidence, generates draft memos anchored to that evidence, and improves over time from operator edits.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  POST /documents/upload   POST /draft   POST /edits/submit      │
└───────────┬─────────────────────┬──────────────────┬───────────┘
            │                     │                  │
            ▼                     ▼                  ▼
┌──────────────────┐  ┌───────────────────┐  ┌─────────────────────┐
│ Document         │  │ Draft Generator   │  │ Improvement Engine  │
│ Processor        │  │                   │  │                     │
│                  │  │ Builds grounded   │  │ Captures operator   │
│ • Text/PDF OCR   │  │ prompt from       │  │ edits → extracts    │
│ • Noise cleaning │  │ retrieved chunks  │  │ reusable patterns   │
│ • Field extract  │  │ + learned         │  │ via Claude → injects│
│ • Chunking       │  │ patterns          │  │ into future prompts │
└────────┬─────────┘  └────────┬──────────┘  └─────────────────────┘
         │                     │
         ▼                     ▼
┌──────────────────────────────────────────┐
│           TF-IDF Retrieval Engine        │
│                                          │
│  Indexes chunks → scores cosine sim →   │
│  returns ranked passages + provenance   │
└──────────────────────────────────────────┘
```

### Component overview

| Module | Location | Responsibility |
|---|---|---|
| Document Processor | `backend/processing/document_processor.py` | OCR/text extraction, cleaning, structured field extraction, chunking |
| Retrieval Engine | `backend/retrieval/retrieval_engine.py` | TF-IDF index, cosine similarity retrieval, provenance metadata |
| Draft Generator | `backend/generation/draft_generator.py` | Grounded prompt assembly, Claude API call, source citation |
| Improvement Engine | `backend/improvement/improvement_engine.py` | Edit capture, pattern extraction via Claude, pattern injection |
| API | `backend/api/main.py` | FastAPI endpoints wiring all modules together |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/naahi-rihan/AI-Engineer-Assesment-document-grounding-pipeline.git
cd AI-Engineer-Assesment-document-grounding-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the server

```bash
uvicorn backend.api.main:app --reload
```

API will be live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Example Workflow

### Step 1 — Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/sample_docs/sample_01_contract_dispute.txt"
```

Response:
```json
{
  "doc_id": "a1b2c3d4e5f6",
  "metadata": { "case_id": "PSL-2024-0847", "matter": "Wyatt Industries v. Norwood Consulting Group", ... },
  "chunks": 7,
  "message": "Document indexed successfully"
}
```

### Step 2 — Generate a draft

```bash
curl -X POST http://localhost:8000/draft \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "a1b2c3d4e5f6", "draft_type": "Internal Case Fact Summary Memo"}'
```

The draft is grounded in retrieved passages. Each claim is tied to a passage number; the SOURCES section lists which chunks were used and their relevance scores.

### Step 3 — Submit an operator edit

After reviewing and editing the draft, submit both versions:

```bash
curl -X POST http://localhost:8000/edits/submit \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "a1b2c3d4e5f6",
    "draft_type": "Internal Case Fact Summary Memo",
    "original_draft": "<original text>",
    "edited_draft": "<your improved version>"
  }'
```

Response includes patterns learned:
```json
{
  "edit_id": "a3f9c12b44",
  "patterns_extracted": [
    "Always include the exact dollar value of the liability cap when mentioned in documents.",
    "Lead every memo with client name and case ID on the first line."
  ],
  "message": "Edit captured. 2 pattern(s) learned."
}
```

### Step 4 — Next draft auto-improves

The next `/draft` call for the same draft type will automatically inject the learned patterns into the system prompt.

---

## Evaluation

### Approach

Evaluation was performed manually using the two sample documents (contract dispute and employment matter) as inputs. For each document:

1. The draft was generated with no learned patterns (baseline).
2. An operator edit was submitted with concrete improvements (see `sample_outputs/`).
3. Patterns were extracted and a second draft was generated with those patterns injected.
4. The two drafts were compared for structure, grounding, and specificity.

### Results

| Dimension | Baseline draft | After 1 edit cycle |
|---|---|---|
| Financial figures specificity | Vague ("significant amounts") | Exact dollar values with section references |
| Structure | Prose paragraphs | Labelled sections (MATTER OVERVIEW, FINANCIAL EXPOSURE, etc.) |
| Unclearness flagging | Inline `[TEXT UNCLEAR]` only | Dedicated UNCLEARNESS FLAGS section |
| Liability cap analysis | Missing | Included with explicit note to confirm |
| Source traceability | Passage numbers present | Passage numbers + chunk IDs + scores |

Retrieval quality: Top-5 TF-IDF chunks consistently surfaced the most relevant passages for queries like `"contract breach termination performance standards damages"`.

---

## Assumptions and Tradeoffs

**TF-IDF over embeddings** — The retrieval layer uses TF-IDF rather than vector embeddings (e.g. `text-embedding-3-small`). This keeps the system zero-dependency and fully local, at the cost of semantic recall. For production, swapping `_score_chunk()` in `retrieval_engine.py` for an embedding similarity call is a clean one-function change — the rest of the pipeline is unaffected.

**Pattern extraction via LLM** — Rather than diff algorithms, Claude analyses the original and edited draft and extracts generalisable instructions. This avoids case-specific leakage (e.g. a specific dollar figure becoming a "rule") and produces patterns that transfer across documents. The tradeoff is an extra API call per edit and occasional over-generalisation, which is mitigated by the deduplication logic in `_save_patterns()`.

**In-memory document store** — `_DOCS` in `main.py` is a simple dict that reloads sample docs on startup and accepts uploads into memory. This is fine for a prototype. In production, replace with a database (e.g. SQLite or Postgres) and persist the TF-IDF index on every write (already supported via `idx.save()`).

**Grounding enforcement via prompting** — The model is instructed via system prompt not to add unsupported facts and to explicitly flag where evidence is absent. This is not a hard technical constraint — a sufficiently capable model can still hallucinate — but it is the practical approach at this scale. A production system would add a post-generation grounding check.

**Chunk size** — Chunks are 400 words with 80-word overlap. This works well for the sample documents. For very long contracts, smaller chunks (200 words) with a reranker would improve precision.

---

## Sample Inputs and Outputs

Sample documents are in `data/sample_docs/`. Sample outputs (generated draft + learned patterns) are in `sample_outputs/`.

- `sample_outputs/draft_output_PSL-2024-0847.json` — full draft generation result for the contract dispute case
- `sample_outputs/learned_patterns.json` — patterns extracted after one operator edit cycle

---

## Project Structure

```
.
├── backend/
│   ├── api/
│   │   └── main.py                  # FastAPI app, all routes
│   ├── processing/
│   │   └── document_processor.py    # OCR, extraction, chunking
│   ├── retrieval/
│   │   └── retrieval_engine.py      # TF-IDF index + retrieval
│   ├── generation/
│   │   └── draft_generator.py       # Grounded draft via Claude
│   └── improvement/
│       └── improvement_engine.py    # Edit capture + pattern learning
├── data/
│   ├── sample_docs/                 # Input documents
│   └── outputs/                     # Index + edit/pattern store (auto-created)
├── sample_outputs/                  # Example draft and learned patterns
├── requirements.txt
├── .env.example
└── README.md
```
