# Mentor Agent — Onboarding Copilot

Mentor is the second stage of the **Onboarding Copilot** pipeline.  It helps
developers navigate an unfamiliar codebase by answering natural-language
questions, grounded exclusively in the codebase itself and the Mapper Agent's
structured analysis.

```
Mapper → mapper_output.json
                │
                ▼
           mentor.py  ←── source files
                │
                ▼
        mentor_output.json
                │
                ▼
             Coach
```

---

## What Mentor Does

1. **Ingests** `mapper_output.json` (validated via `MapperOutput.from_json_file`)
   and every source file listed in its `classification` dict.
2. **Chunks** both sources — Python files are split by function/class using the
   AST; all other files use fixed-size overlapping windows.  Mapper summaries
   are split into one chunk per entry-point, one classification summary chunk,
   and one setup-info chunk.
3. **Vectorises** every chunk using TF-IDF (scikit-learn), fitted in-memory
   once at startup — no vector database, no network calls, no GPU.
4. **Retrieves** the top-K most relevant chunks for each question via cosine
   similarity.
5. **Answers** grounded questions from the retrieved context, or returns
   `"I don't have enough information to answer that."` when no chunk exceeds
   the relevance threshold.
6. **Logs** each exchange (question, answer, developer's self-reported
   confidence, UTC timestamp) into `mentor_output.json` for the Coach Agent.

---

## Architecture — Lightweight RAG without a Vector Database

### Why no vector database?

| Concern | Reasoning |
|---|---|
| **Scale** | A typical codebase produces hundreds, not millions, of chunks. An in-memory NumPy matrix is fast and requires no infrastructure. |
| **Dependencies** | No ChromaDB, FAISS, Pinecone, or Weaviate to install, configure, or keep running. |
| **Portability** | The entire knowledge base lives in a Python list; it can be serialised, passed between functions, and garbage-collected naturally. |
| **Latency** | TF-IDF vectorisation + cosine similarity over <10 000 chunks completes in milliseconds on any laptop. |

### Pipeline

```
Source files + mapper_output.json
          │
          ▼ chunk()
  list[{text, source_path, source_type}]
          │
          ▼ TfidfVectorizer.fit_transform()
  list[{..., vector: ndarray}]          ← build_knowledge_base()
          │
   question ──► TfidfVectorizer.transform()
          │
          ▼ cosine_similarity()
  top-K ranked chunks                   ← retrieve()
          │
          ▼ _generate_answer()
  grounded answer string                ← ask()
          │
          ▼ SessionEntry + MentorOutput.to_json_file()
  mentor_output.json                    ← save_session()
```

### Chunking strategy

| File type | Strategy |
|---|---|
| `.py` | AST walk — one chunk per `def`/`class`; module preamble as one chunk |
| All others | Fixed-size windows (1 500 chars) with 200-char overlap |
| Mapper JSON | One chunk per entry-point, one classification summary, one setup-info chunk |

### Relevance gating

If the highest cosine-similarity score among the top-K retrieved chunks is
below `RELEVANCE_THRESHOLD` (default **0.10**), `ask()` returns
`"I don't have enough information to answer that."` rather than hallucinating
an answer from low-signal context.

---

## Installation

```bash
pip install scikit-learn numpy
```

> **Python ≥ 3.9** required.  No other runtime dependencies.

---

## Running Standalone

```bash
# Interactive mode (prompts for paths)
python src/mentor.py

# Direct invocation
python src/mentor.py path/to/mapper_output.json path/to/repo/root
```

The session is saved to `mentor_output.json` in the current working directory
when you type `exit` or `quit`.

---

## Programmatic / Pipeline Use

```python
from src.mentor import build_knowledge_base, ask, retrieve, save_session
from src.mentor_schema import SessionEntry, Confidence
from datetime import datetime, timezone

# Build once
chunks = build_knowledge_base("mapper_output.json", "/path/to/repo")

# Ask a question
answer = ask("What does the AuthService do?", chunks)
print(answer)

# Retrieve raw chunks (e.g. for your own LLM integration)
results = retrieve("How is the database configured?", chunks, top_k=3)
for r in results:
    print(r["source_path"], r["score"], r["text"][:120])

# Log & save manually
entry = SessionEntry(
    question="What does the AuthService do?",
    answer=answer,
    developer_confidence=Confidence.HIGH,
    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
)
save_session([entry], output_path="mentor_output.json")
```

---

## Output — `mentor_output.json`

Consumed by the **Coach Agent** to identify knowledge gaps and tailor follow-up
learning material.

```json
{
  "entries": [
    {
      "question": "What does the AuthService do?",
      "answer": "Answering: ...",
      "developer_confidence": "high",
      "timestamp": "2024-01-15T10:30:00"
    }
  ],
  "session_metadata": {
    "total_questions": 1,
    "saved_at": "2024-01-15T10:31:00+00:00"
  }
}
```

---

## Schema Files

| File | Role | Modify? |
|---|---|---|
| `mapper_schema.py` | Input — Mapper Agent output model | ❌ No |
| `mentor_schema.py` | Output — Session entry + confidence enum | ❌ No |

---

## Extending Mentor

**Swap in a real LLM**: Replace the body of `_generate_answer()` in
`mentor.py` with an API call (e.g. watsonx.ai, OpenAI) that passes
`_SYSTEM_PROMPT` + `context_block` as the prompt.  All retrieval, chunking,
and logging logic stays the same.

**Tune retrieval**: Adjust `RELEVANCE_THRESHOLD`, `FIXED_CHUNK_SIZE`,
`FIXED_CHUNK_OVERLAP`, or the `TfidfVectorizer` hyperparameters at the top of
`mentor.py`.
