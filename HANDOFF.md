# Mentor Agent — Handoff Summary

**Owner:** Zahra M (branch `zahram-mentor`)
**Status:** Built, debugged, tested (50/50 passing), real output generated and sent to Coach.

This doc exists so anyone building against Mentor's output — or prompting Bob to do so — doesn't repeat the schema-mismatch bugs we already hit and fixed. Paste this to Bob as context before generating any code that reads `mentor_output.json`.

---

## Files I own (do not redesign these)

| File | Purpose |
|---|---|
| `src/mentor.py` | Mentor Agent logic — RAG pipeline (chunk → TF-IDF vectorize → retrieve → answer) |
| `src/mentor_schema.py` | Data contract for Mentor's output — **do not modify**, shared integration boundary with Coach |
| `src/test_mentor.py` | 50-test suite, all passing, safe to re-run for free (`python -m pytest src/test_mentor.py -v`) |
| `src/README.md` | Full architecture explanation |
| `mentor_output.json` | Real sample output, already sent to Coach |

---

## Critical: `mentor_schema.py` is a plain `dataclass`, NOT Pydantic

This matters because Mapper's schema (`mapper_schema.py`) **is** Pydantic — the pipeline does not use one consistent framework end to end. Mentor's output is serialized with plain `json.dump` / `json.load`; there is **no automatic validation on load**.

### Real shape of `MentorOutput`

```json
{
  "entries": [
    {
      "question": "string",
      "answer": "string",
      "developer_confidence": "high" | "medium" | "low",
      "timestamp": "2024-01-15T10:30:00"
    }
  ],
  "session_metadata": {
    "total_questions": 1,
    "saved_at": "2024-01-15T10:31:00+00:00"
  }
}
```

### Field-name gotchas (these caused real bugs — save yourself the pain)

- Top-level field is **`entries`**, not `session`.
- Confidence field is **`developer_confidence`**, not `confidence` — intentionally distinct from Mapper's `EntryPoint.confidence`, which means something different (import-count importance, not developer knowledge).
- `timestamp` is a plain **ISO-8601 string**, not a `datetime` object.
- `Confidence` enum values are lowercase strings: `"high"`, `"medium"`, `"low"`.

### How to read it correctly

```python
from mentor_schema import MentorOutput

output = MentorOutput.from_json_file("mentor_output.json")
# output.entries          -> list[SessionEntry]
# output.session_metadata -> dict
```

Plain dict/dataclass access only — no Pydantic validation happens on load, so malformed data will not raise unless you check it yourself.

---

## What Mentor actually does (context only, not something you need to touch)

- Reads `mapper_output.json` (via `mapper_schema.py`, Pydantic) plus the raw source files it references.
- Chunks everything in-memory — no vector database, deliberately lightweight (TF-IDF + cosine similarity), appropriate for hackathon-scale repos.
- Answers questions grounded only in retrieved chunks.
- Returns exactly the string `"I don't have enough information to answer that."` when nothing relevant is found — verbatim, useful if you want to detect/handle this case downstream.

---

## Open dependency for whoever builds next

`recommended_task.file` (per Coach's own schema) is meant to come from `mapper_output.entry_points` — so anything downstream likely needs **both** `mapper_output.json` and `mentor_output.json` loaded together, not just Mentor's output alone.

---

## If something breaks when you read `mentor_output.json`

Before spending a Bobcoin re-prompting Bob to "fix" it, check this doc first — it is very likely a field-name mismatch against what's described above, not a real bug in the data.