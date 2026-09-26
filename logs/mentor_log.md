# Mentor Agent — Bob Prompt Log

## Prompt 1 — Build Mentor (Agent mode)
**Purpose:** Build the full Mentor RAG module (chunking, in-memory TF-IDF retrieval,
grounded Q&A, session logging, conversation loop, README) in one call.

**Outcome:** Generated `mentor.py`, `README.md`. Initial run surfaced several
schema-mismatch bugs (Mentor code guessed field names on `EntryPoint`, `SetupInfo`,
and `MentorOutput`/`SessionEntry` instead of matching the real `mapper_schema.py`/
`mentor_schema.py`). Fixed manually (outside Bob, no extra coins spent) by cross-
checking every field against the actual schema files. Also found `mentor_schema.py`
had since become a plain dataclass (not Pydantic) with fields `entries` /
`session_metadata` on `MentorOutput` — flagged for Coach's build too.

**Result after fixes:** `python src/mentor.py` runs end-to-end — builds a 294-chunk
knowledge base, answers questions grounded in retrieved chunks, falls back to
"I don't have enough information to answer that." when ungrounded, and saves a
valid `mentor_output.json`.

---

## Prompt 2 — Generate test suite (Agent mode)
**Purpose:** Generate `test_mentor.py` + sample data to verify Mentor's behavior
without repeated LLM/embedding API calls, so tests can be re-run for free.

**Outcome:** Generated `test_mentor.py` (50 tests) covering chunking, knowledge
base construction, retrieval, grounded/ungrounded Q&A, and `mentor_schema.py`
round-tripping (including the exact `entries`/`session_metadata` field names).

**Result:** `python -m pytest src/test_mentor.py -v` → 50/50 passed.

---

## Coins used: 2 Agent-mode prompts (build + tests). Debugging after each was
done manually to conserve Bobcoins, per team plan.