# PROVISIO — Onboarding Copilot

> **Built for the IBM Bob 2.0 Hackathon** using Bob IDE's Agent mode.

---

## What it does

New developers waste days reading unfamiliar codebases before they can contribute anything useful.
Most onboarding tools respond to this by *explaining* the code.
PROVISIO goes one step further: it **explains the code, then verifies that the developer actually understood it** before recommending real work.
The result is a structured three-stage pipeline that turns a raw repository path into a quiz score and a matched starter task — no hallucinated context, no generic tutorials, no sending someone to read a file they aren't ready for.

---

## Architecture

```
          repo path
              │
              ▼
       ┌─────────────┐
       │   Mapper    │  ← analyzes repo structure, ranks files by import
       │  (src/mapper.py)   count, classifies each file, detects setup steps
       └──────┬──────┘
              │  mapper_output.json
              ▼
       ┌─────────────┐
       │   Mentor    │  ← RAG-based Q&A grounded exclusively in the codebase;
       │ (src/mentor.py)    chunks source files + mapper analysis, retrieves
       │             │    relevant context via TF-IDF + cosine similarity,
       │             │    records each exchange + developer confidence level
       └──────┬──────┘
              │  mentor_output.json
              ▼
       ┌─────────────┐
       │    Coach    │  ← quizzes the developer on what was covered, scores
       │ (src/coach.py)     each answer, and recommends exactly one real
       │             │    starter task matched to their demonstrated knowledge
       └─────────────┘
              │
              ▼
       coach_output.json
```

| Stage | Responsibility | Owner |
|---|---|---|
| **Mapper** | Parses the repo, ranks files by import count, classifies every file (`core_logic`, `testing`, …), and generates a setup guide | Maira |
| **Mentor** | Answers natural-language questions about the codebase using lightweight RAG (TF-IDF + cosine similarity, no vector DB required) and logs the full Q&A session with confidence ratings | Zahra M |
| **Coach** | Generates a targeted quiz from the session log, scores answers by keyword overlap, and recommends a starter task from the Mapper's ranked entry-point list | Zahra |

---

## Why this is different

Most onboarding tools stop at *explaining* code.

PROVISIO **proves the developer understood it** before pointing them at real work.
The quiz is generated from the developer's own session — not a generic test — and the recommended task is selected from the repository's actual entry points, matched to the topics the developer scored well on.
A developer who cannot pass the quiz gets told which topics to revisit, not a random ticket.

---

## Setup

**Requirements:** Python ≥ 3.9

```bash
pip install -r requirements.txt
```

`requirements.txt` currently contains:

```
pydantic
scikit-learn
numpy
```

> `scikit-learn` and `numpy` are required by the Mentor Agent's TF-IDF vectoriser.
> `pydantic` is used by the Mapper and Coach schemas.

---

## Running the pipeline

Run each agent in order, passing the output of one stage as input to the next.

```bash
# Stage 1 — Mapper: analyze the target repository
python src/mapper.py /path/to/target/repo

# Stage 2 — Mentor: interactive Q&A session
python src/mentor.py mapper_output.json /path/to/target/repo
# Type your questions; type `exit` or `quit` to save mentor_output.json

# Stage 3 — Coach: quiz, score, and recommend a task
python src/coach.py \
    --mapper-output mapper_output.json \
    --mentor-output  mentor_output.json \
    --answers        answers.json        # optional: pre-supplied developer answers
```

All three stages can also be imported as Python modules and wired together in an orchestrator script using the schemas in `mapper_schema.py`, `mentor_schema.py`, and `coach_schema.py`.

---

## Example

**Input:** the PROVISIO repository itself (`repo_path = "."`)

After the Mapper and a short Mentor session, the Coach produces `coach_output.json`:

```json
{
  "generated_at": "2026-09-26T09:33:48",
  "repo_root": ".",
  "quiz": [
    {
      "topic": "the entry points in mapper.py",
      "question": "In your own words, explain the entry points in mapper.py — hint: Entry-point: src/mapper.",
      "expected_keywords": ["entry", "chunks", "files", "chunk", "mentor", "output", "path"],
      "developer_answer": "Files are ranked by how many other modules import them.",
      "verdict": "understood"
    }
  ],
  "topic_scores": {
    "the entry points in mapper.py": "understood"
  },
  "recommended_task": {
    "file": "src/mapper_schema.py",
    "reason": "Start with `src/mapper_schema.py` — it is the most-imported core_logic file in the repository, making it the best entry point for a new contributor."
  },
  "error": ""
}
```

The developer gets a concrete next step — a specific file, with a reason — instead of "read the docs".

---

## Per-agent documentation

Each agent has its own detailed README covering its internal architecture, input/output schemas, edge cases, and extension points:

| Agent | Owner | Detailed README |
|---|---|---|
| Mapper | **Maira** | *(see `src/mapper.py` inline docstrings)* |
| Mentor | **Zahra M** | [`src/README.md`](src/README.md) |
| Coach | **Zahra** | [`README.md` (root, pre-replacement)](README.md) — see `git log` for the original |

---

## Project info

| | |
|---|---|
| **Event** | IBM Bob 2.0 Hackathon |
| **Tooling** | Bob IDE — Agent mode |
| **Language** | Python 3.9+ |
| **External services** | None — fully offline, no LLM API calls required |
