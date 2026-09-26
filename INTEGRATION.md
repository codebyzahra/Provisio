# INTEGRATION.md — Agent Integration Guide

> **Audience:** developers extending or building on top of PROVISIO.
> For a project overview, see the top-level [`README.md`](README.md).
> For the Mentor Agent's internal architecture, see [`src/README.md`](src/README.md).

---

## 1. Pipeline overview

```
mapper_output.json
       │  (MapperOutput — Pydantic, auto-validated on load)
       ▼
 mentor.py ◄── source files
       │
       ▼
mentor_output.json
       │  (MentorOutput — plain dataclass, NO auto-validation)
       ▼
 coach.py ◄── mapper_output.json (re-loaded for entry_points)
       │
       ▼
coach_output.json
       │  (CoachOutput — Pydantic, auto-validated on load)
```

All intermediate results are written to JSON files in the working directory.
Each stage reads the previous stage's JSON file directly; there is no shared
in-memory state and no message bus.

---

## 2. Data contracts

### 2.1 MapperOutput (`src/mapper_schema.py`) — Pydantic

Pydantic `BaseModel`. Use `MapperOutput.from_json_file(path)` to load;
validation fires automatically and raises `ValidationError` on malformed data.

```python
class MapperOutput(BaseModel):
    repo_root:     str
    generated_at:  datetime                           # UTC, set at generation time
    entry_points:  list[EntryPoint]                   # ranked by import_count desc
    classification: dict[str, ClassificationEntry]   # path → category + reason
    setup:         SetupInfo
    unclear_items: list[UnclearItem]
```

**Supporting models:**

| Model | Key fields | Notes |
|---|---|---|
| `EntryPoint` | `file: str`, `import_count: int`, `confidence: Confidence` | `confidence` here means import-rank certainty (HIGH/MEDIUM/LOW), **not** developer understanding |
| `ClassificationEntry` | `category: Category`, `reason: str` | `Category` ∈ `core_logic`, `testing`, `documentation`, `configuration`, `unclear` |
| `SetupInfo` | `language: str`, `dependencies_file: str\|None`, `run_steps: list[str]` | |
| `UnclearItem` | `path: str`, `reason: str` | Files that matched no classification rule |

All `file`/`path` fields are normalised to forward slashes by a
`field_validator` — never write backslash-keyed lookups against these dicts.

**JSON wire shape:**

```json
{
  "repo_root": ".",
  "generated_at": "2026-09-26T09:30:00",
  "entry_points": [
    { "file": "src/mapper_schema.py", "import_count": 4, "confidence": "high" }
  ],
  "classification": {
    "src/": { "category": "core_logic", "reason": "matched pattern 'src'" }
  },
  "setup": {
    "language": "python",
    "dependencies_file": "requirements.txt",
    "run_steps": ["pip install -r requirements.txt"]
  },
  "unclear_items": []
}
```

---

### 2.2 MentorOutput (`src/mentor_schema.py`) — plain dataclass

**Not Pydantic.** `MentorOutput` and `SessionEntry` are standard
`@dataclass` objects serialised with `json.dump` / `json.load`.
Use `MentorOutput.from_json_file(path)` to load — this calls `json.load`
then manually constructs the dataclass; **no validation fires on malformed data**.

```python
@dataclass
class MentorOutput:
    entries:          list[SessionEntry]   # ordered Q&A log from the session
    session_metadata: dict[str, Any]       # {"total_questions": int, "saved_at": str}

@dataclass
class SessionEntry:
    question:             str
    answer:               str
    developer_confidence: Confidence       # "high" | "medium" | "low"
    timestamp:            str              # ISO-8601 UTC string
```

**Critical field-name rules (these caused real integration bugs — see §3):**

| Field | Correct name | Common wrong name |
|---|---|---|
| Top-level entry list | `entries` | `session` |
| Per-entry confidence | `developer_confidence` | `confidence` |

`Confidence` here is **developer self-reported understanding**, not the
import-rank certainty field of the same name in `mapper_schema.py`.
The two enums share the same string values (`"high"`, `"medium"`, `"low"`)
but are defined independently in separate modules.

`timestamp` is a plain `str` — no `datetime` parsing happens on load.

**JSON wire shape:**

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

### 2.3 CoachOutput (`src/coach_schema.py`) — Pydantic

Pydantic `BaseModel`. Use `CoachOutput.from_json_file(path)` to load.

```python
class CoachOutput(BaseModel):
    generated_at:     datetime
    repo_root:        str
    quiz:             list[QuizItem]
    topic_scores:     dict[str, str]                  # topic → "understood"|"needs_review"
    recommended_task: Optional[TaskRecommendation]    # None if no entry point available
    error:            str                             # non-empty only on fatal error
```

**Supporting models:**

| Model | Key fields |
|---|---|
| `QuizItem` | `topic: str`, `question: str`, `expected_keywords: list[str]`, `developer_answer: str`, `verdict: TopicVerdict` |
| `TaskRecommendation` | `file: str` (from `MapperOutput.entry_points`), `reason: str` |

`TopicVerdict` ∈ `"understood"`, `"needs_review"`.

When `error` is non-empty, all other meaningful fields (`quiz`, `topic_scores`,
`recommended_task`) may be empty or `None`.

**JSON wire shape:**

```json
{
  "generated_at": "2026-09-26T09:33:48",
  "repo_root": ".",
  "quiz": [
    {
      "topic": "the entry points in mapper.py",
      "question": "In your own words, explain the entry points in mapper.py — hint: ...",
      "expected_keywords": ["entry", "chunks", "files"],
      "developer_answer": "Files are ranked by how many other modules import them.",
      "verdict": "understood"
    }
  ],
  "topic_scores": { "the entry points in mapper.py": "understood" },
  "recommended_task": {
    "file": "src/mapper_schema.py",
    "reason": "Start with `src/mapper_schema.py` — it is the most-imported core_logic file..."
  },
  "error": ""
}
```

---

## 3. Lessons learned — integration bugs hit and fixed

### Bug 1 — Mapper/Mentor field-name mismatch (`session` vs `entries`)

**What happened:** Early Coach code was written against a draft schema that
named the top-level list `session`. The Mentor Agent had already shipped
with the field named `entries`. When Coach loaded `mentor_output.json` it
silently got an empty list and produced a "session log is empty" error even
though the file was valid.

**How it was fixed:** The `MentorOutput.from_dict` loader was audited to
confirm the canonical field name is `entries`. All downstream Coach code that
iterated the session was updated to use `mentor_output.entries`. The field
name `session` does not appear anywhere in the live schema — treat any
reference to it as a stale artefact.

---

### Bug 2 — Inconsistent Pydantic vs dataclass across agents

**What happened:** Mapper and Coach schemas use Pydantic `BaseModel`, which
validates types and raises `ValidationError` on load. Mentor's schema uses
plain `@dataclass` with manual `from_dict` / `to_dict`. Early integration
code assumed `MentorOutput.from_json_file` would raise on a malformed file the
same way `MapperOutput.from_json_file` does. It does not — it silently accepts
wrong types or missing optional keys.

**How it was fixed:** The inconsistency was accepted as a deliberate design
choice (Mentor's schema predated the Pydantic decision for the other two
stages), not papered over with a Pydantic wrapper. The fix was documentation:
any code that loads `mentor_output.json` must not assume validation on load,
and must handle the possibility of missing or wrongly-typed fields explicitly
if robustness is required. See §2.2 above for the full list of safe access
patterns.

---

### Bug 3 — Quiz keyword noise from raw RAG chunk text

**What happened:** The first version of `_extract_keywords` in `coach.py`
extracted keywords directly from the raw `answer` string in `SessionEntry`.
When the Mentor runs without a real LLM, its answers are structured RAG blocks:

```
Answering: "What does mapper.py do?"
──────────────────────────────────────
[1] (source: src/mapper.py, relevance: 0.43)
def analyse_repo(path): ...
```

The keyword extractor treated `source`, `relevance`, `chunks`, and
box-drawing characters as high-frequency terms. Every quiz item ended up with
the same generic keywords — `source`, `relevance`, `chunk` — regardless of the
actual topic, making the scoring nearly useless.

**How it was fixed:** Two targeted helpers were added to `coach.py`:

- **`_extract_answer_body(answer)`** — strips RAG wrapper lines (the
  `Answering:` header, separator lines of 10+ dashes/box-drawing chars, and
  `[n] (source: …, relevance: …)` metadata lines) before any keyword or hint
  extraction runs. Plain prose answers pass through unchanged.
- **Topic-affinity boost in `_extract_keywords`** — after stripping the RAG
  wrapper, words that also appear in the topic phrase receive a 3× frequency
  multiplier, ensuring domain-specific terms outrank generic retrieval
  meta-words that survive the initial strip.

The net result: quiz keywords now reflect the actual subject matter of each
Q&A exchange rather than the structure of the RAG response format.

---

## 4. How to integrate a new agent

### 4.1 Schema conventions

| Decision | Convention |
|---|---|
| **Validation framework** | Use Pydantic `BaseModel` for new agents. Mentor's plain dataclass is a legacy exception, not a template. |
| **File location** | `src/<agent_name>_schema.py` |
| **Serialisation helpers** | Implement `to_json_file(path)` and `from_json_file(path)` on the root output model — the orchestrator calls these by convention. |
| **Path normalisation** | Add a `field_validator` on any `file`/`path` field that normalises backslashes to forward slashes. |
| **Timestamps** | Store as `datetime` in Python; let Pydantic serialise to ISO-8601. Do not store raw `datetime` objects in plain dataclasses. |
| **Error field** | Include `error: str = ""` on the root output model. Set it and leave other fields empty/None when the agent cannot complete its work. |

### 4.2 Output file convention

Each agent writes exactly one JSON file to the working directory:

| Agent | Output file |
|---|---|
| Mapper | `mapper_output.json` |
| Mentor | `mentor_output.json` |
| Coach | `coach_output.json` |
| New agent | `<agent_name>_output.json` |

The next stage in the pipeline loads this file by that fixed name.
Do not make the output path configurable without also exposing the corresponding
`--<agent>-output` CLI flag for the consuming stage, as Coach does for both
`--mapper-output` and `--mentor-output`.

### 4.3 How the orchestrator calls each stage

The pipeline has no formal orchestrator script today — stages are run in order
from the command line (see `README.md`). Any orchestrator written against the
current API should follow this pattern:

```python
from src.mapper_schema  import MapperOutput
from src.mentor_schema  import MentorOutput
from src.coach_schema   import CoachOutput
from src.coach          import run_coach_session

# Stage 1 — Mapper writes mapper_output.json (run externally or import mapper.main())
mapper_out  = MapperOutput.from_json_file("mapper_output.json")   # Pydantic: validates

# Stage 2 — Mentor writes mentor_output.json (run externally or import mentor interactively)
mentor_out  = MentorOutput.from_json_file("mentor_output.json")   # dataclass: no validation

# Stage 3 — Coach: call run_coach_session() directly rather than shelling out
coach_out   = run_coach_session(mapper_out, mentor_out, answers=[])
coach_out.to_json_file("coach_output.json")

if coach_out.error:
    raise RuntimeError(coach_out.error)
```

`run_coach_session` is the canonical programmatic entry point for the Coach
stage. It accepts live Python objects (not file paths) and returns a
`CoachOutput` without any subprocess or file I/O side effects.

### 4.4 Adding the new agent's schema file

1. Create `src/<agent_name>_schema.py`.
2. Define a root `<AgentName>Output(BaseModel)` with `to_json_file` /
   `from_json_file` methods following the pattern in `mapper_schema.py` or
   `coach_schema.py`.
3. If your agent reads the output of a previous stage, import that stage's
   schema directly — do not re-parse the JSON manually.
4. Add the new schema file to the `Schema Files` table in the relevant agent's
   `README` and note which other agents depend on it.
5. Do not modify an existing `*_schema.py` without coordinating with the owner
   of every agent that reads that schema — field renames are silent breaking
   changes for plain-dataclass consumers (Mentor) that have no Pydantic
   migration safety net.

---

## 5. File ownership quick reference

| File | Owner | Framework | Consumers |
|---|---|---|---|
| `src/mapper.py` | Maira | — | Mentor, Coach |
| `src/mapper_schema.py` | Maira | Pydantic | Mentor, Coach |
| `src/mentor.py` | Zahra M | — | Coach |
| `src/mentor_schema.py` | Zahra M | dataclass | Coach |
| `src/coach.py` | Zahra | — | — |
| `src/coach_schema.py` | Zahra | Pydantic | — |
