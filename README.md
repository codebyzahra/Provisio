# Coach Agent

**Owner:** Zahra · **Branch:** `zahra-coach`  
**Pipeline position:** Mapper → Mentor → **Coach**

---

## What It Does

The Coach Agent is the third and final stage of the Onboarding Copilot pipeline.
It takes the structured outputs of the Mapper Agent and the Mentor Agent, runs a
targeted 3-question quiz to test what the developer actually learned, scores their
answers, and recommends exactly one real starter task from the repository that
matches their demonstrated understanding.

The full session result is written to `coach_output.json` and printed to stdout
as a single JSON object, ready to be consumed by an orchestrator without extra
parsing.

### Steps performed

1. **Load inputs** — reads `mapper_output.json` (Mapper Agent) and
   `mentor_output.json` (Mentor Agent).
2. **Generate quiz** — produces 3 questions derived from the topics covered in
   the session log, prioritising topics where the developer reported lower
   confidence.
3. **Score answers** — keyword-overlap scoring classifies each topic as
   `"understood"` or `"needs_review"`.
4. **Recommend a task** — picks one file from `entry_points` (the Mapper's
   import-ranked list) whose classification matches what the developer understood.
5. **Write output** — saves `coach_output.json` and prints to stdout.

---

## Files

| File | Role |
|---|---|
| `src/coach.py` | Main implementation and CLI entry point |
| `src/coach_schema.py` | Pydantic output models (`CoachOutput`, `QuizItem`, `TaskRecommendation`) |
| `src/mentor_schema.py` | Pydantic contract for Mentor Agent output (`MentorOutput`, `SessionEntry`) |
| `src/mapper_schema.py` | Shared upstream contract (Mapper Agent — do not modify) |

---

## Input Format

The Coach Agent reads two JSON files.

### `mapper_output.json` — produced by the Mapper Agent

```json
{
  "repo_root": "/path/to/repo",
  "generated_at": "2024-01-15T10:00:00",
  "entry_points": [
    { "file": "src/main.py",          "import_count": 12, "confidence": "high"   },
    { "file": "src/utils/helpers.py", "import_count":  7, "confidence": "medium" }
  ],
  "classification": {
    "src/main.py":          { "category": "core_logic",  "reason": "root file 'main.py' → core_logic" },
    "src/utils/helpers.py": { "category": "core_logic",  "reason": "folder 'src' → core_logic"        },
    "tests/test_main.py":   { "category": "testing",     "reason": "filename matches 'test_*.py'"     }
  },
  "setup": {
    "language": "python",
    "dependencies_file": "requirements.txt",
    "run_steps": ["pip install -r requirements.txt", "python src/main.py"]
  },
  "unclear_items": []
}
```

### `mentor_output.json` — produced by the Mentor Agent

> **Mentor implementors:** your agent MUST write this exact schema.
> Load/save via `MentorOutput.from_json_file()` / `to_json_file()`.

```json
{
  "session": [
    {
      "question":              "What is the role of the entry-point ranking in the Mapper?",
      "answer":                "The Mapper ranks Python files by how many other files import them, so the most central files surface first for new contributors.",
      "developer_confidence":  "high",
      "timestamp":             "2024-01-15T10:05:00"
    },
    {
      "question":              "How does the Mapper classify a file as core_logic?",
      "answer":                "Files inside a 'src' folder or named 'main.py' at the root are classified as core_logic using folder and filename pattern rules.",
      "developer_confidence":  "medium",
      "timestamp":             "2024-01-15T10:07:00"
    },
    {
      "question":              "What does the setup guide detect?",
      "answer":                "It looks for requirements.txt, package.json, or a Dockerfile at the repo root and generates ordered run steps.",
      "developer_confidence":  "low",
      "timestamp":             "2024-01-15T10:09:00"
    }
  ]
}
```

### `answers.json` — developer's quiz answers (optional)

A plain JSON array of strings, one per quiz question in order:

```json
[
  "The Mapper ranks files by import count so the most central modules appear first.",
  "main.py at the root or any file inside src/ is classified as core_logic.",
  "It checks for requirements.txt and generates pip install steps."
]
```

Omit this file (or pass no `--answers` flag) to score all questions as
`"needs_review"`.

---

## Output Format

Written to `coach_output.json` and printed to stdout.

```json
{
  "generated_at": "2024-01-15T10:12:00",
  "repo_root": "/path/to/repo",
  "quiz": [
    {
      "topic":              "setup guide detect",
      "question":           "In your own words, explain setup guide detect (hint: think about \"It looks for requirements.txt\").",
      "expected_keywords":  ["requirements.txt", "package.json", "dockerfile", "repo", "root", "generates", "ordered", "steps"],
      "developer_answer":   "It checks for requirements.txt and generates pip install steps.",
      "verdict":            "understood"
    },
    {
      "topic":              "mapper classify file core_logic",
      "question":           "In your own words, explain mapper classify file core_logic (hint: think about \"Files inside a 'src' folder\").",
      "expected_keywords":  ["files", "inside", "folder", "named", "root", "classified", "core_logic", "folder", "filename", "pattern", "rules"],
      "developer_answer":   "main.py at the root or any file inside src/ is classified as core_logic.",
      "verdict":            "understood"
    },
    {
      "topic":              "role entry-point ranking mapper",
      "question":           "In your own words, explain role entry-point ranking mapper (hint: think about \"The Mapper ranks Python files\").",
      "expected_keywords":  ["mapper", "ranks", "python", "files", "other", "files", "import", "them", "most", "central", "files", "surface", "first", "contributors"],
      "developer_answer":   "The Mapper ranks files by import count so the most central modules appear first.",
      "verdict":            "understood"
    }
  ],
  "topic_scores": {
    "setup guide detect":             "understood",
    "mapper classify file core_logic": "understood",
    "role entry-point ranking mapper": "understood"
  },
  "recommended_task": {
    "file":   "src/main.py",
    "reason": "Start with `src/main.py` — it is a core_logic file (root file 'main.py' → core_logic) and your quiz results show you already understand \"setup guide detect\", \"mapper classify file core_logic\", which maps directly to this file's responsibilities."
  },
  "error": ""
}
```

### Error response (when session log is empty or inputs cannot be loaded)

```json
{
  "generated_at": "2024-01-15T10:12:00",
  "repo_root":    "",
  "quiz":         [],
  "topic_scores": {},
  "recommended_task": null,
  "error": "Session log is empty — the Mentor Agent produced no question-answer entries.  Run the Mentor Agent first."
}
```

---

## Usage

### CLI

```bash
# All defaults (reads mapper_output.json and mentor_output.json in cwd)
python src/coach.py --mapper-output mapper_output.json \
                    --mentor-output  mentor_output.json \
                    --answers        answers.json

# Custom output path
python src/coach.py --mapper-output mapper_output.json \
                    --mentor-output  mentor_output.json \
                    --answers        answers.json \
                    --output         results/coach_output.json

# No answers provided — all topics scored as needs_review
python src/coach.py --mapper-output mapper_output.json \
                    --mentor-output  mentor_output.json
```

### Python / Orchestrator

```python
from mapper_schema import MapperOutput
from mentor_schema import MentorOutput
from coach import run_coach_session

mapper_output = MapperOutput.from_json_file("mapper_output.json")
mentor_output = MentorOutput.from_json_file("mentor_output.json")
answers = ["answer 1", "answer 2", "answer 3"]

output = run_coach_session(mapper_output, mentor_output, answers)
output.to_json_file("coach_output.json")
print(output.model_dump_json(indent=2))
```

---

## Edge Cases

| Condition | Behaviour |
|---|---|
| Empty session log | Returns `CoachOutput` with `error` set; no crash |
| Fewer than 3 session entries | Quiz has fewer than 3 questions; scoring still works |
| Missing `answers.json` | All questions scored `"needs_review"`; warning printed to stderr |
| `answers.json` shorter than quiz | Unanswered questions scored `"needs_review"` |
| No `entry_points` in Mapper output | `recommended_task` is `null`; no crash |
| Missing Mapper/Mentor file | `error` field set with helpful message; exits with code 1 |

---

## Pipeline Integration

```
mapper.py ──► mapper_output.json ──┐
                                    ├──► coach.py ──► coach_output.json
mentor.py ──► mentor_output.json ──┘
```

The Coach Agent consumes both upstream outputs without modification.  No field
renames or adapter layers are needed as long as both upstream agents write the
schemas defined in `mapper_schema.py` and `mentor_schema.py`.

---

## Dependencies

```
pydantic
```

Same single dependency as the rest of the pipeline (see `requirements.txt`).
