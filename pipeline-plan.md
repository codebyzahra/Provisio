# Pipeline Orchestrator Plan

## Top-Level Overview

Build `src/pipeline.py` — a single-file orchestrator that runs Mapper → Mentor → Coach as
direct in-process Python function calls (no subprocess, no stdin injection).

The pipeline:
1. Calls `mapper.main(["--path", ...] | ["--url", ...])` to produce a `MapperOutput`.
2. Builds a Mentor knowledge base and runs `ask()` over a fixed question list to produce a
   `MentorOutput`.
3. Calls `coach.run_coach_session()` with both prior outputs to produce a `CoachOutput`.
4. Validates each output against its schema after each stage; on failure prints one clear line
   and exits cleanly (no raw traceback).
5. Saves `mapper_output.json`, `mentor_output.json`, `coach_output.json` to disk after each
   stage, and prints a short human-readable summary after each stage.

The pipeline accepts `--path DIR` or `--url URL` as its own CLI arguments (same contract as
`mapper.py`).

---

## Sub-Tasks

---

### Sub-Task 1 — Refactor `mapper.main()` to return `MapperOutput`

**Intent**
`mapper.main()` currently returns `None`. The pipeline needs to receive the `MapperOutput`
object directly. A minimal one-line change makes `main()` return the assembled output while
remaining fully backward-compatible with the existing CLI (since callers that ignore the
return value are unaffected).

**Expected Outcomes**
- `mapper.main(["--path", "."])` returns a `MapperOutput` instance.
- The CLI still works unchanged (`python mapper.py --path .` still prints JSON and exits).
- No other behaviour in `mapper.py` changes.

**Todo List**
1. In `mapper.main()`, locate the last line before the function returns (where
   `MapperOutput(...)` is assembled and written to file).
2. Assign the assembled `MapperOutput` to a local variable (e.g. `output`).
3. Add `return output` as the final statement.

**Relevant Context**
- `src/mapper.py` → `main()` function (lines ~497–548).
- The assembled `MapperOutput` is constructed just before `to_json_file()` is called.

**Status**: [x] done

---

### Sub-Task 2 — Build `src/pipeline.py`

**Intent**
Implement the full orchestrator as a standalone module with its own `argparse`-based CLI
(`--path`/`--url`, same contract as mapper). The pipeline calls each stage in sequence,
validates outputs, saves JSON files, and prints human-readable stage summaries. Failures at
any stage produce one clear error line and a clean `sys.exit(1)`.

**Expected Outcomes**
- Running `python src/pipeline.py --path <repo>` executes all three stages in order.
- After Mapper: `mapper_output.json` exists; summary line printed.
- After Mentor: `mentor_output.json` exists; summary line printed.
- After Coach: `coach_output.json` exists; summary line printed.
- On any validation failure: one error line identifying the stage and reason; process exits 1.
- No raw tracebacks visible to the user; `sys.tracebacklimit` is NOT modified globally.

**Todo List**

#### CLI & Entry Point
1. Define `parse_args(argv)` using `argparse` with a mutually exclusive required group:
   `--path DIR` and `--url URL` (mirrors mapper's contract exactly).
2. Define `main(argv=None)` that calls `parse_args` and drives all three stages.
3. Add `if __name__ == "__main__": main()` guard.

#### Stage 1 — Mapper
4. Build the argv list for mapper from the pipeline's parsed args
   (e.g. `["--path", args.path]` or `["--url", args.url]`).
5. Call `mapper.main(mapper_argv)` inside a `try/except` block.
6. Validate the returned `MapperOutput` using Pydantic's `.model_validate()` (re-validate
   the instance against `MapperOutput`).  On `ValidationError`, call the shared error
   handler.
7. Call `output.to_json_file("mapper_output.json")` (mapper.main already does this, but an
   explicit save here guards against any future refactor; alternatively, rely on mapper's
   internal save and skip the duplicate — see design note below).
8. Print summary: `Step 1: Mapper analysed {N} files, found {M} entry points`.
   - N = `len(output.classification)`
   - M = `len(output.entry_points)`

**Design note on duplicate save**: `mapper.main()` already saves to `mapper_output.json`
internally. The pipeline should rely on that internal save and not call `to_json_file` again
for Mapper (avoids duplicate writes). For Mentor and Coach the pipeline is responsible for
the save, so it calls `to_json_file` explicitly there.

#### Stage 2 — Mentor
9. Call `mentor.build_knowledge_base("mapper_output.json", output.repo_root)` to build the
   chunk index.
10. Define a fixed list of sample questions (5–7 representative developer-onboarding
    questions hardcoded in the pipeline, covering entry points, setup steps, testing, and
    architecture).
11. Loop over questions: call `mentor.ask(q, chunks)` for each; determine
    `developer_confidence` from the answer:
    - If answer equals `mentor.INSUFFICIENT_INFO_RESPONSE` → `Confidence.LOW`
    - Otherwise → `Confidence.MEDIUM`
12. Assemble a `MentorOutput` with `SessionEntry` objects (one per question) and
    `session_metadata = {"total_questions": N, "saved_at": <UTC ISO-8601>}`.
13. Validate `MentorOutput`: assert `isinstance(mentor_output.entries, list)` and that each
    `SessionEntry` has `question`, `answer`, `developer_confidence`, `timestamp` attributes.
    On `AttributeError` or `TypeError`, call the shared error handler.
14. Call `mentor.save_session(mentor_output.entries, "mentor_output.json")`.
15. Print summary: `Step 2: Mentor answered {N} questions ({K} with sufficient context)`.
    - N = total questions
    - K = count where answer ≠ `INSUFFICIENT_INFO_RESPONSE`

#### Stage 3 — Coach
16. Provide a fixed `answers` list (one per quiz question the Coach will generate — use empty
    strings or representative placeholder answers; the pipeline is non-interactive so answers
    are hardcoded or left empty and Coach scores them accordingly).
17. Call `coach.run_coach_session(mapper_output, mentor_output, answers=[])`.
    Use an empty list for answers — Coach's `score_quiz` handles shorter-than-3 lists
    gracefully (any missing answer → NEEDS_REVIEW).
18. Validate the returned `CoachOutput` using Pydantic `.model_validate()`.
    On `ValidationError`, call shared error handler.
    Additionally, if `coach_output.error` is non-empty, treat it as a stage failure.
19. Call `coach_output.to_json_file("coach_output.json")`.
20. Print summary:
    `Step 3: Coach generated {N} quiz items; recommended task: {file} ({verdict})`.
    - N = `len(coach_output.quiz)`
    - file = `coach_output.recommended_task.file` if not None, else "none"
    - verdict = comma-joined topic verdicts from `coach_output.topic_scores`

#### Shared Error Handler
21. Define a module-level `_fail(stage: str, reason: str) -> NoReturn` helper:
    - Prints `f"Pipeline failed at {stage}: {reason}"` to stderr.
    - Calls `sys.exit(1)` to exit cleanly.
    - Wrap each stage's call in `try/except (ValidationError, Exception)` but only catch
      `ValidationError` for validation failures; let unexpected exceptions surface via the
      except block that calls `_fail` with `str(e)` — never prints raw traceback by wrapping
      each stage call in `except Exception as e: _fail(stage, str(e))`.

**Relevant Context**
- `src/mapper.py` → `main()`, `parse_args()`
- `src/mapper_schema.py` → `MapperOutput`
- `src/mentor.py` → `build_knowledge_base()`, `ask()`, `save_session()`,
  `INSUFFICIENT_INFO_RESPONSE`, `run_session()`
- `src/mentor_schema.py` → `MentorOutput`, `SessionEntry`, `Confidence`
- `src/coach.py` → `run_coach_session()`
- `src/coach_schema.py` → `CoachOutput`

**Status**: [ ] pending

---

## Notes

- `MentorOutput` and `SessionEntry` are plain Python **dataclasses**, not Pydantic models.
  Validation is structural (field presence/type checks), not Pydantic `.model_validate()`.
- `MapperOutput` and `CoachOutput` are Pydantic models — use `.model_validate(obj.model_dump())`
  to re-validate after construction.
- The pipeline does not modify `mentor.py` or `coach.py`.
- The pipeline's own `--path`/`--url` args are passed through to `mapper.main()` unchanged.
- The fixed sample questions should be representative of real onboarding questions
  (e.g. "What are the main entry points?", "How do I run the tests?",
  "What language and framework does this project use?").
- `datetime.utcnow()` is used in `mentor_schema.py` for timestamps; replicate that pattern
  in the pipeline's `SessionEntry` construction.
