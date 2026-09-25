"""
test_mentor.py
==============
Smoke-tests for mentor.py.

Run from the src/ directory:
    python test_mentor.py

Requires: scikit-learn, numpy
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest

# Make sure src/ is on the path when running from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from mentor import (
    INSUFFICIENT_INFO_RESPONSE,
    RELEVANCE_THRESHOLD,
    _chunk_mapper_output,
    _chunk_source_file,
    _content_chunks,
    _fixed_chunks,
    _python_chunks,
    ask,
    build_knowledge_base,
    retrieve,
    save_session,
)
from mentor_schema import Confidence, MentorOutput, SessionEntry
from mapper_schema import EntryPoint, MapperOutput, SetupInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_MAPPER = {
    "entry_points": [
        {
            "name": "main",
            "file_path": "app/main.py",
            "line_number": 1,
            "description": "Application entry point that starts the server.",
            "metadata": {"framework": "FastAPI"},
        }
    ],
    "classification": {
        "app/main.py": "service",
        "app/models.py": "model",
    },
    "setup_info": {
        "language": "Python",
        "framework": "FastAPI",
        "dependencies": ["fastapi", "sqlalchemy"],
        "notes": "Uses async handlers throughout.",
    },
}

SAMPLE_PYTHON = textwrap.dedent(
    """\
    import os

    CONSTANT = 42

    def hello(name: str) -> str:
        \"\"\"Return a greeting.\"\"\"
        return f"Hello, {name}"

    class Greeter:
        def greet(self, name: str) -> str:
            return hello(name)
    """
)

SAMPLE_TEXT = "This is a plain text file with no Python syntax.\n" * 40


def _make_mapper_file(tmp_dir: str, data: dict | None = None) -> str:
    data = data or SAMPLE_MAPPER
    path = os.path.join(tmp_dir, "mapper_output.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_repo(tmp_dir: str) -> str:
    """Create a minimal fake repo matching SAMPLE_MAPPER's classification."""
    os.makedirs(os.path.join(tmp_dir, "app"), exist_ok=True)
    with open(os.path.join(tmp_dir, "app", "main.py"), "w") as f:
        f.write(SAMPLE_PYTHON)
    with open(os.path.join(tmp_dir, "app", "models.py"), "w") as f:
        f.write("# models\nclass User:\n    pass\n")
    return tmp_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFixedChunks(unittest.TestCase):
    def test_produces_chunks(self):
        chunks = _fixed_chunks(SAMPLE_TEXT, "readme.txt", "code")
        self.assertGreater(len(chunks), 0)

    def test_chunk_fields(self):
        chunks = _fixed_chunks("hello world " * 100, "x.txt", "code", chunk_size=50, overlap=10)
        for c in chunks:
            self.assertIn("text", c)
            self.assertIn("source_path", c)
            self.assertIn("source_type", c)
            self.assertEqual(c["source_type"], "code")
            self.assertEqual(c["source_path"], "x.txt")

    def test_overlap(self):
        text = "A" * 200
        chunks = _fixed_chunks(text, "f.txt", "code", chunk_size=100, overlap=50)
        # step = 100 - 50 = 50 → expect 4 chunks for 200 chars
        self.assertEqual(len(chunks), 4)

    def test_empty_text(self):
        chunks = _fixed_chunks("", "empty.txt", "code")
        self.assertEqual(chunks, [])


class TestPythonChunks(unittest.TestCase):
    def test_functions_become_chunks(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("hello" in t for t in texts))

    def test_class_becomes_chunk(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("Greeter" in t for t in texts))

    def test_preamble_chunk(self):
        chunks = _python_chunks(SAMPLE_PYTHON, "sample.py")
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("preamble" in t for t in texts))

    def test_syntax_error_falls_back(self):
        bad_code = "def broken(\n  pass"
        chunks = _python_chunks(bad_code, "bad.py")
        self.assertGreater(len(chunks), 0)  # fixed-size fallback


class TestChunkSourceFile(unittest.TestCase):
    def test_py_file_uses_ast(self):
        chunks = _chunk_source_file(SAMPLE_PYTHON, "s.py")
        self.assertTrue(any("hello" in c["text"] for c in chunks))

    def test_non_py_uses_fixed(self):
        chunks = _chunk_source_file(SAMPLE_TEXT, "readme.md")
        for c in chunks:
            self.assertEqual(c["source_type"], "code")


class TestChunkMapperOutput(unittest.TestCase):
    def setUp(self):
        self.mo = MapperOutput.from_dict(SAMPLE_MAPPER)

    def test_entry_point_chunk(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("main" in t for t in texts))
        self.assertTrue(any("entry point" in t.lower() for t in texts))

    def test_classification_chunk(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("classification" in t.lower() for t in texts))

    def test_setup_info_chunk(self):
        chunks = _chunk_mapper_output(self.mo)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("FastAPI" in t for t in texts))

    def test_all_mapper_type(self):
        chunks = _chunk_mapper_output(self.mo)
        for c in chunks:
            self.assertEqual(c["source_type"], "mapper_summary")


class TestBuildKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mapper_path = _make_mapper_file(self.tmp)
        self.repo_root = _make_repo(self.tmp)

    def test_returns_list(self):
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        self.assertIsInstance(chunks, list)

    def test_has_content_chunks(self):
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        content = _content_chunks(chunks)
        self.assertGreater(len(content), 0)

    def test_vectors_attached(self):
        import numpy as np
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        for c in _content_chunks(chunks):
            self.assertIn("vector", c)
            self.assertIsInstance(c["vector"], np.ndarray)

    def test_vectorizer_sentinel(self):
        chunks = build_knowledge_base(self.mapper_path, self.repo_root)
        self.assertIn("__vectorizer__", chunks[-1])

    def test_missing_source_file_is_skipped(self):
        # Add a classification entry for a non-existent file
        data = dict(SAMPLE_MAPPER)
        data["classification"] = {**data["classification"], "app/ghost.py": "util"}
        mapper_path = _make_mapper_file(self.tmp, data)
        # Should NOT raise, just warn
        chunks = build_knowledge_base(mapper_path, self.repo_root)
        self.assertIsInstance(chunks, list)


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mapper_path = _make_mapper_file(self.tmp)
        self.repo_root = _make_repo(self.tmp)
        self.chunks = build_knowledge_base(self.mapper_path, self.repo_root)

    def test_returns_list(self):
        results = retrieve("hello function", self.chunks)
        self.assertIsInstance(results, list)

    def test_top_k_respected(self):
        results = retrieve("hello", self.chunks, top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_score_present(self):
        results = retrieve("greeting", self.chunks)
        for r in results:
            self.assertIn("score", r)
            self.assertGreaterEqual(r["score"], 0.0)

    def test_sorted_descending(self):
        results = retrieve("hello function greeting", self.chunks, top_k=5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_vector_in_result(self):
        results = retrieve("hello", self.chunks, top_k=1)
        for r in results:
            self.assertNotIn("vector", r)


class TestAsk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mapper_path = _make_mapper_file(self.tmp)
        self.repo_root = _make_repo(self.tmp)
        self.chunks = build_knowledge_base(self.mapper_path, self.repo_root)

    def test_returns_string(self):
        answer = ask("What does the hello function do?", self.chunks)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_grounded_answer_contains_context(self):
        answer = ask("What does the hello function do?", self.chunks)
        if answer != INSUFFICIENT_INFO_RESPONSE:
            self.assertIn("hello", answer)

    def test_irrelevant_question_returns_insufficient(self):
        # A completely unrelated question with no matching tokens in the KB
        answer = ask(
            "zzzzqqqqqxxxx irrelevant gobbledygook unrelated 12345",
            self.chunks,
        )
        self.assertEqual(answer, INSUFFICIENT_INFO_RESPONSE)

    def test_top_k_param_forwarded(self):
        # Should not raise with top_k=1
        answer = ask("hello", self.chunks, top_k=1)
        self.assertIsInstance(answer, str)


class TestSaveSession(unittest.TestCase):
    def test_writes_json(self):
        entries = [
            SessionEntry(
                question="What does hello do?",
                answer="It greets a name.",
                developer_confidence=Confidence.HIGH,
                timestamp="2024-01-01T00:00:00",
            )
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            out_path = f.name

        try:
            save_session(entries, output_path=out_path)
            loaded = MentorOutput.from_json_file(out_path)
            self.assertEqual(len(loaded.entries), 1)
            self.assertEqual(loaded.entries[0].question, "What does hello do?")
            self.assertEqual(loaded.entries[0].developer_confidence, Confidence.HIGH)
            self.assertIn("total_questions", loaded.session_metadata)
        finally:
            os.unlink(out_path)

    def test_empty_entries(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            out_path = f.name
        try:
            save_session([], output_path=out_path)
            loaded = MentorOutput.from_json_file(out_path)
            self.assertEqual(loaded.entries, [])
        finally:
            os.unlink(out_path)


class TestMentorSchema(unittest.TestCase):
    def test_confidence_values(self):
        self.assertEqual(Confidence.HIGH.value, "high")
        self.assertEqual(Confidence.MEDIUM.value, "medium")
        self.assertEqual(Confidence.LOW.value, "low")

    def test_session_entry_roundtrip(self):
        entry = SessionEntry(
            question="q",
            answer="a",
            developer_confidence=Confidence.LOW,
            timestamp="2024-01-01T00:00:00",
        )
        d = entry.to_dict()
        restored = SessionEntry.from_dict(d)
        self.assertEqual(restored.question, "q")
        self.assertEqual(restored.developer_confidence, Confidence.LOW)

    def test_mentor_output_roundtrip(self):
        mo = MentorOutput(
            entries=[
                SessionEntry("q", "a", Confidence.MEDIUM, "2024-01-01T00:00:00")
            ],
            session_metadata={"key": "val"},
        )
        d = mo.to_dict()
        restored = MentorOutput.from_dict(d)
        self.assertEqual(len(restored.entries), 1)
        self.assertEqual(restored.session_metadata["key"], "val")


class TestMapperSchema(unittest.TestCase):
    def test_from_dict_roundtrip(self):
        mo = MapperOutput.from_dict(SAMPLE_MAPPER)
        d = mo.to_dict()
        restored = MapperOutput.from_dict(d)
        self.assertEqual(len(restored.entry_points), 1)
        self.assertEqual(restored.entry_points[0].name, "main")
        self.assertEqual(restored.setup_info.framework, "FastAPI")

    def test_from_json_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(SAMPLE_MAPPER, f)
            path = f.name
        try:
            mo = MapperOutput.from_json_file(path)
            self.assertEqual(mo.classification["app/main.py"], "service")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
