import os
import sys
import tempfile
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mapper import walk_repo, classify_paths, generate_setup_guide, rank_by_imports
from mapper_schema import Category


# ---------------------------------------------------------------------------
# 1. Happy Path Test
# ---------------------------------------------------------------------------
def test_mapper_happy_path():
    """Verify that a standard project structure is correctly classified and ranked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock project files
        req_path = os.path.join(tmpdir, "requirements.txt")
        with open(req_path, "w", encoding="utf-8") as f:
            f.write("fastapi\npydantic\n")

        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Sample Project")

        main_py = os.path.join(tmpdir, "main.py")
        with open(main_py, "w", encoding="utf-8") as f:
            f.write("import utils\nprint('hello')\n")

        utils_py = os.path.join(tmpdir, "utils.py")
        with open(utils_py, "w", encoding="utf-8") as f:
            f.write("def helper(): pass\n")

        # 1. Directory discovery
        files = list(walk_repo(tmpdir))
        assert "main.py" in files
        assert "utils.py" in files
        assert "requirements.txt" in files

        # 2. Setup generation
        setup = generate_setup_guide(tmpdir)
        assert setup.language == "python"
        assert setup.dependencies_file == "requirements.txt"
        assert "pip install -r requirements.txt" in setup.run_steps

        # 3. Path Classification
        classified = classify_paths(files, repo_root=tmpdir)
        assert classified["requirements.txt"]["category"] == Category.CONFIGURATION.value
        assert classified["README.md"]["category"] == Category.DOCUMENTATION.value
        assert classified["main.py"]["category"] == Category.CORE_LOGIC.value

        # 4. AST Import Ranking (utils.py is imported by main.py)
        ranked = rank_by_imports(files, tmpdir)
        utils_entry = next((e for e in ranked if e.file == "utils.py"), None)
        assert utils_entry is not None
        assert utils_entry.import_count == 1


# ---------------------------------------------------------------------------
# 2. Edge Case: Empty Repository
# ---------------------------------------------------------------------------
def test_mapper_empty_directory():
    """Verify mapper handles an empty directory gracefully without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = list(walk_repo(tmpdir))
        assert files == []

        setup = generate_setup_guide(tmpdir)
        assert setup.language == "unknown"
        assert setup.dependencies_file is None
        assert setup.run_steps == []

        ranked = rank_by_imports([], tmpdir)
        assert ranked == []


# ---------------------------------------------------------------------------
# 3. Edge Case: Unclear File Fallback
# ---------------------------------------------------------------------------
def test_mapper_unclear_item_fallback():
    """Verify that unsupported file extensions fall back to 'unclear' safely."""
    classified = classify_paths(["data/wallet.csv", "sample.unsupported_ext"])
    assert classified["data/wallet.csv"]["category"] == Category.UNCLEAR.value
    assert classified["data/wallet.csv"]["reason"] == "no rule matched"
    assert classified["sample.unsupported_ext"]["category"] == Category.UNCLEAR.value