"""
Repository discovery layer for the Onboarding Copilot pipeline.
Owner: Maira (Mapper Agent)

Accepts a local path (--path) or a remote git URL (--url), ensures the
repo is on disk, then walks the tree and prints every non-ignored file
path to stdout (one per line, repo-relative, forward-slash normalised).
No classification, no LLM calls.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch as _fnmatch
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator

from mapper_schema import (
    Category,
    ClassificationEntry,
    Confidence,
    EntryPoint,
    MapperOutput,
    SetupInfo,
    UnclearItem,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLONE_DEST: str = "data/target_repo"

SKIP_DIRS: frozenset[str] = frozenset({".git", "node_modules", "venv", "__pycache__"})

IMPORT_CONFIDENCE_HIGH: int = 5
IMPORT_CONFIDENCE_MEDIUM: int = 2


# ---------------------------------------------------------------------------
# Classification rule tables
# ---------------------------------------------------------------------------

# Folder-level rules: a path segment (not the filename) that pins a category.
_FOLDER_CATEGORIES: dict[str, Category] = {
    # core_logic
    "src":     Category.CORE_LOGIC,
    "lib":     Category.CORE_LOGIC,
    "core":    Category.CORE_LOGIC,
    "app":     Category.CORE_LOGIC,
    "engine":  Category.CORE_LOGIC,
    "modules": Category.CORE_LOGIC,
    # testing
    "test":      Category.TESTING,
    "tests":     Category.TESTING,
    "__tests__": Category.TESTING,
    "spec":      Category.TESTING,
    "specs":     Category.TESTING,
    # documentation
    "docs":          Category.DOCUMENTATION,
    "doc":           Category.DOCUMENTATION,
    "documentation": Category.DOCUMENTATION,
    # configuration
    "config":  Category.CONFIGURATION,
    "configs": Category.CONFIGURATION,
    ".github": Category.CONFIGURATION,
    ".vscode": Category.CONFIGURATION,
}

# File-level pattern rules (testing / docs / config).
# Evaluated with fnmatch; first match wins within this list.
_FILE_RULES: list[tuple[Category, str]] = [
    # testing
    (Category.TESTING, "test_*.py"),
    (Category.TESTING, "*_test.py"),
    (Category.TESTING, "*.spec.js"),
    (Category.TESTING, "*.test.js"),
    # documentation
    (Category.DOCUMENTATION, "README*"),
    (Category.DOCUMENTATION, "CHANGELOG*"),
    (Category.DOCUMENTATION, "LICENSE*"),
    (Category.DOCUMENTATION, "*.md"),
    (Category.DOCUMENTATION, "*.rst"),
    # configuration
    (Category.CONFIGURATION, "*.yml"),
    (Category.CONFIGURATION, "*.yaml"),
    (Category.CONFIGURATION, "*.toml"),
    (Category.CONFIGURATION, "*.ini"),
    (Category.CONFIGURATION, "*.cfg"),
    (Category.CONFIGURATION, ".env*"),
    (Category.CONFIGURATION, "Dockerfile"),
    (Category.CONFIGURATION, "docker-compose*"),
    (Category.CONFIGURATION, "requirements.txt"),
    (Category.CONFIGURATION, "package.json"),
    (Category.CONFIGURATION, "pyproject.toml"),
    (Category.CONFIGURATION, "setup.py"),
]

# Root-level filenames that classify as core_logic.
# __init__.py at root also requires a non-empty size check (done at runtime).
_ROOT_CORE_LOGIC_FILES: frozenset[str] = frozenset({
    "main.py", "app.py", "index.js", "server.py", "__init__.py",
})


# ---------------------------------------------------------------------------
# Sub-Task 4a — Path classifier
# ---------------------------------------------------------------------------

def classify_paths(
    files: list[str],
    repo_root: str | None = None,
) -> dict[str, dict[str, str]]:
    """Classify each path in *files* into a category using pure pattern matching.

    Args:
        files:      Repo-relative, forward-slash-normalised paths (as yielded
                    by :func:`walk_repo`).
        repo_root:  Repository root on disk.  Required only for the
                    ``__init__.py``-non-empty check; omit (or pass ``None``)
                    to skip the size check (the file will still classify as
                    ``core_logic`` when it sits at the repo root).

    Returns:
        A ``dict`` mapping each path to ``{"category": str, "reason": str}``.

    Classification rules (in precedence order):

    1. **File-level rules** — root-core-logic exact names, then filename
       patterns (testing / docs / config).
    2. **Folder-level rules** — first matching folder segment wins.
    3. If both tiers fire and **agree** → use the file-level reason.
    4. If both tiers fire and **disagree** → ``unclear`` /
       ``ambiguous: matched X and Y``.
    5. If nothing matches → ``unclear`` / ``no rule matched``.
    """
    result: dict[str, dict[str, str]] = {}

    for path in files:
        parts = path.split("/")
        filename = parts[-1]
        folder_parts = parts[:-1]  # every segment except the filename

        # ------------------------------------------------------------------ #
        # 1. File-level match                                                 #
        # ------------------------------------------------------------------ #
        file_category: Category | None = None
        file_reason: str = ""

        # 1a. Root-level core-logic exact names (no parent folders).
        if len(folder_parts) == 0 and filename in _ROOT_CORE_LOGIC_FILES:
            if filename == "__init__.py":
                # Non-empty guard: only classify if we can confirm size > 0.
                if repo_root is not None:
                    abs_path = os.path.join(repo_root, filename)
                    try:
                        if os.path.getsize(abs_path) > 0:
                            file_category = Category.CORE_LOGIC
                            file_reason = f"root file '{filename}' (non-empty) → core_logic"
                    except OSError:
                        pass  # can't stat → leave unclassified at file level
                else:
                    # No root provided — accept without the size check.
                    file_category = Category.CORE_LOGIC
                    file_reason = f"root file '{filename}' → core_logic"
            else:
                file_category = Category.CORE_LOGIC
                file_reason = f"root file '{filename}' → core_logic"

        # 1b. Filename pattern rules (testing / docs / config).
        if file_category is None:
            for cat, pattern in _FILE_RULES:
                if _fnmatch.fnmatch(filename, pattern):
                    file_category = cat
                    file_reason = f"filename matches '{pattern}' → {cat.value}"
                    break  # first match wins

        # ------------------------------------------------------------------ #
        # 2. Folder-level match (outermost matching segment wins)             #
        # ------------------------------------------------------------------ #
        folder_category: Category | None = None
        folder_reason: str = ""

        for segment in folder_parts:
            if segment in _FOLDER_CATEGORIES:
                folder_category = _FOLDER_CATEGORIES[segment]
                folder_reason = f"folder '{segment}' → {folder_category.value}"
                break

        # ------------------------------------------------------------------ #
        # 3. Combine tiers — file-level takes precedence                      #
        # ------------------------------------------------------------------ #
        if file_category is not None and folder_category is None:
            category: Category = file_category
            reason: str = file_reason

        elif file_category is None and folder_category is not None:
            category = folder_category
            reason = folder_reason

        elif file_category is not None and folder_category is not None:
            if file_category == folder_category:
                # Both tiers agree — prefer the more specific file-level reason.
                category = file_category
                reason = file_reason
            else:
                # Conflict between file rule and folder rule.
                category = Category.UNCLEAR
                reason = (
                    f"ambiguous: matched {folder_category.value} "
                    f"(folder) and {file_category.value} (filename)"
                )

        else:
            # No rule fired at either tier.
            category = Category.UNCLEAR
            reason = "no rule matched"

        result[path] = {"category": category.value, "reason": reason}

    return result


# ---------------------------------------------------------------------------
# Setup guide generator
# ---------------------------------------------------------------------------

# Ordered list of (filename, language, dependencies_file, run_steps).
# Each entry describes one root-level indicator file.  The first matching
# entry sets the primary language and dependencies_file; all matching entries
# contribute their run_steps (in declaration order).
_SETUP_RULES: list[tuple[str, str, str | None, list[str]]] = [
    (
        "requirements.txt",
        "python",
        "requirements.txt",
        ["pip install -r requirements.txt"],
    ),
    (
        "pyproject.toml",
        "python",
        "pyproject.toml",
        ["pip install ."],
    ),
    (
        "package.json",
        "node",
        "package.json",
        ["npm install"],
    ),
    (
        "Dockerfile",
        "docker",
        None,
        ["docker build ."],
    ),
    (
        "docker-compose.yml",
        "docker",
        None,
        ["docker compose up --build"],
    ),
]


def generate_setup_guide(root: str) -> SetupInfo:
    """Detect well-known root-level files and return a :class:`~mapper_schema.SetupInfo`.

    Checks for the presence of ``requirements.txt``, ``pyproject.toml``,
    ``package.json``, ``Dockerfile``, and ``docker-compose.yml`` directly
    inside *root*.  No file contents are read; only ``os.path.isfile`` is used.

    - ``language`` is taken from the **first** matched rule (declaration order).
    - ``dependencies_file`` is taken from the **first** matched rule that
      provides one.
    - ``run_steps`` aggregates steps from **all** matched rules in order.

    If no indicator file is present, ``language`` defaults to ``"unknown"`` and
    both ``dependencies_file`` and ``run_steps`` are empty / ``None``.
    """
    matched_language: str | None = None
    matched_deps_file: str | None = None
    run_steps: list[str] = []

    for filename, language, deps_file, steps in _SETUP_RULES:
        if os.path.isfile(os.path.join(root, filename)):
            if matched_language is None:
                matched_language = language
            if matched_deps_file is None and deps_file is not None:
                matched_deps_file = deps_file
            run_steps.extend(steps)

    return SetupInfo(
        language=matched_language or "unknown",
        dependencies_file=matched_deps_file,
        run_steps=run_steps,
    )


# ---------------------------------------------------------------------------
# Sub-Task 1 — CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk a git repository and print all discoverable file paths.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--path",
        metavar="DIR",
        help="Local path to an existing repository.",
    )
    group.add_argument(
        "--url",
        metavar="URL",
        help="Remote git URL to shallow-clone into data/target_repo/.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Sub-Task 2 — URL → local path resolution
# ---------------------------------------------------------------------------

def clone_repo(url: str) -> str:
    """Shallow-clone *url* into CLONE_DEST and return that path."""
    if os.path.exists(CLONE_DEST):
        raise FileExistsError(
            f"Clone destination already exists: '{CLONE_DEST}'. "
            "Remove or rename it before cloning a new repository."
        )
    os.makedirs(os.path.dirname(CLONE_DEST), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, CLONE_DEST], check=True)
    return CLONE_DEST


# ---------------------------------------------------------------------------
# Sub-Task 3 — Directory walker
# ---------------------------------------------------------------------------

def walk_repo(root: str) -> Iterator[str]:
    """Yield repo-relative, forward-slash-normalised paths for every file under *root*.

    Directories named .git, node_modules, venv, or __pycache__ are pruned at
    any depth in the tree.
    """
    for dirpath, dirs, filenames in os.walk(root, topdown=True):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
            yield rel_path


# ---------------------------------------------------------------------------
# Sub-Task 5 — Import-based entry-point ranker
# ---------------------------------------------------------------------------

def _build_module_index(files: list[str]) -> dict[str, str]:
    """Map every importable dotted name for each .py file to its repo-relative path.

    For example, ``src/utils/helpers.py`` produces keys:
      ``src.utils.helpers``, ``utils.helpers``, ``helpers``

    ``pkg/__init__.py`` produces keys ``pkg.__init__``, ``__init__``, and ``pkg``
    (the ``pkg`` key is emitted as the special package-root alias).
    """
    index: dict[str, str] = {}
    for file in files:
        if not file.endswith(".py"):
            continue
        # Strip the .py extension and split into segments.
        without_ext = file[:-3]  # e.g. "src/utils/helpers"
        segments = without_ext.split("/")  # ["src", "utils", "helpers"]

        # Emit all suffix sub-paths as dotted names.
        for start in range(len(segments)):
            key = ".".join(segments[start:])
            if key not in index:
                index[key] = file

        # Special case: pkg/__init__.py  →  also register "pkg" as a key.
        if segments[-1] == "__init__" and len(segments) >= 2:
            package_key = ".".join(segments[:-1])
            if package_key not in index:
                index[package_key] = file

    return index


def _parse_imports(abs_path: str) -> list[str]:
    """Return all imported module name strings found in *abs_path*.

    Uses ``ast.parse`` to extract:
    - ``import foo.bar``      → ``"foo.bar"``
    - ``from foo.bar import`` → ``"foo.bar"``

    Returns an empty list on any read or parse error.
    """
    try:
        with open(abs_path, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=abs_path)
    except Exception:  # SyntaxError, UnicodeDecodeError, OSError, …
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
                # Also try "module.name" for each imported name so that
                # `from pkg import core` can resolve to pkg/core.py.
                for alias in node.names:
                    names.append(f"{node.module}.{alias.name}")
    return names


def rank_by_imports(
    files: list[str],
    root: str,
    top_n: int | None = None,
) -> list[EntryPoint]:
    """Rank in-repo Python files by how many other in-repo files import them.

    Args:
        files:  Repo-relative, forward-slash-normalised paths (as yielded by
                :func:`walk_repo`).
        root:   Repository root — used to build absolute paths for parsing.
        top_n:  If provided, cap the returned list to this many entries.
                ``None`` (default) returns all Python files.

    Returns:
        A list of :class:`~mapper_schema.EntryPoint` objects sorted by
        ``import_count`` descending.  Confidence is assigned by threshold:

        - ``import_count >= IMPORT_CONFIDENCE_HIGH``   → HIGH
        - ``import_count >= IMPORT_CONFIDENCE_MEDIUM`` → MEDIUM
        - otherwise                                    → LOW
    """
    module_index = _build_module_index(files)

    # Initialise every Python file at zero so files that are never imported
    # still appear in the output (tagged LOW).
    counts: Counter[str] = Counter({f: 0 for f in files if f.endswith(".py")})

    for file in files:
        if not file.endswith(".py"):
            continue
        abs_path = os.path.join(root, file.replace("/", os.sep))
        for module_name in _parse_imports(abs_path):
            resolved = module_index.get(module_name)
            if resolved is not None and resolved != file:
                counts[resolved] += 1

    def _confidence(count: int) -> Confidence:
        if count >= IMPORT_CONFIDENCE_HIGH:
            return Confidence.HIGH
        if count >= IMPORT_CONFIDENCE_MEDIUM:
            return Confidence.MEDIUM
        return Confidence.LOW

    ranked = [
        EntryPoint(file=file, import_count=count, confidence=_confidence(count))
        for file, count in counts.most_common()
    ]

    return ranked[:top_n] if top_n is not None else ranked


# ---------------------------------------------------------------------------
# Sub-Task 4 — main() wiring
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. Resolve root path (clone if URL).
    if args.url:
        root = clone_repo(args.url)
    else:
        root = args.path

    if not os.path.isdir(root):
        print(f"error: '{root}' is not an existing directory.", file=sys.stderr)
        sys.exit(1)

    # 2. Discover all files.
    files: list[str] = list(walk_repo(root))

    # 3. Rank top-10 entry points by inward import count.
    entry_points: list[EntryPoint] = rank_by_imports(files, root, top_n=10)

    # 4. Classify every path and build ClassificationEntry objects.
    raw_classification = classify_paths(files, repo_root=root)
    classification: dict[str, ClassificationEntry] = {
        path: ClassificationEntry(
            category=Category(info["category"]),
            reason=info["reason"],
        )
        for path, info in raw_classification.items()
    }

    # Collect unclear items as a convenience list.
    unclear_items: list[UnclearItem] = [
        UnclearItem(path=path, reason=entry.reason)
        for path, entry in classification.items()
        if entry.category == Category.UNCLEAR
    ]

    # 5. Generate setup guide.
    setup_info: SetupInfo = generate_setup_guide(root)

    # 6. Assemble the final output object.
    mapper_output = MapperOutput(
        repo_root=root,
        entry_points=entry_points,
        classification=classification,
        setup=setup_info,
        unclear_items=unclear_items,
    )

    # 7. Print as JSON.
    print(mapper_output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
