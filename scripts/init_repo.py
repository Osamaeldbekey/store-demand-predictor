"""
Repository scaffolding script for the Store Demand Predictor project.

Creates the full folder structure, Python package markers (__init__.py),
and .gitkeep placeholders. Idempotent: running it twice is safe and will
never overwrite an existing file.

Usage (from the project root):
    python scripts/init_repo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Directory layout (relative to project root)
# ---------------------------------------------------------------------------
DIRECTORIES: list[str] = [
    "data/raw",           # Kaggle CSVs (gitignored)
    "data/processed",     # Parquet artifacts (gitignored)
    "models",             # Trained model artifacts (gitignored)
    "src/data",           # ingestion & cleaning
    "src/features",       # feature engineering
    "src/models",         # training / tuning / inference
    "src/engines",        # business decision engines (ROP, transfers, bundles)
    "src/utils",          # shared helpers (metrics, plotting)
    "app",                # Streamlit entrypoint (app.py goes here)
    "app/pages",          # Streamlit multipage pages
    "notebooks",          # EDA only — throwaway analysis
    "scripts",            # operational scripts (this one, download, etc.)
    "tests",              # pytest suites
]

# Python packages that need an __init__.py marker
INIT_FILES: list[str] = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/engines/__init__.py",
    "src/utils/__init__.py",
]

# Data/artifact dirs that git cannot track empty → add .gitkeep
GITKEEP_DIRS: list[str] = [
    "data/raw",
    "data/processed",
    "models",
    "app/pages",
    "notebooks",
]


def build_tree(root: Path = PROJECT_ROOT) -> None:
    created = 0

    for rel in DIRECTORIES:
        path = root / rel
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [dir ]  {rel}/")
            created += 1

    for rel in INIT_FILES:
        path = root / rel
        if not path.exists():
            path.write_text("", encoding="utf-8")
            print(f"  [file]  {rel}")
            created += 1

    for rel in GITKEEP_DIRS:
        path = root / rel / ".gitkeep"
        if not path.exists():
            path.write_text("", encoding="utf-8")
            print(f"  [file]  {rel}/.gitkeep")
            created += 1

    if created == 0:
        print("Repository structure already complete - nothing to do.")
    else:
        print(f"\nCreated {created} missing entries.")


def print_tree(root: Path = PROJECT_ROOT, indent: str = "  ") -> None:
    """Print a compact view of the repository tree (2 levels deep)."""
    print(f"\n{indent}{root.name}/")
    _walk(root, indent, depth=0, max_depth=2)


def _walk(path: Path, indent: str, depth: int, max_depth: int) -> None:
    if depth >= max_depth:
        return
    entries = sorted(
        [p for p in path.iterdir() if p.name != ".git" and not p.name.startswith(".")],
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    for entry in entries:
        suffix = "/" if entry.is_dir() else ""
        print(f"{indent}{indent * depth}- {entry.name}{suffix}")
        if entry.is_dir():
            _walk(entry, indent, depth + 1, max_depth)


if __name__ == "__main__":
    print(f"Scaffolding project at: {PROJECT_ROOT}\n")
    build_tree()
    print_tree()
    print("\nDone. Next steps:")
    print("  1. git init && git add -A && git commit -m 'chore: scaffold repo structure'")
    print("  2. pip install -r requirements.txt")
    print("  3. Download the Kaggle dataset into data/raw/ (see README)")
    print(f"\nPython: {sys.version.split()[0]}")
