"""Syntax-check Python files in a release copy without importing dependencies.

This uses ast.parse instead of py_compile so it does not create __pycache__
files and does not import heavy training or diffusion dependencies.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import tokenize


def check_python_syntax(root: Path) -> list[tuple[Path, str, str]]:
    failures: list[tuple[Path, str, str]] = []

    for path in sorted(root.rglob("*.py")):
        try:
            with tokenize.open(path) as handle:
                source = handle.read()
            ast.parse(source, filename=str(path))
        except Exception as exc:  # noqa: BLE001 - report every parse/read issue.
            message = str(exc).replace("\n", " ")
            failures.append((path, type(exc).__name__, message))

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="release/VFM-FP-essential",
        help="Release directory to scan.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Release directory not found: {root}", file=sys.stderr)
        return 2

    files = sorted(root.rglob("*.py"))
    failures = check_python_syntax(root)

    print(f"Python files scanned: {len(files)}")
    print(f"Syntax failures: {len(failures)}")
    for path, kind, message in failures:
        print(f"{path}: {kind}: {message}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
