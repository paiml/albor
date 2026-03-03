#!/usr/bin/env python3
"""Convert a directory of Python source files into Parquet format.

Schema matches Tier 1 data: file (Utf8), source (Utf8), text (Utf8).
Each Python file becomes one row.

Usage:
    python scripts/source-to-parquet.py ~/src/pytorch pytorch data/parquet/tier2/pytorch.parquet
"""
import sys
import os
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("pip install pyarrow", file=sys.stderr)
    sys.exit(1)


SKIP_DIRS = frozenset({"node_modules", "__pycache__", "venv", ".venv"})


def _should_skip(py_file: Path) -> bool:
    return any(p.startswith(".") for p in py_file.parts) or bool(SKIP_DIRS & set(py_file.parts))


def _read_file(py_file: Path, max_size: int) -> str | None:
    try:
        text = py_file.read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError):
        return None
    return text if 50 <= len(text) <= max_size else None


def collect_python_files(root: Path, source_name: str, max_file_size: int = 100_000):
    files, sources, texts = [], [], []
    for py_file in sorted(root.rglob("*.py")):
        if _should_skip(py_file):
            continue
        text = _read_file(py_file, max_file_size)
        if text is None:
            continue
        files.append(str(py_file.relative_to(root)))
        sources.append(source_name)
        texts.append(text)
    return files, sources, texts


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_dir> <source_name> <output.parquet>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    source_name = sys.argv[2]
    output_path = Path(sys.argv[3])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    files, sources, texts = collect_python_files(input_dir, source_name)
    print(f"Collected {len(files)} Python files from {input_dir}")

    table = pa.table({
        "file": pa.array(files, type=pa.utf8()),
        "source": pa.array(sources, type=pa.utf8()),
        "text": pa.array(texts, type=pa.utf8()),
    })

    pq.write_table(table, str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Written {output_path} ({size_mb:.1f} MB, {len(files)} rows)")


if __name__ == "__main__":
    main()
