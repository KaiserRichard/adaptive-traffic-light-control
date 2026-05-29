"""
file_io.py

Utility functions for ATLC cryptographic research outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_bytes(path: Path, data: bytes) -> None:
    ensure_parent(path)
    path.write_bytes(data)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")
