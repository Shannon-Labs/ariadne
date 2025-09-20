from __future__ import annotations

from pathlib import Path
from typing import Optional


def write_report(name: str, text: str, folder: Optional[Path] = None) -> Path:
    folder = folder or Path("reports")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.md"
    path.write_text(text)
    return path

