"""
Punto de entrada para Uvicorn: añade `src/` al path de importación.

Evita errores si se escribe mal `--app-dir` (p. ej. `sr` en lugar de `src`).

Uso desde la carpeta `project/`::

    uvicorn run:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_src))

from api.main import app  # noqa: E402

__all__ = ["app"]
