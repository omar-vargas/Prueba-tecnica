"""
Construye el índice FAISS del documento BRE-B (PDF local → embeddings → disco).

Uso desde la carpeta ``project/``::

    $env:PYTHONPATH="src"
    .\\.venv\\Scripts\\python.exe scripts\\build_breb_index.py

Opciones::

    python scripts/build_breb_index.py --pdf ruta\\al.pdf --index data\\vectorstore\\mi_index --no-demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BRE-B FAISS index (local embeddings).")
    parser.add_argument("--pdf", type=str, default=None, help="Ruta al PDF BRE-B")
    parser.add_argument("--index", type=str, default=None, help="Carpeta de salida del índice")
    parser.add_argument("--no-demo", action="store_true", help="No ejecutar búsqueda de prueba")
    args = parser.parse_args()

    from storage.faiss_index import build_breb_index

    try:
        build_breb_index(
            pdf_path=args.pdf,
            index_path=args.index,
            run_demo_query=not args.no_demo,
        )
    except FileNotFoundError:
        return 1
    except Exception:
        logging.exception("build_breb_index terminó con error.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
