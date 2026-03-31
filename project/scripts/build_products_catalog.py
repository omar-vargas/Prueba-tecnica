"""
Genera ``data/processed/products_catalog.json`` desde el PDF de portafolio (requiere LLM en .env).

Uso desde ``project/``::

    $env:PYTHONPATH="src"
    .\\.venv\\Scripts\\python.exe scripts\\build_products_catalog.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> int:
    from data_processing.load_products_pdf import _print_extraction_summary, build_products_catalog
    from utils.settings import build_chat_llm

    try:
        llm = build_chat_llm()
        products = build_products_catalog(llm)
        _print_extraction_summary(products)
    except Exception:
        logging.exception("build_products_catalog falló.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
