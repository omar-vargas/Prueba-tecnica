"""
Extracción de texto del PDF técnico BRE-B para chunking e indexación FAISS.

Usa ``PyPDFLoader`` de LangChain (todas las páginas unidas en un solo string).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf_text(pdf_path: str) -> str:
    """
    Extrae todo el texto de un PDF y lo devuelve como un único string.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Texto completo con espacios colapsados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(pdf_path)
    if not path.is_file():
        logger.error("PDF no encontrado: %s", pdf_path)
        raise FileNotFoundError(f"No existe el PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(str(path.resolve()))
        documents = loader.load()
    except Exception:
        logger.exception("Fallo PyPDFLoader al leer: %s", pdf_path)
        raise

    n_pages = len(documents)
    parts: List[str] = []
    for doc in documents:
        page_text = doc.page_content or ""
        parts.append(page_text)

    raw = "\n".join(parts)
    cleaned = re.sub(r"[ \t]+", " ", raw)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    logger.info(
        "PDF cargado: %s | páginas=%s | longitud_texto=%s",
        pdf_path,
        n_pages,
        len(cleaned),
    )
    return cleaned


def extract_text_pages(pdf_path: Path) -> List[str]:
    """
    Compatibilidad: devuelve una lista con el texto por página.

    Args:
        pdf_path: Ruta al PDF.

    Returns:
        Lista de strings (una por página).
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el PDF: {pdf_path}")
    loader = PyPDFLoader(str(path.resolve()))
    docs = loader.load()
    return [d.page_content or "" for d in docs]
