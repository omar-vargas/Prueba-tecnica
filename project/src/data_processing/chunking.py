"""
Chunking para RAG con solapamiento usando RecursiveCharacterTextSplitter (LangChain).
"""

from __future__ import annotations

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 100


def chunk_text(text: str) -> List[str]:
    """
    Divide el texto en chunks con solapamiento fijo (500 / 100).

    Args:
        text: Texto completo del documento.

    Returns:
        Lista de fragmentos no vacíos.
    """
    if not text.strip():
        logger.info("chunk_text: texto vacío, 0 chunks.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_DEFAULT_CHUNK_SIZE,
        chunk_overlap=_DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        chunks = splitter.split_text(text)
    except Exception:
        logger.exception("Error al ejecutar RecursiveCharacterTextSplitter.")
        raise

    chunks = [c.strip() for c in chunks if c.strip()]
    logger.info(
        "Chunking: generados %s chunks (size=%s overlap=%s).",
        len(chunks),
        _DEFAULT_CHUNK_SIZE,
        _DEFAULT_CHUNK_OVERLAP,
    )
    return chunks
