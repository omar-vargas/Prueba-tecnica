"""
Herramienta RAG sobre documentación técnica BRE-B (FAISS + embeddings locales).

Carga el índice persistido (misma carpeta que genera ``build_breb_index``) y recupera
los fragmentos más similares a la pregunta.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from storage.faiss_index import FAISSVectorStore
from utils.settings import get_settings

logger = logging.getLogger(__name__)


def _index_files_present(index_dir: Path) -> bool:
    """Comprueba que LangChain haya guardado ``index.faiss`` y ``index.pkl``."""
    if not index_dir.is_dir():
        return False
    return (index_dir / "index.faiss").is_file() and (index_dir / "index.pkl").is_file()


@lru_cache(maxsize=4)
def _get_loaded_store(resolved_index_path: str) -> FAISSVectorStore:
    """
    Carga el vector store una vez por ruta (caché en proceso).

    Args:
        resolved_index_path: Ruta absoluta normalizada del directorio del índice.
    """
    store = FAISSVectorStore()
    store.load_index(resolved_index_path)
    return store


def clear_breb_vector_cache() -> None:
    """Vacía la caché de índice (útil tras regenerar FAISS en caliente)."""
    _get_loaded_store.cache_clear()


def warmup_breb_vector_index() -> bool:
    """
    Precarga el índice FAISS en memoria si los archivos existen.

    Returns:
        True si el warmup tuvo éxito.
    """
    settings = get_settings()
    index_dir = settings.breb_faiss_index_dir.resolve()
    if not _index_files_present(index_dir):
        return False
    try:
        _get_loaded_store(str(index_dir))
        return True
    except Exception:
        logger.exception("warmup_breb_vector_index falló.")
        return False


def _missing_index_help(index_dir: Path) -> str:
    return (
        "No hay índice vectorial BRE-B listo: falta la carpeta o los archivos "
        "`index.faiss` / `index.pkl`.\n\n"
        "**Qué hacer**\n\n"
        "1. Coloca el PDF técnico BRE-B en `data/docs/` con el nombre por defecto "
        "`documento-tecnico-bre-b-febrero-2026.pdf`, o define en `.env` la variable "
        "`BREB_PDF_PATH` con la ruta a tu PDF.\n\n"
        "2. Desde la carpeta `project/`, ejecuta (con el venv activado):\n\n"
        "```\n"
        "python -c \"import sys; sys.path.insert(0,'src'); "
        "from storage.faiss_index import build_breb_index; build_breb_index()\"\n"
        "```\n\n"
        "Eso lee el PDF, crea chunks, calcula embeddings (sentence-transformers, CPU) "
        "y guarda el índice.\n\n"
        f"3. La carpeta de salida debe ser **`{index_dir}`** "
        "(variable `BREB_FAISS_INDEX_DIR` si quieres otra ruta).\n\n"
        "4. Reinicia Uvicorn después de generar el índice la primera vez."
    )


def query_breb_document(
    question: str,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Recupera fragmentos del PDF BRE-B relevantes para la pregunta vía FAISS.

    Args:
        question: Consulta en lenguaje natural.
        llm: Reservado para re-ranking o síntesis futura.

    Returns:
        Dict con ``content`` (texto agregado para el compositor), metadatos y
        ``chunks_retrieved``.
    """
    _ = llm
    settings = get_settings()
    index_dir = settings.breb_faiss_index_dir.resolve()
    top_k = settings.breb_rag_top_k
    path_str = str(index_dir)

    if not question.strip():
        return {
            "content": "Indica una pregunta sobre el documento BRE-B.",
            "source": "breb_pdf_rag",
            "indexes_dir": path_str,
            "chunks_retrieved": 0,
        }

    if not _index_files_present(index_dir):
        logger.warning(
            "Índice BRE-B no encontrado o incompleto en %s (¿ejecutaste build_breb_index?).",
            index_dir,
        )
        return {
            "content": _missing_index_help(index_dir),
            "source": "breb_pdf_rag",
            "indexes_dir": path_str,
            "chunks_retrieved": 0,
            "index_missing": True,
        }

    try:
        store = _get_loaded_store(path_str)
        hits: List[Dict[str, Any]] = store.similarity_search(question.strip(), top_k=top_k)
    except Exception:
        logger.exception("Fallo en búsqueda vectorial BRE-B (FAISS).")
        _get_loaded_store.cache_clear()
        return {
            "content": (
                "No se pudo consultar el índice vectorial BRE-B. "
                "Revisa los logs del servidor y que el índice se generó con el mismo "
                "modelo de embeddings (sentence-transformers/all-MiniLM-L6-v2)."
            ),
            "source": "breb_pdf_rag",
            "indexes_dir": path_str,
            "chunks_retrieved": 0,
            "error": True,
        }

    if not hits:
        return {
            "content": (
                "La búsqueda en el documento técnico no devolvió fragmentos. "
                "Prueba con términos del dominio: BRE-B, interoperabilidad, transferencias, QR, MOL, DICE."
            ),
            "source": "breb_pdf_rag",
            "indexes_dir": path_str,
            "chunks_retrieved": 0,
        }

    parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        txt = (h.get("text") or "").strip()
        score = float(h.get("score") or 0.0)
        parts.append(f"### Fragmento {i} (relevancia: {score:.3f})\n{txt}")

    content = "\n\n".join(parts)
    logger.info(
        "query_breb_document: %s fragmentos (top_k=%s) para pregunta len=%s",
        len(hits),
        top_k,
        len(question),
    )

    return {
        "content": content,
        "source": "breb_pdf_rag",
        "indexes_dir": path_str,
        "chunks_retrieved": len(hits),
        "scores": [float(h.get("score") or 0.0) for h in hits],
    }
