"""
Índice vectorial FAISS con embeddings locales (sentence-transformers) para RAG BRE-B.

Incluye ``FAISSVectorStore``, construcción del índice desde PDF y utilidad ``build_breb_index``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _project_root() -> Path:
    """Raíz del proyecto (carpeta que contiene ``data/`` y ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _l2_distance_to_similarity(distance: float) -> float:
    """Convierte distancia L2 en un score tipo similitud (mayor = más parecido)."""
    return float(1.0 / (1.0 + max(0.0, float(distance))))


class FAISSVectorStore:
    """
    Vector store FAISS con ``HuggingFaceEmbeddings`` local (CPU, sin APIs externas).

    Persistencia compatible con ``save_local`` / ``load_local`` de LangChain.
    """

    def __init__(
        self,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """
        Args:
            embedding_model_name: Identificador del modelo en sentence-transformers / HF.
        """
        self._embedding_model_name = embedding_model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._store: Optional[FAISS] = None
        self._chunks: List[str] = []

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info("Cargando embeddings locales: %s (CPU)", self._embedding_model_name)
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self._embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception:
                logger.exception("No se pudo cargar HuggingFaceEmbeddings.")
                raise
        return self._embeddings

    def build_index(self, chunks: List[str]) -> None:
        """
        Genera embeddings, crea el índice FAISS y guarda los chunks como contenido de documentos.

        Args:
            chunks: Fragmentos de texto (p. ej. salida de ``chunk_text``).

        Raises:
            ValueError: Si ``chunks`` está vacío.
        """
        cleaned = [c.strip() for c in chunks if c and c.strip()]
        if not cleaned:
            logger.error("build_index: lista de chunks vacía.")
            raise ValueError("Se requiere al menos un chunk no vacío.")

        self._chunks = list(cleaned)
        logger.info("Construyendo índice FAISS con %s chunks.", len(self._chunks))

        try:
            emb = self._get_embeddings()
            documents = [
                Document(page_content=text, metadata={"chunk_index": i, "source": "breb_pdf"})
                for i, text in enumerate(self._chunks)
            ]
            self._store = FAISS.from_documents(documents, emb)
        except Exception:
            logger.exception("Fallo al construir el índice FAISS.")
            raise

        logger.info("Índice FAISS creado en memoria.")

    def save_index(self, path: str) -> None:
        """
        Persiste índice y docstore en disco (carpeta destino).

        Args:
            path: Directorio (p. ej. ``data/vectorstore/breb_index``).

        Raises:
            RuntimeError: Si no hay índice construido.
        """
        if self._store is None:
            logger.error("save_index: no hay índice; llama primero a build_index.")
            raise RuntimeError("El índice no está construido.")

        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            self._store.save_local(str(folder.resolve()))
            logger.info("Índice guardado en: %s", folder.resolve())
        except Exception:
            logger.exception("Error al guardar índice en %s", path)
            raise

    def load_index(self, path: str) -> None:
        """
        Carga un índice previamente guardado con ``save_index``.

        Args:
            path: Directorio donde está el índice serializado.
        """
        folder = Path(path)
        if not folder.is_dir():
            logger.error("No existe el directorio de índice: %s", path)
            raise FileNotFoundError(f"No existe el directorio del índice: {path}")

        try:
            emb = self._get_embeddings()
            self._store = FAISS.load_local(
                str(folder.resolve()),
                emb,
                allow_dangerous_deserialization=True,
            )
            self._chunks = []
            logger.info("Índice FAISS cargado desde: %s", folder.resolve())
        except Exception:
            logger.exception("Error al cargar índice desde %s", path)
            raise

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Búsqueda por similitud sobre el índice.

        Args:
            query: Consulta en lenguaje natural.
            top_k: Número de fragmentos a devolver.

        Returns:
            Lista de dicts con ``text`` y ``score`` (similitud derivada de la distancia L2).
        """
        if self._store is None:
            logger.error("similarity_search: índice no inicializado.")
            raise RuntimeError("Construye o carga el índice antes de buscar.")

        if not query.strip():
            return []

        try:
            pairs = self._store.similarity_search_with_score(query, k=top_k)
        except Exception:
            logger.exception("Error en similarity_search_with_score.")
            raise

        results: List[Dict[str, Any]] = []
        for doc, dist in pairs:
            text = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
            results.append(
                {
                    "text": text,
                    "score": _l2_distance_to_similarity(dist),
                }
            )
        logger.debug("similarity_search: %s resultados para top_k=%s", len(results), top_k)
        return results


def build_breb_index(
    pdf_path: Optional[str] = None,
    index_path: Optional[str] = None,
    run_demo_query: bool = False,
) -> None:
    """
    Pipeline completo: PDF → texto → chunks → FAISS → disco.

    Rutas por defecto relativas a la raíz del proyecto:

    * PDF: ``data/docs/documento-tecnico-bre-b-febrero-2026.pdf``
    * Índice: ``data/vectorstore/breb_index``

    Args:
        pdf_path: Ruta al PDF (opcional).
        index_path: Carpeta de salida del vector store (opcional).
        run_demo_query: Si True, ejecuta una búsqueda de prueba e imprime 3 resultados.
    """
    root = _project_root()
    if pdf_path:
        pdf = Path(pdf_path)
    else:
        env_pdf = (os.getenv("BREB_PDF_PATH") or "").strip()
        if env_pdf:
            p = Path(env_pdf)
            pdf = p if p.is_absolute() else (root / p)
        else:
            pdf = root / "data" / "docs" / "documento-tecnico-bre-b-febrero-2026.pdf"

    if index_path:
        out_dir = Path(index_path)
    else:
        env_idx = (os.getenv("BREB_FAISS_INDEX_DIR") or "").strip()
        if env_idx:
            p = Path(env_idx)
            out_dir = p if p.is_absolute() else (root / p)
        else:
            out_dir = root / "data" / "vectorstore" / "breb_index"

    from data_processing.chunking import chunk_text
    from data_processing.load_breb_pdf import load_pdf_text

    logger.info("build_breb_index: PDF=%s | salida=%s", pdf, out_dir)

    try:
        text = load_pdf_text(str(pdf))
        chunks = chunk_text(text)
        store = FAISSVectorStore()
        store.build_index(chunks)
        store.save_index(str(out_dir))
    except FileNotFoundError:
        logger.exception(
            "No se encontró el PDF. Colócalo en: %s",
            pdf,
        )
        raise
    except Exception:
        logger.exception("Fallo el pipeline build_breb_index.")
        raise

    if run_demo_query:
        _run_demo_search(store, "¿Qué es Bre-B?")


def _run_demo_search(store: FAISSVectorStore, query: str) -> None:
    """Imprime los 3 mejores chunks para una consulta de prueba."""
    print("\n--- Demo búsqueda ---")
    print(f"Query: {query!r}\n")
    try:
        hits = store.similarity_search(query, top_k=3)
        for i, h in enumerate(hits, start=1):
            preview = (h.get("text") or "")[:400].replace("\n", " ")
            print(f"[{i}] score={h.get('score', 0):.4f} | {preview}...")
    except Exception:
        logger.exception("Demo search falló.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    build_breb_index(run_demo_query=True)
