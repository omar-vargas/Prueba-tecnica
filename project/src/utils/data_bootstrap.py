"""
Precarga y preparación de fuentes de datos al arrancar FastAPI.

* SQLite ``reviews``: si no hay filas con comentario, intenta ingerir desde Excel
  (ruta ``REVIEWS_EXCEL_PATH``).
* JSON de productos: lectura en caché para dejar lista la primera consulta.
* FAISS BRE-B: si faltan ``index.faiss`` / ``index.pkl`` pero existe el PDF,
  ejecuta ``build_breb_index``; luego precarga embeddings en memoria.

Si ``STRICT_DATA_BOOTSTRAP=true`` y alguna fuente obligatoria falla, se lanza error
y la API no queda en servicio.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

from data_processing.load_reviews import load_reviews_excel_to_sqlite
from storage.faiss_index import build_breb_index
from storage.product_catalog import load_product_catalog_cached
from tools.breb_rag_tool import (
    clear_breb_vector_cache,
    warmup_breb_vector_index,
)
from tools.reviews_tool import _get_retriever, clear_reviews_retriever_cache
from utils.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)


def _resolve_reviews_excel_file() -> tuple[Path | None, str]:
    """
    Localiza el Excel de reseñas: ruta configurada o búsqueda en ``data/docs`` y ``data/raw``.

    Returns:
        Tupla (ruta si existe, texto para logs / mensajes).
    """
    settings = get_settings()
    primary = settings.reviews_excel_path.resolve()
    if primary.is_file():
        return primary, str(primary)

    candidates: list[Path] = []
    for folder in (PROJECT_ROOT / "data" / "docs", PROJECT_ROOT / "data" / "raw"):
        if not folder.is_dir():
            continue
        for pat in (
            "*bank*review*.xlsx",
            "*review*colombia*.xlsx",
            "*reviews*.xlsx",
            "*.xlsx",
        ):
            for p in sorted(folder.glob(pat)):
                if p.is_file() and not p.name.startswith("~$"):
                    candidates.append(p.resolve())

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if not unique:
        logger.warning(
            "No se encontró ningún .xlsx de reseñas (config=%s ni carpetas docs/raw).",
            primary,
        )
        return None, str(primary)

    for p in unique:
        low = p.name.lower()
        if "review" in low or "reseña" in low or "resena" in low:
            logger.info("Excel de reseñas autodetectado: %s", p)
            return p, str(p)

    chosen = unique[0]
    logger.info("Usando primer Excel encontrado en data/: %s", chosen)
    return chosen, str(chosen)


def _count_nonempty_reviews(db_path: Path) -> int:
    """Cuenta filas en ``reviews`` con comentario no vacío."""
    if not db_path.is_file():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM reviews
                WHERE comment IS NOT NULL AND TRIM(comment) != ''
                """
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()
    except Exception:
        logger.exception("No se pudo contar filas en %s", db_path)
        return 0


def _bootstrap_reviews(report: Dict[str, Any]) -> None:
    settings = get_settings()
    db_path = settings.sqlite_db_path.resolve()

    n = _count_nonempty_reviews(db_path)
    if n > 0:
        report["reviews"] = f"ok_sqlite_rows={n}"
        report["reviews_excel_used"] = "sqlite_already_populated"
        logger.info("Reseñas SQLite listas: %s filas con comentario en %s", n, db_path)
        return

    excel_path, excel_label = _resolve_reviews_excel_file()
    report["reviews_excel_candidate"] = excel_label

    if excel_path is not None and excel_path.is_file():
        logger.info(
            "SQLite sin datos; ingiriendo Excel → SQLite (%s → %s).",
            excel_path,
            db_path,
        )
        try:
            load_reviews_excel_to_sqlite(str(excel_path), str(db_path))
            n = _count_nonempty_reviews(db_path)
            report["reviews"] = f"ingested_rows={n}"
            report["reviews_excel_used"] = str(excel_path)
            logger.info("Ingesta Excel completada: %s filas.", n)
        except Exception:
            logger.exception("Fallo la ingesta de reseñas desde Excel.")
            report["reviews"] = "ingest_error"
            return
    else:
        logger.warning(
            "No hay reseñas en SQLite y no se encontró un Excel válido (última ruta probada: %s). "
            "Coloca el archivo en data/docs o data/raw o define REVIEWS_EXCEL_PATH.",
            excel_label,
        )
        report["reviews"] = "missing_excel_and_empty_sqlite"


def _bootstrap_products(report: Dict[str, Any]) -> None:
    settings = get_settings()
    path = settings.product_catalog_path.resolve()
    try:
        products = load_product_catalog_cached(str(path))
        if not products:
            report["products"] = "missing_or_empty"
            logger.warning("Catálogo de productos vacío o inexistente: %s", path)
        else:
            report["products"] = f"ok_count={len(products)}"
            logger.info("Catálogo JSON cargado: %s productos (%s).", len(products), path)
    except Exception:
        logger.exception("Error al precargar catálogo JSON.")
        report["products"] = "load_error"


def _bootstrap_breb_faiss(report: Dict[str, Any]) -> None:
    settings = get_settings()
    index_dir = settings.breb_faiss_index_dir.resolve()
    pdf_path = settings.breb_pdf_path.resolve()

    clear_breb_vector_cache()

    from tools.breb_rag_tool import _index_files_present

    if _index_files_present(index_dir):
        report["breb_faiss"] = "index_present"
        logger.info("Índice BRE-B encontrado en %s", index_dir)
    elif pdf_path.is_file():
        logger.info(
            "Construyendo índice FAISS BRE-B al arranque (puede tardar varios minutos)…"
        )
        try:
            build_breb_index(pdf_path=str(pdf_path), index_path=str(index_dir))
            report["breb_faiss"] = "built_at_startup"
        except Exception:
            logger.exception("No se pudo construir el índice BRE-B.")
            report["breb_faiss"] = "build_error"
            return
    else:
        logger.warning(
            "Sin índice FAISS en %s y sin PDF en %s. Define BREB_PDF_PATH o genera el índice antes.",
            index_dir,
            pdf_path,
        )
        report["breb_faiss"] = "missing_index_and_pdf"
        return

    if warmup_breb_vector_index():
        report["breb_faiss_warmup"] = "ok"
    else:
        report["breb_faiss_warmup"] = "skipped_or_failed"


def _strict_check(report: Dict[str, Any]) -> None:
    settings = get_settings()
    if not settings.strict_data_bootstrap:
        return

    errors: list[str] = []
    rev = str(report.get("reviews", ""))
    if rev.startswith("missing") or rev == "ingest_error" or rev == "ingested_rows=0":
        errors.append(f"reviews:{rev}")

    if str(report.get("reviews_tfidf_rows")) == "error":
        errors.append("reviews_tfidf:error")

    prod = str(report.get("products", ""))
    if prod.startswith("missing") or prod == "load_error":
        errors.append(f"products:{prod}")

    breb = str(report.get("breb_faiss", ""))
    if "missing" in breb or breb == "build_error":
        errors.append(f"breb_faiss:{breb}")

    if errors:
        msg = "STRICT_DATA_BOOTSTRAP: fuentes no listas → " + "; ".join(errors)
        logger.error(msg)
        raise RuntimeError(msg)


def bootstrap_data_sources() -> Dict[str, Any]:
    """
    Ejecuta la preparación completa de fuentes (llamar una vez al iniciar la app).

    Returns:
        Diccionario resumen para ``app.state`` y ``/health``.
    """
    report: Dict[str, Any] = {}
    logger.info("Iniciando bootstrap de fuentes de datos…")

    _bootstrap_reviews(report)

    clear_reviews_retriever_cache()
    try:
        retriever = _get_retriever()
        report["reviews_tfidf_rows"] = len(retriever.comments)
        logger.info("TF-IDF reseñas listo: %s comentarios indexados.", len(retriever.comments))
    except Exception:
        logger.exception("Warmup ReviewsRetriever falló.")
        report["reviews_tfidf_rows"] = "error"

    _bootstrap_products(report)
    _bootstrap_breb_faiss(report)

    _strict_check(report)
    logger.info("Bootstrap de fuentes completado: %s", report)
    return report


def is_bootstrap_healthy(report: Dict[str, Any]) -> bool:
    """Devuelve False si falta alguna fuente crítica según el reporte de arranque."""
    rev = str(report.get("reviews", ""))
    prod = str(report.get("products", ""))
    breb = str(report.get("breb_faiss", ""))
    tfidf = str(report.get("reviews_tfidf_rows", ""))

    if rev.startswith("missing") or rev in ("ingest_error", "ingested_rows=0"):
        return False
    if tfidf == "error":
        return False
    if prod.startswith("missing") or prod == "load_error":
        return False
    if "missing" in breb or breb == "build_error":
        return False
    return True
