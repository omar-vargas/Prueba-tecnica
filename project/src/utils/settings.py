"""
Configuración centralizada desde variables de entorno.

Carga ``.env`` con python-dotenv y resuelve rutas relativas respecto a la raíz del proyecto
(carpeta que contiene ``src/`` y ``run.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Carga `.env` del proyecto con prioridad sobre variables heredadas del shell (evita claves/URLs de prueba).
load_dotenv(PROJECT_ROOT / ".env", override=True)
# Opcional: completar con un `.env` en el CWD sin pisar claves ya definidas por el archivo del proyecto.
load_dotenv(override=False)


def _parse_positive_int(raw: Optional[str], *, default: int) -> int:
    """Parsea un entero > 0 desde env; si falla, devuelve ``default``."""
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip())
        return n if n > 0 else default
    except ValueError:
        return default


def _strip_env(value: Optional[str]) -> Optional[str]:
    """Quita espacios y comillas sueltas al inicio/fin (p. ej. ``KEY= \"https://...`` sin cierre correcto)."""
    if value is None:
        return None
    s = value.strip()
    while s and s[0] in {'"', "'"}:
        s = s[1:].lstrip()
    while s and s[-1] in {'"', "'"}:
        s = s[:-1].rstrip()
    return s or None


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


@dataclass(frozen=True)
class AppSettings:
    """Valores de configuración de la aplicación."""

    log_level: str
    data_raw_dir: Path
    data_processed_dir: Path
    data_indexes_dir: Path
    sqlite_db_path: Path
    reviews_excel_path: Path
    product_catalog_path: Path
    breb_faiss_index_dir: Path
    breb_pdf_path: Path
    breb_rag_top_k: int
    strict_data_bootstrap: bool
    orchestrator_api_url: str
    azure_openai_api_key: Optional[str]
    azure_openai_endpoint: Optional[str]
    azure_openai_deployment: Optional[str]
    azure_openai_api_version: Optional[str]
    openai_api_key: Optional[str]
    openai_default_model: str
    llm_temperature: float


@lru_cache
def get_settings() -> AppSettings:
    """
    Lee variables de entorno una vez (cacheada).

    Returns:
        Instancia inmutable con rutas ya resueltas.
    """
    api_ver = _strip_env(os.getenv("AZURE_OPENAI_API_VERSION")) or _strip_env(
        os.getenv("OPENAI_API_VERSION")
    )
    temp_raw = _strip_env(os.getenv("LLM_TEMPERATURE", "0")) or "0"
    try:
        llm_temperature = float(temp_raw)
    except ValueError:
        llm_temperature = 0.0

    return AppSettings(
        log_level=_strip_env(os.getenv("LOG_LEVEL", "INFO")) or "INFO",
        data_raw_dir=_resolve_path(_strip_env(os.getenv("DATA_RAW_DIR", "./data/raw")) or "./data/raw"),
        data_processed_dir=_resolve_path(
            _strip_env(os.getenv("DATA_PROCESSED_DIR", "./data/processed")) or "./data/processed"
        ),
        data_indexes_dir=_resolve_path(
            _strip_env(os.getenv("DATA_INDEXES_DIR", "./data/indexes")) or "./data/indexes"
        ),
        sqlite_db_path=_resolve_path(
            _strip_env(os.getenv("SQLITE_DB_PATH", "./data/processed/bank_reviews.sqlite"))
            or "./data/processed/bank_reviews.sqlite"
        ),
        reviews_excel_path=_resolve_path(
            _strip_env(os.getenv("REVIEWS_EXCEL_PATH", "./data/raw/bank_reviews_colombia.xlsx"))
            or "./data/raw/bank_reviews_colombia.xlsx"
        ),
        product_catalog_path=_resolve_path(
            _strip_env(os.getenv("PRODUCT_CATALOG_PATH", "./data/processed/products_catalog.json"))
            or "./data/processed/products_catalog.json"
        ),
        breb_faiss_index_dir=_resolve_path(
            _strip_env(os.getenv("BREB_FAISS_INDEX_DIR", "./data/vectorstore/breb_index"))
            or "./data/vectorstore/breb_index"
        ),
        breb_pdf_path=_resolve_path(
            _strip_env(
                os.getenv(
                    "BREB_PDF_PATH",
                    "./data/docs/documento-tecnico-bre-b-febrero-2026.pdf",
                )
            )
            or "./data/docs/documento-tecnico-bre-b-febrero-2026.pdf"
        ),
        breb_rag_top_k=_parse_positive_int(os.getenv("BREB_RAG_TOP_K"), default=6),
        strict_data_bootstrap=(
            (_strip_env(os.getenv("STRICT_DATA_BOOTSTRAP")) or "").lower()
            in ("1", "true", "yes", "on")
        ),
        orchestrator_api_url=_strip_env(os.getenv("ORCHESTRATOR_API_URL", "http://127.0.0.1:8000"))
        or "http://127.0.0.1:8000",
        azure_openai_api_key=_strip_env(os.getenv("AZURE_OPENAI_API_KEY")),
        azure_openai_endpoint=_strip_env(os.getenv("AZURE_OPENAI_ENDPOINT")),
        azure_openai_deployment=_strip_env(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),
        azure_openai_api_version=api_ver,
        openai_api_key=_strip_env(os.getenv("OPENAI_API_KEY")),
        openai_default_model=_strip_env(os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"))
        or "gpt-4o-mini",
        llm_temperature=llm_temperature,
    )


def build_chat_llm() -> BaseChatModel:
    """
    Construye el modelo de chat: Azure OpenAI si las variables están completas; si no, OpenAI público.

    Returns:
        Instancia lista para router y compositor.

    Raises:
        ValueError: Si no hay credenciales suficientes para ningún proveedor.
    """
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    s = get_settings()

    if s.azure_openai_api_key and s.azure_openai_endpoint and s.azure_openai_deployment:
        api_version = s.azure_openai_api_version or "2024-02-15-preview"
        endpoint = s.azure_openai_endpoint.rstrip("/")
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=s.azure_openai_deployment,
            api_version=api_version,
            api_key=s.azure_openai_api_key,
            temperature=s.llm_temperature,
        )

    if s.openai_api_key:
        return ChatOpenAI(
            model=s.openai_default_model,
            api_key=s.openai_api_key,
            temperature=s.llm_temperature,
        )

    raise ValueError(
        "Configura Azure OpenAI (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_DEPLOYMENT_NAME) o bien OPENAI_API_KEY para OpenAI directo."
    )
