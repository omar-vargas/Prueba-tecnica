"""
Extracción estructurada del portafolio de productos desde PDF usando LLM (LangChain).

Genera un catálogo JSON listo para ``product_catalog`` y ``products_tool``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from storage.product_catalog import save_product_catalog

logger = logging.getLogger(__name__)

_PRODUCT_KEYS: tuple[str, ...] = (
    "name",
    "category",
    "description",
    "interest_rate",
    "minimum_amount",
    "maximum_amount",
    "term",
    "management_fee",
    "benefits",
    "requirements",
    "target_customer",
    "raw_text",
)

_ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {
        "ahorro",
        "cuenta_corriente",
        "tarjeta",
        "credito",
        "inversion",
        "seguro",
        "otro",
    }
)

_MAX_PAGE_CHARS = 14_000


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def load_products_pdf_pages(pdf_path: str) -> List[str]:
    """
    Extrae el texto de cada página del PDF de portafolio.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Lista con el texto limpio de cada página (puede incluir páginas vacías).

    Raises:
        FileNotFoundError: Si el PDF no existe.
    """
    path = Path(pdf_path)
    if not path.is_file():
        logger.error("PDF no encontrado: %s", pdf_path)
        raise FileNotFoundError(f"No existe el PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(str(path.resolve()))
        documents = loader.load()
    except Exception:
        logger.exception("Error al cargar PDF: %s", pdf_path)
        raise

    pages: List[str] = []
    for doc in documents:
        raw = doc.page_content or ""
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        pages.append(cleaned)

    logger.info(
        "PDF productos: %s páginas extraídas desde %s",
        len(pages),
        pdf_path,
    )
    return pages


def validate_product_record(product: Dict[str, Any]) -> bool:
    """
    Verifica estructura y tipos mínimos de un registro de producto.

    Args:
        product: Dict normalizado.

    Returns:
        True si cumple el esquema esperado.
    """
    if not isinstance(product, dict):
        return False
    for key in _PRODUCT_KEYS:
        if key not in product:
            return False
    if not isinstance(product.get("benefits"), list):
        return False
    if not isinstance(product.get("requirements"), list):
        return False
    if not isinstance(product.get("name"), str):
        return False
    if not isinstance(product.get("category"), str):
        return False
    if not isinstance(product.get("raw_text"), str):
        return False
    return True


def _normalize_category(value: str) -> str:
    c = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if c == "crédito" or c == "credito":
        return "credito"
    if c == "inversión" or c == "inversion":
        return "inversion"
    if c in _ALLOWED_CATEGORIES:
        return c
    return "otro"


def _normalize_product_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Completa claves faltantes y normaliza tipos."""
    out: Dict[str, Any] = {}
    for key in _PRODUCT_KEYS:
        if key in ("benefits", "requirements"):
            v = raw.get(key, [])
            if v is None:
                out[key] = []
            elif isinstance(v, list):
                out[key] = [str(x).strip() for x in v if str(x).strip()]
            else:
                out[key] = [str(v).strip()] if str(v).strip() else []
        else:
            val = raw.get(key, "")
            out[key] = "" if val is None else str(val).strip()

    out["category"] = _normalize_category(out["category"])
    return out


def extract_product_with_llm(page_text: str, llm: BaseChatModel) -> Optional[Dict[str, Any]]:
    """
    Pide al LLM extraer un único producto desde el texto de una página.

    Args:
        page_text: Texto de la página (se trunca si es muy largo).
        llm: Modelo de chat (OpenAI o Azure vía LangChain).

    Returns:
        Dict con el esquema acordado, o ``None`` si la página no describe un producto
        o el modelo devuelve ``null`` / JSON inválido.
    """
    if not page_text.strip():
        return None

    snippet = page_text[:_MAX_PAGE_CHARS]
    if len(page_text) > _MAX_PAGE_CHARS:
        snippet += "\n\n[... texto truncado para el modelo ...]"

    categories_list = ", ".join(sorted(_ALLOWED_CATEGORIES))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
Eres un extractor estricto de productos bancarios. Debes responder ÚNICAMENTE con:
- un objeto JSON válido que cumpla el esquema indicado, o
- la palabra null (sin comillas) si la página NO describe claramente un producto financiero.

Prohibido: markdown, bloques ```, explicaciones, texto antes o después del JSON.

category debe ser EXACTAMENTE una de: {categories_list}

Esquema obligatorio (todas las claves presentes; copia esta forma con comillas dobles):
{{{{ 
  "name": "",
  "category": "",
  "description": "",
  "interest_rate": "",
  "minimum_amount": "",
  "maximum_amount": "",
  "term": "",
  "management_fee": "",
  "benefits": [],
  "requirements": [],
  "target_customer": "",
  "raw_text": ""
}}}}

Si un dato no aparece, usa "" o []. raw_text debe resumir o copiar el contenido relevante de la página.
                """.strip(),
            ),
            (
                "human",
                "Texto de la página del portafolio:\n\n{page_text}",
            ),
        ]
    )

    try:
        chain = prompt | llm
        response = chain.invoke({"page_text": snippet})
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = _strip_code_fence(content)
    except Exception:
        logger.exception("Fallo la invocación al LLM para extracción de producto.")
        return None

    lowered = content.strip().lower()
    if lowered in {"null", "none"}:
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("JSON inválido del LLM (primeros 200 chars): %r", content[:200])
        return None

    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        return None

    normalized = _normalize_product_record(parsed)
    if not validate_product_record(normalized):
        logger.warning("Registro normalizado no pasa validate_product_record.")
        return None

    if not normalized["name"].strip():
        return None
    if not normalized["raw_text"].strip():
        return None

    return normalized


def build_products_catalog(
    llm: BaseChatModel,
    pdf_path: str = "data/docs/portafolio_productos_bancarios_v2-1 (1).pdf",
    output_path: str = "data/processed/products_catalog.json",
) -> List[Dict[str, Any]]:
    """
    Pipeline: páginas PDF → extracción LLM por página → validación → JSON en disco.

    Args:
        llm: Instancia de chat (``ChatOpenAI``, ``AzureChatOpenAI``, etc.).
        pdf_path: Ruta al PDF (relativa a la raíz del proyecto si no es absoluta).
        output_path: Ruta del JSON de salida (relativa a la raíz del proyecto si no es absoluta).

    Returns:
        Lista de productos incluidos en el catálogo.
    """
    root = _project_root()
    pdf = Path(pdf_path)
    if not pdf.is_absolute():
        pdf = root / pdf_path
    out = Path(output_path)
    if not out.is_absolute():
        out = root / output_path

    pages = load_products_pdf_pages(str(pdf))
    products: List[Dict[str, Any]] = []

    for i, page_text in enumerate(pages):
        logger.info("Procesando página %s/%s", i + 1, len(pages))
        if not page_text.strip():
            logger.debug("Página %s vacía, se omite.", i + 1)
            continue

        try:
            record = extract_product_with_llm(page_text, llm)
        except Exception:
            logger.exception("Error inesperado en página %s; se continúa.", i + 1)
            continue

        if record is None:
            logger.debug("Página %s: sin producto detectado.", i + 1)
            continue

        if not str(record.get("name", "")).strip():
            logger.debug("Página %s: producto sin nombre, se omite.", i + 1)
            continue
        if not str(record.get("category", "")).strip():
            record["category"] = "otro"
        if not str(record.get("raw_text", "")).strip():
            logger.debug("Página %s: sin raw_text, se omite.", i + 1)
            continue

        products.append(record)
        logger.info("Página %s: producto extraído → %r", i + 1, record.get("name", "")[:80])

    try:
        save_product_catalog(products, str(out))
    except Exception:
        logger.exception("No se pudo guardar el catálogo en %s", out)
        raise

    return products


def _print_extraction_summary(products: List[Dict[str, Any]]) -> None:
    """Imprime cantidad total y los primeros 5 nombres (utilidad de demo)."""
    print(f"\nProductos extraídos: {len(products)}")
    for j, p in enumerate(products[:5], start=1):
        print(f"  {j}. {p.get('name', '(sin nombre)')}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    src = _project_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from utils.settings import build_chat_llm

    _llm = build_chat_llm()
    _products = build_products_catalog(_llm)
    _print_extraction_summary(_products)
