"""
Herramienta de catálogo de productos bancarios (JSON generado desde PDF vía LLM).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from langchain_core.language_models.chat_models import BaseChatModel

from storage.product_catalog import load_product_catalog_cached
from utils.settings import get_settings

logger = logging.getLogger(__name__)

_STOPWORDS: Set[str] = {
    "que",
    "qué",
    "cual",
    "cuáles",
    "cuales",
    "como",
    "cómo",
    "donde",
    "dónde",
    "tienen",
    "tiene",
    "tengo",
    "hay",
    "son",
    "los",
    "las",
    "una",
    "unos",
    "unas",
    "del",
    "por",
    "para",
    "con",
    "sus",
    "nuestro",
    "nuestros",
    "este",
    "esta",
    "estos",
    "estas",
    "muy",
    "tan",
    "the",
    "and",
}


def _question_tokens(question: str) -> Set[str]:
    """
    Tokeniza la pregunta en español quitando signos tipográficos que rompen el match
    (p. ej. ``tienen?`` → ``tienen`` no aplicaría sin esto).
    """
    q = question.lower()
    q = re.sub(r"[¿?¡!.,;:\"'()\[\]{}]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    out: Set[str] = set()
    for t in q.split():
        t = t.strip()
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        out.add(t)
    return out


def _product_search_blob(product: Dict[str, Any]) -> str:
    """Texto agregado del producto para coincidencias."""
    parts: List[str] = [
        str(product.get("name", "")),
        str(product.get("category", "")),
        str(product.get("description", "")),
        str(product.get("target_customer", "")),
        str(product.get("raw_text", "")),
        str(product.get("interest_rate", "")),
    ]
    benefits = product.get("benefits")
    if isinstance(benefits, list):
        parts.extend(str(b) for b in benefits)
    return " ".join(parts).lower()


def _score_product_relevance(question: str, product: Dict[str, Any]) -> int:
    """Heurística: tokens en texto del producto + refuerzos por intención (ahorro, crédito, etc.)."""
    tokens = _question_tokens(question)
    blob = _product_search_blob(product)
    score = 0
    for tok in tokens:
        if tok in blob:
            score += 2

    qlow = question.lower()
    cat = str(product.get("category", "")).lower()
    name = str(product.get("name", "")).lower()

    if any(k in qlow for k in ("ahorro", "ahorrar", "ahorros")):
        if "ahorro" in cat or "ahorro" in name or "ahorros" in blob:
            score += 12

    if "crédito" in qlow or "credito" in qlow:
        if any(x in cat for x in ("credito", "crédito", "tarjeta")) or "crédito" in blob or "credito" in blob:
            score += 8

    if "tarjeta" in qlow:
        if "tarjeta" in cat or "tarjeta" in name:
            score += 8

    if "cdt" in qlow or "depósito" in qlow or "deposito" in qlow:
        if "cdt" in blob or "depósito" in blob or "deposito" in blob:
            score += 10

    if "cuenta" in qlow:
        if "cuenta" in cat or "cuenta" in name:
            score += 4

    if "productos" in qlow or "producto" in qlow:
        score += 1

    return score


def _filter_by_intent(question: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Si la pregunta es clara (p. ej. solo ahorro), prioriza productos alineados."""
    qlow = question.lower()
    if any(k in qlow for k in ("ahorro", "ahorrar", "ahorros")):
        matched = [
            p
            for p in products
            if "ahorro" in str(p.get("category", "")).lower()
            or "ahorro" in str(p.get("name", "")).lower()
            or "ahorros" in _product_search_blob(p)[:300]
        ]
        if matched:
            return matched
    return []


def query_products(
    question: str,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Consulta el portafolio JSON y devuelve los productos más alineados con la pregunta.

    Args:
        question: Pregunta del usuario.
        llm: Modelo opcional (reservado para enriquecimiento futuro).

    Returns:
        Dict con ``content``, ``raw_products`` (lista de dicts), ``matches``, etc.
    """
    _ = llm
    settings = get_settings()
    catalog_path = settings.product_catalog_path
    products = load_product_catalog_cached(str(catalog_path))

    if not products:
        body = (
            f"No hay catálogo cargado o el archivo está vacío: {catalog_path}. "
            "Ejecuta la pipeline de extracción desde el PDF (``build_products_catalog``)."
        )
        return {
            "content": body,
            "raw_products": [],
            "source": "products_json_catalog",
            "catalog_path": str(catalog_path),
            "matches": 0,
        }

    scored: List[tuple[int, Dict[str, Any]]] = []
    for p in products:
        s = _score_product_relevance(question, p)
        scored.append((s, p))

    scored.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))
    intent_hits = _filter_by_intent(question, products)

    if intent_hits:
        sub = [(_score_product_relevance(question, p), p) for p in intent_hits]
        sub.sort(key=lambda x: (-x[0], str(x[1].get("name", ""))))
        sub_pos = [(s, p) for s, p in sub if s > 0]
        top = [p for s, p in sub_pos[:8]] if sub_pos else intent_hits[:8]
    else:
        positive = [(s, p) for s, p in scored if s > 0]
        if positive:
            top = [p for s, p in positive[:8]]
        else:
            top = [p for _, p in scored[:5]]

    lines: List[str] = []
    for p in top:
        lines.append(
            f"- **{p.get('name', '')}** (categoría: {p.get('category', '')}): "
            f"{(p.get('description') or '')[:500]}"
        )
    content = "\n".join(lines) if lines else "No se encontraron productos en el catálogo."

    logger.debug(
        "query_products: %s productos en catálogo, %s devueltos, tokens=%s",
        len(products),
        len(top),
        _question_tokens(question),
    )

    return {
        "content": content,
        "raw_products": top,
        "source": "products_json_catalog",
        "catalog_path": str(catalog_path),
        "matches": len(top),
    }
