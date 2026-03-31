"""
Router numérico: decide la ruta (0–3) y las fuentes a consultar (especialmente en multi-source).

Salida del LLM en JSON estricto; fallback por reglas si falla el parseo.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

RouteCode = Literal[0, 1, 2, 3]
RouterSource = Literal["reviews", "breb_doc", "products"]

ALLOWED_ROUTER_SOURCES: frozenset[str] = frozenset({"reviews", "breb_doc", "products"})


class RouteDecision(TypedDict):
    """Resultado de ``classify_route``."""

    route: RouteCode
    sources: List[str]


_ROUTER_SYSTEM = """
Eres un router para un orquestador bancario. Tu salida debe ser ÚNICAMENTE un JSON válido,
sin markdown, sin cercas ```, sin explicaciones ni texto antes o después.

Estructura obligatoria:
{{"route": <entero 0-3>, "sources": [<lista de strings>]}}

Códigos de ruta:
- route 0: la pregunta es sobre reseñas, comentarios, satisfacción, sedes, problemas recurrentes en sucursales.
  Debes usar sources: ["reviews"].
- route 1: la pregunta es sobre BRE-B, interoperabilidad, transferencias, QR, DICE, MOL, sistema de pagos.
  Debes usar sources: ["breb_doc"].
- route 2: la pregunta es sobre productos bancarios, cuentas, tarjetas, créditos, CDT, portafolio.
  Debes usar sources: ["products"].
- route 3: la pregunta requiere combinar dos o más fuentes. Enumera en "sources" solo las necesarias,
  eligiendo entre exactamente estos strings: "reviews", "breb_doc", "products".

Reglas:
- Para route 0, 1 o 2, "sources" debe tener un solo elemento acorde al código.
- Para route 3, "sources" debe tener al menos dos elementos (sin duplicados).
- Los strings en sources deben ser exactamente: reviews, breb_doc o products.
""".strip()


def build_router_prompt() -> ChatPromptTemplate:
    """Plantilla de mensajes para el clasificador de ruta (JSON)."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _ROUTER_SYSTEM),
            ("human", "Pregunta del usuario:\n{question}"),
        ]
    )


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def classify_route_fallback(question: str) -> RouteDecision:
    """
    Clasificador de respaldo por palabras clave (sin LLM).

    Returns:
        Diccionario con ``route`` y ``sources``.
    """
    text = question.lower()

    mentions_reviews = bool(
        re.search(
            r"\b(reseña|reseñas|review|reviews|sede|sedes|comentario|comentarios|satisfacción|satisfaccion|calificación|calificacion)\b",
            text,
        )
    )
    mentions_breb = bool(
        re.search(
            r"\b(bre-?b|transferencia|interoperabilidad|mol|dice|qr|sistema de pagos)\b",
            text,
        )
    )
    mentions_products = bool(
        re.search(
            r"\b(producto|productos|portafolio|catálogo|catalogo|cuenta|tarjeta|crédito|credito|prestamo|préstamo|cdt)\b",
            text,
        )
    )

    count = sum([mentions_reviews, mentions_breb, mentions_products])
    if count >= 2:
        sources: List[str] = []
        if mentions_reviews:
            sources.append("reviews")
        if mentions_breb:
            sources.append("breb_doc")
        if mentions_products:
            sources.append("products")
        return {"route": 3, "sources": sources}

    if mentions_reviews:
        return {"route": 0, "sources": ["reviews"]}
    if mentions_breb:
        return {"route": 1, "sources": ["breb_doc"]}
    if mentions_products:
        return {"route": 2, "sources": ["products"]}

    logger.info("Fallback router: sin señal clara → route=0 reviews.")
    return {"route": 0, "sources": ["reviews"]}


def _normalize_route_decision(raw_route: Any, raw_sources: Any, question: str) -> RouteDecision:
    """Valida y corrige la salida del LLM."""
    try:
        r = int(raw_route)
    except (TypeError, ValueError):
        logger.warning("route inválido del LLM: %s", raw_route)
        return classify_route_fallback(question)

    if r not in (0, 1, 2, 3):
        logger.warning("route fuera de rango: %s", r)
        return classify_route_fallback(question)

    src_list: List[str] = []
    if isinstance(raw_sources, list):
        for item in raw_sources:
            s = str(item).strip()
            if s in ALLOWED_ROUTER_SOURCES:
                src_list.append(s)
    src_list = list(dict.fromkeys(src_list))

    if r == 0:
        return {"route": 0, "sources": ["reviews"]}
    if r == 1:
        return {"route": 1, "sources": ["breb_doc"]}
    if r == 2:
        return {"route": 2, "sources": ["products"]}

    # r == 3
    if len(src_list) < 2:
        logger.warning(
            'route=3 con sources insuficientes %s; se usan las tres fuentes.',
            src_list,
        )
        src_list = ["reviews", "breb_doc", "products"]
    return {"route": 3, "sources": src_list}


def classify_route(question: str, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Clasifica la pregunta en una ruta numérica y lista de fuentes lógicas del router.

    Args:
        question: Texto del usuario.
        llm: Modelo de chat (Azure/OpenAI vía LangChain).

    Returns:
        ``{{"route": 0|1|2|3, "sources": ["reviews"|"breb_doc"|"products", ...]}}``
    """
    prompt = build_router_prompt()
    chain = prompt | llm

    try:
        response = chain.invoke({"question": question})
        raw = response.content if isinstance(response.content, str) else str(response.content)
        content = _strip_code_fence(raw)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("La raíz del JSON no es un objeto")

        decision = _normalize_route_decision(
            parsed.get("route"),
            parsed.get("sources"),
            question,
        )
        logger.info("classify_route (LLM): %s", decision)
        return dict(decision)

    except Exception as exc:
        logger.exception("Error en classify_route con LLM: %s", exc)
        fb = classify_route_fallback(question)
        logger.info("classify_route (fallback): %s", fb)
        return dict(fb)
