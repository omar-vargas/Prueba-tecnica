"""
Servicio de orquestación: router numérico, tools selectivas y composición multi-fuente.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from orchestrator.response_composer import compose_final_answer
from orchestrator.router import classify_route
from tools.breb_rag_tool import query_breb_document
from tools.products_tool import query_products
from tools.reviews_tool import query_reviews

logger = logging.getLogger(__name__)

# Fuente interna del router → etiqueta para contexto y sources_used (API)
_SOURCE_TO_LABEL: Dict[str, str] = {
    "reviews": "bank_reviews_colombia",
    "breb_doc": "documento_tecnico_bre_b",
    "products": "portafolio_productos",
}

_ROUTE_TO_SOURCE: Dict[int, str] = {
    0: "reviews",
    1: "breb_doc",
    2: "products",
}

ToolFn = Callable[..., Dict[str, Any]]
_SOURCE_TOOL: Dict[str, Tuple[str, ToolFn]] = {
    "reviews": ("query_reviews", query_reviews),
    "breb_doc": ("query_breb_document", query_breb_document),
    "products": ("query_products", query_products),
}


def _append_context(
    segments: List[Dict[str, str]],
    router_key: str,
    content: str,
) -> None:
    label = _SOURCE_TO_LABEL.get(router_key, router_key)
    if str(content).strip():
        segments.append({"source": label, "content": str(content).strip()})


def _tool_meta(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if k != "content"}


class OrchestratorService:
    """Orquesta classify_route → tools → compose_final_answer."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def _run_tool(
        self,
        question: str,
        router_src: str,
        context_segments: List[Dict[str, str]],
        sources_used: List[str],
        trace: List[Dict[str, Any]],
        tools_run: List[str],
        *,
        isolate_errors: bool,
    ) -> None:
        """Ejecuta una tool por clave de router y actualiza segmentos, traza y fuentes usadas."""
        spec = _SOURCE_TOOL.get(router_src)
        if not spec:
            return
        tool_name, fn = spec
        try:
            result = fn(question, llm=self._llm)
            _append_context(context_segments, router_src, str(result.get("content", "")))
            sources_used.append(_SOURCE_TO_LABEL[router_src])
            tools_run.append(tool_name)
            trace.append(
                {"step": "tool", "name": tool_name, "meta": _tool_meta(result)}
            )
        except Exception:
            logger.exception("Fallo en tool %s (router_src=%s)", tool_name, router_src)
            trace.append({"step": "tool_error", "name": tool_name})
            if not isolate_errors:
                raise

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline: ruta numérica, herramientas según fuentes, respuesta unificada.

        Returns:
            Dict con question, route, sources_used, answer, trace.
        """
        trace: List[Dict[str, Any]] = []
        sources_used: List[str] = []
        context_segments: List[Dict[str, str]] = []
        tools_run: List[str] = []

        routing = classify_route(question, self._llm)
        route = int(routing["route"])
        router_sources: List[str] = list(routing.get("sources") or [])

        trace.append(
            {
                "step": "classify_route",
                "route": route,
                "router_sources": router_sources,
            }
        )
        logger.info("Ruta %s, fuentes router: %s", route, router_sources)

        try:
            if route in _ROUTE_TO_SOURCE:
                primary = _ROUTE_TO_SOURCE[route]
                self._run_tool(
                    question,
                    primary,
                    context_segments,
                    sources_used,
                    trace,
                    tools_run,
                    isolate_errors=False,
                )
            elif route == 3:
                for src in router_sources:
                    self._run_tool(
                        question,
                        src,
                        context_segments,
                        sources_used,
                        trace,
                        tools_run,
                        isolate_errors=True,
                    )
                trace.append({"step": "multi_source_tools", "executed": list(tools_run)})

        except Exception:
            logger.exception("Error ejecutando tools para route=%s", route)
            trace.append({"step": "orchestrator_tools_error", "route": route})

        if not context_segments and route in (0, 1, 2, 3):
            context_segments.append(
                {
                    "source": "vacío",
                    "content": "No se recuperó contexto de las herramientas.",
                }
            )

        answer = compose_final_answer(
            question=question,
            route=route,
            context_segments=context_segments,
            llm=self._llm,
        )
        trace.append(
            {
                "step": "compose_final_answer",
                "answer_length": len(answer),
                "context_sources": [s.get("source") for s in context_segments],
            }
        )

        return {
            "question": question,
            "route": route,
            "sources_used": sources_used,
            "answer": answer,
            "trace": trace,
        }
