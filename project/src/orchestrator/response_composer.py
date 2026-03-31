"""
Compone la respuesta final con LangChain integrando una o varias fuentes de contexto.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def compose_final_answer(
    question: str,
    route: int,
    context_segments: List[Dict[str, str]],
    llm: BaseChatModel,
) -> str:
    """
    Genera la respuesta en español usando solo los contextos entregados (una o varias fuentes).

    Args:
        question: Pregunta original.
        route: Código de ruta (0–3); 3 indica multi-fuente explícita.
        context_segments: Lista de ``{{"source": "<id>", "content": "<texto>"}}``.
        llm: Modelo de chat.

    Returns:
        Texto final para el usuario.
    """
    blocks: List[str] = []
    for seg in context_segments:
        src = str(seg.get("source", "desconocida")).strip()
        body = str(seg.get("content", "")).strip()
        if body:
            blocks.append(f"### Fuente: {src}\n{body}")

    if not blocks:
        context_block = "(No hay contexto recuperado de las fuentes de datos.)"
    else:
        context_block = "\n\n---\n\n".join(blocks)

    route_note = (
        "Varias fuentes deben integrarse; indica qué parte de la respuesta apoya cada fuente cuando sea útil."
        if route == 3
        else "Usa la fuente indicada en el contexto."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres el asistente de un orquestador bancario. Responde en español.

Reglas estrictas:
- Basa tu respuesta ÚNICAMENTE en el bloque "Contexto recuperado" (una o varias fuentes etiquetadas).
- Integra varias fuentes si aparecen; no inventes relaciones causales o datos que no estén soportados por el texto.
- Si el contexto no permite conectar ideas o falta información, dilo con claridad
  (por ejemplo: "Con la información disponible no puedo relacionar X con Y.").
- No inventes cifras, nombres de productos ni hechos que no figuren en el contexto.
- Sé conciso y profesional.
- {route_note}
                """.strip(),
            ),
            (
                "human",
                """
Ruta numérica detectada: {route}

Pregunta del usuario:
{question}

Contexto recuperado:
{context}
                """.strip(),
            ),
        ]
    )

    chain = prompt | llm
    try:
        out = chain.invoke(
            {
                "route_note": route_note,
                "route": route,
                "question": question,
                "context": context_block,
            }
        )
        text = out.content if isinstance(out.content, str) else str(out.content)
        return text.strip()
    except Exception as exc:
        logger.exception("Error al componer respuesta final: %s", exc)
        return (
            "No pude generar una respuesta en este momento. "
            "Verifica la configuración del modelo o inténtalo de nuevo."
        )
