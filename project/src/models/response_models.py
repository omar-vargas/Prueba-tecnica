"""
Modelos Pydantic para respuestas HTTP del orquestador.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskResponse(BaseModel):
    """Respuesta del endpoint POST /ask con payload del orquestador."""

    question: str = Field(..., description="Pregunta procesada.")
    route: int = Field(..., ge=0, le=3, description="Ruta numérica del router (0–3).")
    sources_used: List[str] = Field(
        default_factory=list,
        description="Fuentes de datos usadas (p. ej. bank_reviews_colombia, documento_tecnico_bre_b).",
    )
    answer: str = Field(..., description="Respuesta final integrando el contexto recuperado.")
    trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pasos: classify_route, tools, compose_final_answer.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Eco opcional del identificador de sesión enviado por el cliente.",
    )


class HealthResponse(BaseModel):
    """Respuesta del health check."""

    status: str = Field(..., description="ok | degraded | error")
    version: str = Field(default="0.1.0", description="Versión de la API.")
    data_sources: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resumen del bootstrap (SQLite, JSON productos, FAISS BRE-B).",
    )
