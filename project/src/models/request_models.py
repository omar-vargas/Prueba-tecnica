"""
Modelos Pydantic para peticiones HTTP del orquestador.

Define el contrato del endpoint POST /ask y metadatos opcionales del cliente.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Cuerpo de la petición para hacer una pregunta al agente orquestador."""

    question: str = Field(..., min_length=1, description="Pregunta del usuario en lenguaje natural.")
    session_id: Optional[str] = Field(
        default=None,
        description="Identificador de sesión para trazabilidad (se devuelve en la respuesta si se envía).",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadatos adicionales del cliente (canal, idioma, etc.); reservado para extensiones.",
    )


class HealthQuery(BaseModel):
    """Parámetros opcionales para comprobaciones de salud extendidas (reservado)."""

    detail: bool = Field(default=False, description="Si true, incluir detalles de dependencias.")
