"""

Aplicación FastAPI: expone el orquestador mediante HTTP.



Al arrancar, precarga SQLite (desde Excel si hace falta), catálogo JSON y FAISS BRE-B.

"""



from __future__ import annotations



import os

from contextlib import asynccontextmanager

from typing import Any, Dict



from fastapi import FastAPI, HTTPException



from models.request_models import AskRequest

from models.response_models import AskResponse, HealthResponse

from orchestrator.orchestrator_service import OrchestratorService

from utils.data_bootstrap import bootstrap_data_sources, is_bootstrap_healthy

from utils.logging import setup_logging

from utils.settings import build_chat_llm, get_settings



_settings = get_settings()

os.environ.setdefault("LOG_LEVEL", _settings.log_level)



logger = setup_logging(name="api.main", level=_settings.log_level)





@asynccontextmanager

async def lifespan(app: FastAPI):

    """Bootstrap de fuentes antes de aceptar tráfico."""

    try:

        app.state.data_sources = bootstrap_data_sources()

    except Exception:

        logger.exception("Fallo crítico en bootstrap de datos.")

        raise

    yield





app = FastAPI(

    title="Agent Orchestrator API",

    description="API de prueba técnica: orquestador con reviews SQLite, RAG BRE-B y catálogo JSON.",

    version="0.3.0",

    lifespan=lifespan,

)



_llm = build_chat_llm()

_orchestrator = OrchestratorService(_llm)





@app.get("/health", response_model=HealthResponse)

def health() -> HealthResponse:

    """Comprueba que el servicio está vivo y expone el estado de las fuentes de datos."""

    report = getattr(app.state, "data_sources", None)

    payload: Dict[str, Any] = {}

    if isinstance(report, dict):

        payload = dict(report)



    status = "ok"

    if payload and not is_bootstrap_healthy(payload):

        status = "degraded"



    return HealthResponse(

        status=status,

        version="0.3.0",

        data_sources=payload if payload else None,

    )





@app.post("/ask", response_model=AskResponse)

def ask(body: AskRequest) -> AskResponse:

    """

    Recibe una pregunta y devuelve la respuesta orquestada (router + tools + composición LLM).



    Raises:

        HTTPException: 500 si falla el procesamiento interno.

    """

    try:

        payload_dict: Dict[str, Any] = _orchestrator.process_question(body.question.strip())

        if body.session_id:

            payload_dict["session_id"] = body.session_id

            trace = list(payload_dict.get("trace") or [])

            trace.insert(0, {"step": "request", "session_id": body.session_id})

            payload_dict["trace"] = trace

        return AskResponse(**payload_dict)

    except Exception as exc:

        logger.exception("Error en /ask: %s", exc)

        raise HTTPException(

            status_code=500,

            detail="Error interno al procesar la pregunta.",

        ) from exc


