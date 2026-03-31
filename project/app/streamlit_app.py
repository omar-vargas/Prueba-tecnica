"""
Interfaz Streamlit tipo chatbot para el orquestador FastAPI.

Ejecutar desde la raíz del proyecto ``project/``::

    streamlit run app/streamlit_app.py

Por defecto consume ``http://127.0.0.1:8000/ask`` (equivalente a ``localhost:8000``;
configurable con ``ORCHESTRATOR_API_URL`` o el campo en la barra lateral).
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

DEFAULT_API_BASE = os.getenv("ORCHESTRATOR_API_URL", "http://127.0.0.1:8000")

SUGGESTED_QUESTIONS: List[str] = [
    "¿Qué comentarios hay sobre la sede Chapinero?",
    "¿Qué es BRE-B y cómo funciona?",
    "¿Qué productos de ahorro tienen?",
    "¿Qué problemas reportan los usuarios y qué productos podrían ayudar?",
]

PLACEHOLDER_INPUT = (
    "Ej: ¿Qué comentarios hay sobre la sede Chapinero? · ¿Qué es BRE-B? · "
    "¿Qué productos de ahorro tienen?"
)


def inject_custom_css() -> None:
    """Inyecta estilos corporativos (paleta #2596be, chat, sidebar)."""
    primary = "#2596be"
    primary_dark = "#1e7a9a"
    soft_bg = "#f2f8fb"
    border_soft = "#d4e8f0"
    text_muted = "#5a7582"
    text_dark = "#1e3a47"
    st.markdown(
        f"""
        <style>
        /* Fondo general y columna principal centrada */
        .stApp {{
            background: linear-gradient(180deg, {soft_bg} 0%, #ffffff 45%, #fafcfd 100%);
        }}
        .main .block-container {{
            max-width: 52rem;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, {soft_bg} 100%);
            border-right: 1px solid {border_soft};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
        }}
        /* Header principal */
        .bank-main-header {{
            padding: 0.25rem 0 1.25rem 0;
            border-bottom: 1px solid {border_soft};
            margin-bottom: 1.25rem;
        }}
        .bank-main-header h1 {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {text_dark};
            letter-spacing: -0.02em;
            margin: 0 0 0.35rem 0;
        }}
        .bank-main-header h1 span.accent {{
            color: {primary};
        }}
        .bank-main-header .bank-subtitle {{
            font-size: 0.95rem;
            color: {text_muted};
            line-height: 1.45;
            margin: 0;
            max-width: 42rem;
        }}
        /* Chat nativo Streamlit (st.chat_message): sin HTML manual en el cuerpo */
        [data-testid="stChatMessage"] {{
            border-radius: 12px !important;
            border: 1px solid {border_soft} !important;
            margin-bottom: 0.65rem !important;
            background-color: #ffffff !important;
            box-shadow: 0 1px 3px rgba(37, 150, 190, 0.06) !important;
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar-user"]) {{
            border-color: {primary}55 !important;
            background: linear-gradient(135deg, #ffffff 0%, {soft_bg} 100%) !important;
        }}
        [data-testid="stChatMessageContent"] {{
            font-size: 0.95rem !important;
            line-height: 1.55 !important;
            color: {text_dark} !important;
        }}
        /* Tarjeta sugerencias */
        .suggest-card {{
            background: #ffffff;
            border: 1px solid {border_soft};
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 2px 8px rgba(37, 150, 190, 0.06);
        }}
        .suggest-card h3 {{
            margin: 0 0 0.65rem 0;
            font-size: 0.85rem;
            font-weight: 600;
            color: {text_dark};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        /* Sidebar títulos */
        .sb-brand {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {text_dark};
            margin-bottom: 0.35rem;
        }}
        .sb-brand span {{
            color: {primary};
        }}
        .sb-desc {{
            font-size: 0.82rem;
            color: {text_muted};
            line-height: 1.45;
            margin-bottom: 1.25rem;
        }}
        .sb-section-title {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {text_muted};
            font-weight: 600;
            margin: 1.1rem 0 0.5rem 0;
            padding-top: 0.75rem;
            border-top: 1px solid {border_soft};
        }}
        .sb-section-title:first-of-type {{
            border-top: none;
            padding-top: 0;
            margin-top: 0;
        }}
        /* Badges route / fuentes */
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            margin: 0.2rem 0.25rem 0 0;
        }}
        .badge-route {{
            background: {primary};
            color: #fff;
        }}
        .badge-source {{
            background: {soft_bg};
            color: {primary_dark};
            border: 1px solid {border_soft};
        }}
        /* Botones primarios Streamlit */
        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="baseButton-primary"] {{
            background-color: {primary} !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.45rem 1.1rem !important;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: {primary_dark} !important;
        }}
        div.stButton > button {{
            border-radius: 10px !important;
        }}
        /* Preguntas sugeridas: secundario discreto */
        div[data-testid="column"] div.stButton > button[kind="secondary"] {{
            border: 1px solid {border_soft} !important;
            color: {primary_dark} !important;
            background: #ffffff !important;
            font-size: 0.82rem !important;
            text-align: left !important;
            line-height: 1.35 !important;
            min-height: 3.2rem !important;
        }}
        div[data-testid="column"] div.stButton > button[kind="secondary"]:hover {{
            border-color: {primary} !important;
            background: {soft_bg} !important;
        }}
        /* Input redondeado */
        .stTextInput input, .stTextArea textarea {{
            border-radius: 10px !important;
            border-color: {border_soft} !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {primary} !important;
            box-shadow: 0 0 0 1px {primary}33 !important;
        }}
        /* Alertas error */
        div[data-testid="stAlert"] {{
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Inicializa claves de ``st.session_state`` si no existen."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_route" not in st.session_state:
        st.session_state.last_route = None
    if "last_sources_used" not in st.session_state:
        st.session_state.last_sources_used = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE.strip().rstrip("/")
    if "api_base_url_input" not in st.session_state:
        st.session_state.api_base_url_input = st.session_state.api_base_url


def call_ask_api(base_url: str, question: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Envía POST a ``/ask`` del orquestador.

    Args:
        base_url: URL base sin barra final (ej. ``http://127.0.0.1:8000``).
        question: Texto de la pregunta.
        timeout: Segundos de espera.

    Returns:
        JSON parseado del backend.

    Raises:
        requests.RequestException: Si la red o el HTTP fallan.
        ValueError: Si la respuesta no es JSON válido.
    """
    url = f"{base_url.rstrip('/')}/ask"
    response = requests.post(
        url,
        json={"question": question.strip()},
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ValueError("La API no devolvió JSON válido.") from exc


def update_trace_from_response(data: Dict[str, Any]) -> None:
    """Actualiza metadatos de trazabilidad en sesión a partir de la respuesta."""
    st.session_state.last_route = data.get("route")
    st.session_state.last_sources_used = list(data.get("sources_used") or [])
    st.session_state.last_trace = list(data.get("trace") or [])


def clear_trace_metadata() -> None:
    """Limpia route, fuentes y trace en sidebar."""
    st.session_state.last_route = None
    st.session_state.last_sources_used = []
    st.session_state.last_trace = []


def submit_question(question: str) -> None:
    """
    Añade mensaje de usuario, llama a la API y añade respuesta del asistente.

    En error de red o HTTP, muestra un mensaje de asistente con el fallo.
    Finaliza con ``st.rerun()`` para refrescar la vista (excepto si la pregunta está vacía).
    """
    q = question.strip()
    if not q:
        return

    st.session_state.messages.append({"role": "user", "content": q})
    base = st.session_state.api_base_url

    try:
        with st.spinner("Consultando fuentes de conocimiento…"):
            data = call_ask_api(base, q)
        answer = (data.get("answer") or "").strip() or "(Sin texto en la respuesta.)"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        update_trace_from_response(data)
    except requests.Timeout:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Tiempo de espera agotado.\n\n"
                    "El servidor tardó demasiado en responder. Comprueba que la API está activa en:\n"
                    f"{base}"
                ),
            }
        )
        clear_trace_metadata()
    except requests.ConnectionError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "No hay conexión con el backend.\n\n"
                    "Inicia el servicio FastAPI, por ejemplo:\n"
                    "uvicorn run:app --reload --host 127.0.0.1 --port 8000"
                ),
            }
        )
        clear_trace_metadata()
    except requests.HTTPError as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.text[:500]
            except Exception:
                detail = str(exc)
        code = exc.response.status_code if exc.response else "?"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error HTTP {code}\n\n{detail}",
            }
        )
        clear_trace_metadata()
    except (requests.RequestException, ValueError) as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error al contactar la API: {exc}",
            }
        )
        clear_trace_metadata()

    st.rerun()


def render_main_header() -> None:
    """Título y subtítulo corporativos en la zona principal."""
    st.markdown(
        """
        <div class="bank-main-header">
            <h1>Asistente <span class="accent">Orquestador</span> Bancario</h1>
            <p class="bank-subtitle">
                Consulta centralizada de reseñas, documento BRE-B y portafolio de productos.
                Las respuestas se generan a partir de las fuentes seleccionadas por el orquestador.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggested_questions() -> None:
    """Muestra preguntas sugeridas como botones secundarios."""
    st.markdown(
        """
        <div class="suggest-card">
            <h3>Preguntas sugeridas</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(
                suggestion,
                key=f"sug_{i}",
                use_container_width=True,
                type="secondary",
            ):
                submit_question(suggestion)


def render_sidebar() -> None:
    """Barra lateral: marca, descripción, trazabilidad y limpiar chat."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">Orquestador <span>inteligente</span></div>
            <div class="sb-desc">
                Demo ejecutiva: enrutamiento multi-fuente (reseñas, BRE-B, catálogo)
                con trazabilidad del flujo.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_input(
            "URL base de la API",
            key="api_base_url_input",
            help="Sin barra final. Ej: http://localhost:8000 o http://127.0.0.1:8000",
        )
        st.session_state.api_base_url = st.session_state.api_base_url_input.strip().rstrip(
            "/"
        )

        st.markdown('<div class="sb-section-title">Trazabilidad</div>', unsafe_allow_html=True)

        route = st.session_state.last_route
        if route is not None:
            st.markdown(
                f'<span class="badge badge-route">Ruta {html.escape(str(route))}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Sin datos aún. Envía una pregunta para ver la ruta detectada.")

        sources = st.session_state.last_sources_used
        if sources:
            st.caption("Fuentes usadas")
            parts = [
                f'<span class="badge badge-source">{html.escape(str(s))}</span>'
                for s in sources
            ]
            st.markdown(" ".join(parts), unsafe_allow_html=True)
        else:
            st.caption("Fuentes: —")

        trace = st.session_state.last_trace
        if trace:
            with st.expander("Pasos ejecutados (trace)", expanded=False):
                st.code(
                    json.dumps(trace, indent=2, ensure_ascii=False),
                    language="json",
                )
        else:
            st.caption("Trace: vacío hasta la primera respuesta exitosa.")

        st.markdown("---")
        if st.button("Limpiar conversación", use_container_width=True, type="primary"):
            st.session_state.messages = []
            clear_trace_metadata()
            st.rerun()


def render_chat_history() -> None:
    """
    Muestra el historial con ``st.chat_message`` para que el texto del asistente
    se renderice como Markdown y no aparezcan etiquetas HTML crudas en pantalla.
    """
    for msg in st.session_state.messages:
        role = msg.get("role") or "assistant"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        cm_role = "user" if role == "user" else "assistant"
        with st.chat_message(cm_role):
            st.markdown(content)


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Asistente Orquestador Bancario",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    inject_custom_css()
    render_sidebar()

    render_main_header()

    if not st.session_state.messages:
        render_suggested_questions()
    else:
        render_chat_history()

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Escriba su consulta",
            placeholder=PLACEHOLDER_INPUT,
            label_visibility="collapsed",
        )
        send = st.form_submit_button("Enviar", type="primary")

    if send and user_text.strip():
        submit_question(user_text)


if __name__ == "__main__":
    main()
