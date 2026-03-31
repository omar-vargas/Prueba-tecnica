"""
Comprueba credenciales y una llamada mínima al LLM (Azure u OpenAI).

Uso (desde la carpeta `project/`)::

    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_llm.py

Requiere `project/.env` con Azure OpenAI completo o `OPENAI_API_KEY`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from langchain_core.messages import HumanMessage  # noqa: E402


def main() -> int:
    from utils.settings import PROJECT_ROOT, build_chat_llm, get_settings  # noqa: E402

    print(f"Raíz del proyecto: {PROJECT_ROOT}")
    env_file = PROJECT_ROOT / ".env"
    print(f"Archivo .env: {env_file} (existe: {env_file.is_file()})")

    settings = get_settings()
    print(
        "Config detectada:",
        f"azure_key={'sí' if settings.azure_openai_api_key else 'no'}, "
        f"azure_endpoint={'sí' if settings.azure_openai_endpoint else 'no'}, "
        f"deployment={'sí' if settings.azure_openai_deployment else 'no'}, "
        f"openai_key={'sí' if settings.openai_api_key else 'no'}",
    )

    try:
        llm = build_chat_llm()
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    print("Invocando modelo (mensaje mínimo)...")
    msg = HumanMessage(content='Responde únicamente con la palabra "OK" en mayúsculas, sin puntuación extra.')
    try:
        out = llm.invoke([msg])
    except Exception as exc:
        print(f"ERROR al llamar al proveedor: {type(exc).__name__}: {exc}")
        print(
            "Si es 401: revisa AZURE_OPENAI_API_KEY, que AZURE_OPENAI_ENDPOINT sea el de tu recurso "
            "(región correcta) y que AZURE_OPENAI_DEPLOYMENT_NAME coincida con el deployment en Azure."
        )
        return 2

    text = out.content if isinstance(out.content, str) else str(out.content)
    print("Respuesta del modelo:", repr(text[:500]))
    print("Prueba LLM: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
