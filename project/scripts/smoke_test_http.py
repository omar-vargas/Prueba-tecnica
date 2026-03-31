"""
Prueba HTTP contra la API local (health + /ask).

Uso (con el servidor en marcha: ``uvicorn run:app --reload``)::

    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_http.py

Variables opcionales: ``ORCHESTRATOR_API_URL`` (por defecto http://127.0.0.1:8000).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    base = os.getenv("ORCHESTRATOR_API_URL", "http://127.0.0.1:8000").rstrip("/")

    try:
        req = urllib.request.Request(f"{base}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode()
        print("GET /health ->", r.status, body)
        try:
            h = json.loads(body)
            if h.get("data_sources"):
                print("data_sources:", json.dumps(h["data_sources"], ensure_ascii=False))
        except json.JSONDecodeError:
            pass
    except Exception as e:
        print(f"No se pudo contactar {base}/health: {e}")
        print("¿Está uvicorn ejecutándose en esa URL?")
        return 1

    payload = json.dumps({"question": "¿Qué es BRE-B en una frase?"}).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{base}/ask",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        print("POST /ask ->", r.status)
        print("route:", data.get("route"))
        print("answer (preview):", (data.get("answer") or "")[:400])
        print("sources_used:", data.get("sources_used"))
    except urllib.error.HTTPError as e:
        print("POST /ask HTTP error:", e.code, e.read().decode()[:500])
        return 2
    except Exception as e:
        print("POST /ask error:", e)
        return 2

    print("Prueba HTTP: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
