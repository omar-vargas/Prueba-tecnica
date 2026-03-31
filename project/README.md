# Agente orquestador (FastAPI + LangChain)

Guía de instalación detallada (español, Windows/Linux): ver el [`README.md`](../README.md) en la raíz del repositorio.

Prueba técnica: orquestador que integra reseñas en SQLite (Excel), RAG sobre PDF BRE-B (FAISS) y catálogo de productos en JSON.

## Estructura

- `src/api` — FastAPI (`POST /ask`, `GET /health`).
- `src/orchestrator` — Router de intenciones, servicio y composición de respuesta.
- `src/tools` — Stubs de herramientas (reviews, RAG, productos).
- `src/data_processing` — Carga Excel/PDF y chunking.
- `src/storage` — SQLite, FAISS (base), catálogo JSON.
- `src/models` — Pydantic request/response.
- `data/raw`, `data/processed`, `data/indexes` — Datos e índices.
- `app/streamlit_app.py` — UI demo opcional.
- `notebooks/demo_orquestador.ipynb` — Exploración manual.

## Requisitos

- Python 3.11+ recomendado.

## Instalación

```bash
cd project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edita `.env` (no lo subas al repositorio; está en `.gitignore`). La app carga automáticamente `project/.env` aunque ejecutes Uvicorn desde otro directorio.

### Variables principales

| Variable | Uso |
|----------|-----|
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure OpenAI (modo prioritario si las tres están definidas). |
| `AZURE_OPENAI_API_VERSION` o `OPENAI_API_VERSION` | Versión de la API REST de Azure. |
| `OPENAI_API_KEY`, `OPENAI_DEFAULT_MODEL` | OpenAI público si no configuras Azure completo. |
| `LLM_TEMPERATURE` | Temperatura del chat (por defecto `0`). |
| `SQLITE_DB_PATH`, `PRODUCT_CATALOG_PATH`, `DATA_*_DIR` | Rutas usadas por las tools y ETL (relativas a `project/`). |
| `ORCHESTRATOR_API_URL` | URL de la API para Streamlit. |
| `LOG_LEVEL` | Nivel de logging. |

## Ejecutar la API

Desde la carpeta `project` (recomendado — no hace falta `--app-dir`):

```bash
uvicorn run:app --reload
```

Alternativa equivalente (cuidado: el directorio debe ser **`src`**, no `sr`):

```bash
uvicorn api.main:app --reload --app-dir src
```

Documentación interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Probar `/ask`

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"¿Qué dice el estándar BRE-B sobre transferencias?\"}"
```

## Streamlit (opcional)

Con la API en marcha:

```bash
streamlit run app/streamlit_app.py
```

## Próximos pasos

1. Colocar `bank_reviews_colombia.xlsx` en `data/raw` y completar `load_reviews.excel_to_sqlite`.
2. Colocar el PDF BRE-B en `data/raw`, extraer texto, generar embeddings e indexar FAISS.
3. Definir el JSON del portafolio y enlazar `ProductCatalog` con `products_tool`.
