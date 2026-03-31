# Agente orquestador RAG (FastAPI + LangChain)

Sistema que orquesta consultas sobre **reseñas bancarias** (SQLite desde Excel), **RAG sobre documentación BRE-B** (FAISS + embeddings locales) y un **catálogo de productos** (JSON). Incluye API REST y una interfaz opcional con Streamlit.

El código vive en la carpeta [`project/`](project/).

## Requisitos

- **Python 3.11 o superior** (recomendado).
- **Git** (para clonar y versionar).
- Credenciales de **Azure OpenAI** (tres variables) **o** **OpenAI** (`OPENAI_API_KEY`), según configures en `.env`.
- Espacio en disco razonable: la primera ejecución puede descargar modelos de embeddings (sentence-transformers).

## Instalación paso a paso

### 1. Clonar o copiar el repositorio

```bash
git clone <URL-de-tu-repo-github>
cd <nombre-del-repo>
```

Si trabajas solo con la carpeta `project/`, todos los comandos siguientes se ejecutan **dentro de `project/`**.

### 2. Entorno virtual e dependencias

**Windows (PowerShell):**

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3. Variables de entorno

Copia el ejemplo y edita los valores (no subas `.env` a Git; ya está ignorado).

**Windows:**

```powershell
copy .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

La aplicación carga `project/.env` automáticamente aunque ejecutes Uvicorn desde otro directorio.

| Variable | Descripción |
|----------|-------------|
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` | Modo prioritario si las tres están definidas. |
| `AZURE_OPENAI_API_VERSION` o `OPENAI_API_VERSION` | Versión de la API REST de Azure. |
| `OPENAI_API_KEY`, `OPENAI_DEFAULT_MODEL` | OpenAI directo si no usas Azure completo. |
| `LLM_TEMPERATURE` | Temperatura del modelo (por defecto `0`). |
| `SQLITE_DB_PATH`, `PRODUCT_CATALOG_PATH`, `BREB_FAISS_INDEX_DIR`, `DATA_*_DIR` | Rutas de datos (relativas a `project/`). |
| `REVIEWS_EXCEL_PATH` | Excel de reseñas; si no existe, en el arranque se puede buscar un `.xlsx` en `data/docs` o `data/raw`. |
| `ORCHESTRATOR_API_URL` | URL de la API para Streamlit (por defecto `http://127.0.0.1:8000`). |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### 4. Datos e índices (primera vez)

- Coloca el **Excel de reseñas** donde indique `REVIEWS_EXCEL_PATH`, o en `data/raw` / `data/docs`. Al arrancar la API, si SQLite está vacío, puede ingerirse desde el Excel.
- **Índice BRE-B (FAISS)** — desde `project/` con el entorno activado:

  **Windows:**

  ```powershell
  $env:PYTHONPATH = "src"
  python scripts\build_breb_index.py
  ```

  **Linux / macOS:**

  ```bash
  PYTHONPATH=src python scripts/build_breb_index.py
  ```

  Opciones: `--pdf`, `--index`, `--no-demo` (ver ayuda con `python scripts/build_breb_index.py -h`).

- **Catálogo JSON de productos** (requiere LLM configurado en `.env`):

  **Windows:**

  ```powershell
  $env:PYTHONPATH = "src"
  python scripts\build_products_catalog.py
  ```

  **Linux / macOS:**

  ```bash
  PYTHONPATH=src python scripts/build_products_catalog.py
  ```

Si necesitas forzar que la API no arranque sin datos completos, revisa `STRICT_DATA_BOOTSTRAP` en `.env.example`.

## Ejecutar el sistema

### API (FastAPI)

Desde la carpeta **`project/`**:

```bash
uvicorn run:app --reload
```

Documentación interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Comprobar salud:

```bash
curl http://127.0.0.1:8000/health
```

Ejemplo de pregunta:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"¿Qué dice el estándar BRE-B sobre transferencias?\"}"
```

### Interfaz Streamlit (opcional)

Con la API en marcha, en otra terminal (desde `project/`):

```bash
streamlit run app/streamlit_app.py
```

## Estructura del proyecto

| Ruta | Contenido |
|------|-----------|
| `project/src/api` | FastAPI (`POST /ask`, `GET /health`). |
| `project/src/orchestrator` | Router de intenciones y servicio orquestador. |
| `project/src/tools` | Herramientas (reseñas, RAG, productos). |
| `project/src/data_processing` | Carga Excel/PDF y chunking. |
| `project/src/storage` | SQLite, FAISS, catálogo JSON. |
| `project/data/` | Raw, procesados, índices y vectorstore. |
| `project/app/streamlit_app.py` | UI demo. |
| `project/notebooks/` | Notebooks de exploración. |
| `project/scripts/` | Scripts de construcción de índice y utilidades. |

Más detalle técnico en [`project/README.md`](project/README.md).

## Solución de problemas

- **Error de importación al usar Uvicorn:** ejecuta siempre desde `project/` con `uvicorn run:app`, o usa `uvicorn api.main:app --reload --app-dir src` (el directorio debe ser `src`, no otro nombre).
- **Sin respuestas del LLM:** revisa que Azure u OpenAI estén bien configurados en `.env` y que no haya cuotas o firewall bloqueando.
- **RAG vacío o errores BRE-B:** verifica que exista la carpeta del índice en `BREB_FAISS_INDEX_DIR` y que hayas ejecutado `build_breb_index` al menos una vez.

## Publicar en GitHub

En la carpeta del repositorio (`Prueba RAG`) ya está inicializado **Git** con commits listos para subir.

### Opción A: GitHub CLI (recomendado)

1. Inicia sesión (una vez por máquina):

   ```bash
   gh auth login
   ```

2. Crea el repositorio remoto y sube el código (elige un nombre sin espacios, por ejemplo `prueba-rag-orquestador`):

   ```bash
   cd "ruta\a\Prueba RAG"
   git branch -M main
   gh repo create TU_USUARIO/prueba-rag-orquestador --public --source=. --remote=origin --push
   ```

   Si el repo ya existe vacío en GitHub:

   ```bash
   git remote add origin https://github.com/TU_USUARIO/NOMBRE_REPO.git
   git branch -M main
   git push -u origin main
   ```

### Opción B: Solo Git

1. En [github.com/new](https://github.com/new) crea un repositorio **vacío** (sin README ni `.gitignore`).
2. En tu PC:

   ```bash
   cd "ruta\a\Prueba RAG"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/NOMBRE_REPO.git
   git push -u origin main
   ```

Usa **HTTPS** con un [personal access token](https://github.com/settings/tokens) como contraseña, o configura **SSH** (`git@github.com:...`).

## Licencia

Publica hecha por OMAR VARGAS para Banco de Bogota
