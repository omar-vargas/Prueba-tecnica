"""
Herramienta híbrida de reseñas: SQLite (estructura) + TF-IDF (relevancia) + LLM opcional (síntesis).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.settings import get_settings

logger = logging.getLogger(__name__)

_REVIEWS_SELECT = """
    SELECT branch_id, user_id, comment
    FROM reviews
    WHERE comment IS NOT NULL AND TRIM(comment) != ''
"""

# Palabras clave típicas de sedes / ciudades (matching simple en la pregunta)
_DEFAULT_BRANCH_KEYWORDS: tuple[str, ...] = (
    "CHAPINERO",
    "USAQUÉN",
    "USAQUEN",
    "ROSA",
    "SUBA",
    "ENGATIVÁ",
    "ENGATIVA",
    "BOSA",
    "KENNEDY",
    "FONTIBÓN",
    "FONTIBON",
    "BOGOTÁ",
    "BOGOTA",
    "BOG",
    "MEDELLÍN",
    "MEDELLIN",
    "MED",
    "CALI",
    "BARRANQUILLA",
    "CARTAGENA",
    "BUCARAMANGA",
    "PEREIRA",
    "MANIZALES",
)


def extract_branch_from_question(
    question: str,
    known_branch_ids: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Intenta detectar una sede o zona en la pregunta (matching de texto simple).

    Prioridad:
    1. ``branch_id`` completo citado en la pregunta.
    2. Trozos de ``branch_id`` (p. ej. *chapinero* dentro de ``BOG-CHAPINERO-01``).
    3. Patrones ``sede`` / ``agencia`` + zona (con *de la*, *del*, *la*, etc.).
    4. Lista de palabras clave comunes (CHAPINERO, SUBA, MEDELLÍN, etc.).

    El filtro SQL en memoria usa ``branch_keyword.lower() in branch_id.lower()``.
    """
    if not question.strip():
        return None

    q_lower = question.lower()

    if known_branch_ids:
        bids = sorted({str(b) for b in known_branch_ids if str(b)}, key=len, reverse=True)
        for bid in bids:
            if bid.lower() in q_lower:
                return str(bid)

        best_seg: Optional[str] = None
        best_len = 0
        for bid in bids:
            for segment in re.split(r"[-_\s]+", bid.lower()):
                seg = segment.strip()
                if len(seg) < 4 or seg.isdigit():
                    continue
                if seg in q_lower and len(seg) > best_len:
                    best_seg = seg
                    best_len = len(seg)
        if best_seg:
            logger.debug("Sede inferida por segmento de branch_id: %r", best_seg)
            return best_seg

    m = re.search(
        r"\b(?:sede|agencia|oficina|branch)\s+"
        r"(?:de\s+(?:la\s+)?|del\s+|el\s+|la\s+)?"
        r"([A-Za-z0-9áéíóúÁÉÍÓÚñÑ]{3,})",
        question,
        flags=re.IGNORECASE,
    )
    if m:
        token = m.group(1).strip()
        noise = {"del", "las", "los", "una", "para", "por", "con"}
        if len(token) >= 3 and token.lower() not in noise:
            logger.debug("Sede inferida por regex tras sede/agencia: %r", token)
            return token

    q_fold = question.upper()
    replacements = str.maketrans("ÁÉÍÓÚÑ", "AEIOUN")
    q_ascii = q_fold.translate(replacements)

    for kw in sorted(_DEFAULT_BRANCH_KEYWORDS, key=len, reverse=True):
        k_fold = kw.upper()
        k_ascii = k_fold.translate(replacements)
        if k_fold in q_fold or k_ascii in q_ascii:
            return kw

    return None


class ReviewsRetriever:
    """
    Carga ``reviews`` desde SQLite y construye TF-IDF sobre el texto ``comment``.

    El filtrado por sede opera sobre ``branch_id`` alineado con cada fila cargada.
    """

    def __init__(self, db_path: str) -> None:
        """
        Args:
            db_path: Ruta al archivo SQLite que contiene la tabla ``reviews``.
        """
        self._db_path = Path(db_path)
        self.branch_ids: List[str] = []
        self.user_ids: List[str] = []
        self.comments: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix: Any = None

        self._load_from_sqlite()
        self._build_tfidf()

    def _load_from_sqlite(self) -> None:
        """Lee todas las filas válidas de ``reviews``."""
        if not self._db_path.is_file():
            logger.warning("No existe la base SQLite: %s", self._db_path)
            return

        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cur = conn.execute(_REVIEWS_SELECT)
                rows = cur.fetchall()
            finally:
                conn.close()
        except Exception:
            logger.exception("Error leyendo tabla reviews en %s", self._db_path)
            return

        for row in rows:
            b, u, c = str(row[0] or ""), str(row[1] or ""), str(row[2] or "").strip()
            if not c:
                continue
            self.branch_ids.append(b)
            self.user_ids.append(u)
            self.comments.append(c)

        logger.info(
            "ReviewsRetriever: %s filas cargadas desde %s",
            len(self.comments),
            self._db_path,
        )

    def _build_tfidf(self) -> None:
        """Ajusta ``TfidfVectorizer`` sobre ``comments``."""
        if not self.comments:
            logger.warning("Corpus vacío; no se construye TF-IDF.")
            return

        try:
            self._vectorizer = TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(self.comments)
            logger.info("TF-IDF listo, shape=%s", getattr(self._matrix, "shape", None))
        except Exception:
            logger.exception("Fallo al construir TF-IDF.")
            self._vectorizer = None
            self._matrix = None

    def distinct_branch_ids(self) -> List[str]:
        """Valores únicos de ``branch_id`` en memoria."""
        seen: set[str] = set()
        out: List[str] = []
        for b in self.branch_ids:
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

    def filter_indices_by_branch(self, branch_keyword: str) -> List[int]:
        """
        Índices de filas cuyo ``branch_id`` contiene ``branch_keyword`` (sin distinguir mayúsculas).

        Args:
            branch_keyword: Subcadena o identificador de sede.

        Returns:
            Lista de índices en ``0..len(comments)-1``.
        """
        if not branch_keyword.strip():
            return list(range(len(self.comments)))

        key = branch_keyword.lower()
        return [i for i, b in enumerate(self.branch_ids) if key in b.lower()]

    def get_relevant_comments(
        self,
        query: str,
        top_k: int = 5,
        branch_keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rankea comentarios por similitud coseno (TF-IDF), opcionalmente acotado por sede.

        Args:
            query: Consulta en lenguaje natural.
            top_k: Número máximo de resultados.
            branch_keyword: Si se indica, solo se consideran filas cuyo ``branch_id`` lo contiene.

        Returns:
            Lista de dicts con ``branch_id``, ``user_id``, ``comment``, ``score`` (float).
        """
        if not query.strip() or self._vectorizer is None or self._matrix is None:
            return []

        indices = (
            self.filter_indices_by_branch(branch_keyword)
            if branch_keyword
            else list(range(len(self.comments)))
        )
        if not indices:
            logger.info("Sin índices tras filtro branch_keyword=%r.", branch_keyword)
            return []

        try:
            q_vec = self._vectorizer.transform([query])
            sub = self._matrix[indices]
            sims = cosine_similarity(q_vec, sub)[0]
        except Exception:
            logger.exception("Error en similitud TF-IDF.")
            return []

        order = sims.argsort()[::-1][: min(top_k, len(indices))]
        out: List[Dict[str, Any]] = []
        for j in order:
            idx = indices[j]
            out.append(
                {
                    "branch_id": self.branch_ids[idx],
                    "user_id": self.user_ids[idx],
                    "comment": self.comments[idx],
                    "score": float(sims[j]),
                }
            )
        return out


@lru_cache(maxsize=8)
def _retriever_for_db(db_path_resolved: str) -> ReviewsRetriever:
    """Instancia cacheada por ruta canónica de la BD."""
    return ReviewsRetriever(db_path_resolved)


def _get_retriever() -> ReviewsRetriever:
    path = get_settings().sqlite_db_path.resolve()
    return _retriever_for_db(str(path))


def clear_reviews_retriever_cache() -> None:
    """Vacía la caché del recuperador (tras repoblar SQLite en arranque)."""
    _retriever_for_db.cache_clear()


def _summarize_reviews_llm(
    llm: BaseChatModel,
    question: str,
    raw_comments: List[Dict[str, Any]],
    branch_filter: Optional[str],
) -> str:
    """Genera resumen estructurado a partir de comentarios recuperados."""
    lines = []
    for item in raw_comments[:25]:
        lines.append(
            f"- [{item.get('branch_id')}] (sim={item.get('score', 0):.3f}) {item.get('comment', '')[:1500]}"
        )
    block = "\n".join(lines) if lines else "(sin comentarios)"

    branch_note = f"Filtro sede: {branch_filter}. " if branch_filter else ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres analista de experiencia de cliente bancario. Responde en español usando solo los comentarios proporcionados.

Incluye secciones claras:
1) Aspectos positivos
2) Problemas principales
3) Nivel de satisfacción (cualitativo; no inventes porcentajes ni cifras que no estén en los textos)

Si la evidencia es insuficiente, dilo explícitamente.
                """.strip(),
            ),
            (
                "human",
                """
{branch_note}Pregunta del usuario: {question}

Comentarios recuperados (TF-IDF + filtro SQLite cuando aplica):
{comments_block}
                """.strip(),
            ),
        ]
    )
    out = (prompt | llm).invoke(
        {
            "branch_note": branch_note,
            "question": question,
            "comments_block": block,
        }
    )
    text = out.content if isinstance(out.content, str) else str(out.content)
    return text.strip()


def query_reviews(
    question: str,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Pipeline híbrido: detección de sede, filtro lógico por ``branch_id``, ranking TF-IDF, síntesis LLM opcional.

    Args:
        question: Pregunta del usuario.
        llm: Si se proporciona, genera ``content`` como resumen; si no, ``content`` son los comentarios formateados.

    Returns:
        Dict con ``content``, ``raw_comments`` (lista de dicts con score), ``branch_filter`` (str o None),
        y metadatos útiles (``source``, ``db_path``, etc.).
    """
    settings = get_settings()
    db_path = settings.sqlite_db_path

    try:
        retriever = _get_retriever()
    except Exception:
        logger.exception("No se pudo crear ReviewsRetriever.")
        return {
            "content": "Error al acceder al recuperador de reseñas.",
            "raw_comments": [],
            "branch_filter": None,
            "source": "reviews_hybrid",
            "db_path": str(db_path),
            "error": True,
        }

    known = retriever.distinct_branch_ids()
    branch_kw = extract_branch_from_question(question, known_branch_ids=known)

    top_k = 8
    raw = retriever.get_relevant_comments(
        question,
        top_k=top_k,
        branch_keyword=branch_kw,
    )
    logger.info(
        "query_reviews: branch_filter=%r filas_indexadas=%s comentarios_devueltos=%s",
        branch_kw,
        len(retriever.comments),
        len(raw),
    )

    base: Dict[str, Any] = {
        "raw_comments": raw,
        "branch_filter": branch_kw,
        "source": "reviews_sqlite_tfidf",
        "db_path": str(db_path),
        "rows_indexed": len(retriever.comments),
        "comments_returned": len(raw),
    }

    if not raw:
        base["content"] = (
            "No hay comentarios en la base o ninguno coincide con el filtro / la consulta. "
            f"Base: {db_path}"
        )
        return base

    if llm is not None:
        try:
            logger.info("Síntesis LLM sobre %s comentarios recuperados.", len(raw))
            base["content"] = _summarize_reviews_llm(llm, question, raw, branch_kw)
            base["llm_summary"] = True
        except Exception:
            logger.exception("Fallo síntesis LLM; se devuelven comentarios en texto plano.")
            base["llm_summary"] = False
            base["llm_error"] = True
            base["content"] = _format_raw_as_content(raw)
    else:
        base["content"] = _format_raw_as_content(raw)
        base["llm_summary"] = False

    return base


def _format_raw_as_content(raw: List[Dict[str, Any]]) -> str:
    """Formatea la lista recuperada para el compositor sin LLM en la tool."""
    parts: List[str] = []
    for item in raw:
        parts.append(
            f"[{item.get('branch_id')}] (relevancia {item.get('score', 0):.3f}) {item.get('comment', '')}"
        )
    return "\n".join(parts)
