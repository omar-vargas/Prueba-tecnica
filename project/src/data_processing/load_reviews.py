"""
Ingesta del Excel bank_reviews_colombia hacia SQLite (tabla ``reviews``).

Lee con pandas/openpyxl, normaliza columnas a ``branch_id``, ``user_id``, ``comment``,
elimina duplicados y comentarios vacíos, y persiste con ``executemany``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from storage.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)

_COLUMN_ALIASES: Dict[str, Set[str]] = {
    "branch_id": {
        "branch_id",
        "sede",
        "sede_id",
        "id_sede",
        "branch",
        "agencia",
        "office",
        "codigo_sede",
        "código_sede",
    },
    "user_id": {
        "user_id",
        "usuario",
        "id_usuario",
        "user",
        "customer_id",
        "cliente_id",
        "cliente",
    },
    "comment": {
        "comment",
        "comments",
        "comentario",
        "comentarios",
        "review",
        "texto",
        "mensaje",
        "observacion",
        "observación",
    },
}


def _standardize_header(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "", s, flags=re.UNICODE)
    return s


def _map_columns_to_schema(columns: Iterable[str]) -> Dict[str, str]:
    standardized = {_standardize_header(c): c for c in columns}
    reverse_alias: Dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            std = _standardize_header(alias)
            reverse_alias[std] = canonical

    mapping: Dict[str, str] = {}
    for std_name, original in standardized.items():
        if std_name in reverse_alias:
            canonical = reverse_alias[std_name]
            if canonical not in mapping.values():
                mapping[original] = canonical

    found = set(mapping.values())
    required = {"branch_id", "user_id", "comment"}
    missing = required - found
    if missing:
        logger.error(
            "No se pudieron mapear columnas requeridas %s. Columnas en Excel: %s",
            missing,
            list(columns),
        )
        raise ValueError(
            f"No se encontraron columnas para: {sorted(missing)}. "
            f"Columnas leídas: {list(columns)}"
        )
    return mapping


def create_reviews_table(conn: sqlite3.Connection) -> None:
    """
    Crea la tabla ``reviews`` si no existe.

    Esquema: id, branch_id, user_id, comment.
    """
    logger.info("Creando tabla reviews (si no existe).")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_id TEXT,
            user_id TEXT,
            comment TEXT
        )
        """
    )
    conn.commit()


def load_reviews_excel_to_sqlite(excel_path: str, sqlite_path: str) -> None:
    """
    Lee Excel, limpia y vuelca en SQLite (recrea la tabla ``reviews``).

    Raises:
        FileNotFoundError: Si no existe el Excel.
        ValueError: Si el Excel está vacío o no quedan filas válidas.
    """
    excel_file = Path(excel_path)
    if not excel_file.is_file():
        logger.error("No existe el archivo Excel: %s", excel_path)
        raise FileNotFoundError(f"No existe el archivo: {excel_path}")

    sqlite_file = Path(sqlite_path)
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Leyendo Excel: %s", excel_path)
        df = pd.read_excel(excel_path, engine="openpyxl")
    except Exception:
        logger.exception("Fallo al leer el Excel con pandas/openpyxl.")
        raise

    if df.empty:
        logger.warning("El Excel no contiene filas.")
        raise ValueError("El archivo Excel está vacío.")

    col_map = _map_columns_to_schema(df.columns)
    df = df.rename(columns=col_map)
    df = df[["branch_id", "user_id", "comment"]]

    for col in ("branch_id", "user_id", "comment"):
        df[col] = df[col].apply(lambda x: "" if pd.isna(x) else str(x).strip())

    before = len(df)
    df = df[df["comment"] != ""]
    dropped_empty = before - len(df)
    if dropped_empty:
        logger.info("Descartadas %s filas con comentario vacío.", dropped_empty)

    before = len(df)
    df = df.drop_duplicates(subset=["branch_id", "user_id", "comment"], keep="first")
    dup = before - len(df)
    if dup:
        logger.info("Eliminados %s duplicados (branch_id, user_id, comment).", dup)

    if df.empty:
        logger.error("No quedan filas tras limpiar comentarios vacíos y duplicados.")
        raise ValueError("No hay filas válidas para insertar.")

    rows: List[Tuple[str, str, str]] = list(
        zip(df["branch_id"], df["user_id"], df["comment"])
    )

    try:
        conn = sqlite3.connect(str(sqlite_file))
        try:
            logger.info("Recreando tabla reviews en: %s", sqlite_path)
            conn.execute("DROP TABLE IF EXISTS reviews")
            create_reviews_table(conn)

            logger.info("Insertando %s filas en reviews.", len(rows))
            conn.executemany(
                "INSERT INTO reviews (branch_id, user_id, comment) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
            logger.info("Inserción completada correctamente.")
        finally:
            conn.close()
    except Exception:
        logger.exception("Error al escribir en SQLite: %s", sqlite_path)
        raise


def fetch_all_comments(sqlite_path: str) -> List[Tuple[str, str, str]]:
    """
    Devuelve (branch_id, user_id, comment) por cada fila, ordenado por id.
    """
    db = Path(sqlite_path)
    if not db.is_file():
        logger.error("No existe la base SQLite: %s", sqlite_path)
        raise FileNotFoundError(f"No existe la base de datos: {sqlite_path}")

    uri = f"file:{db.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            logger.debug("Leyendo comentarios desde: %s", sqlite_path)
            cur = conn.execute(
                "SELECT branch_id, user_id, comment FROM reviews ORDER BY id"
            )
            return [
                (
                    str(r[0] if r[0] is not None else ""),
                    str(r[1] if r[1] is not None else ""),
                    str(r[2] if r[2] is not None else ""),
                )
                for r in cur.fetchall()
            ]
        finally:
            conn.close()
    except Exception:
        logger.exception("Error al leer comentarios desde SQLite.")
        raise


def excel_to_sqlite(
    excel_path: Path,
    sqlite_client: SQLiteClient,
    table_name: str = "reviews",
) -> int:
    """
    Compatibilidad: delega en ``load_reviews_excel_to_sqlite`` usando la ruta del cliente.

    Args:
        excel_path: Ruta al .xlsx.
        sqlite_client: Cliente SQLite del proyecto.
        table_name: Debe ser ``reviews``; otro valor emite advertencia.

    Returns:
        Número de filas en ``reviews`` tras la carga.
    """
    if table_name != "reviews":
        logger.warning("table_name=%s ignorado; se usa la tabla reviews.", table_name)
    load_reviews_excel_to_sqlite(str(excel_path), str(sqlite_client.db_path))
    return len(fetch_all_comments(str(sqlite_client.db_path)))
