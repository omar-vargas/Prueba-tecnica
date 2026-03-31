"""
Valida ingesta Excel → SQLite: esquema de ``reviews``, conteo y muestra de filas.

Sin argumentos: genera un Excel mínimo en ``data/raw/_ingest_validation.xlsx``,
carga en una BD temporal y comprueba resultados.

Uso::

    .\\.venv\\Scripts\\python.exe scripts\\validate_reviews_ingest.py
    .\\.venv\\Scripts\\python.exe scripts\\validate_reviews_ingest.py --excel docs\\bank_reviews_colombia.xlsx --sqlite data\\processed\\bank_reviews.sqlite
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("validate_reviews_ingest")


def _expected_columns() -> list[tuple[str, str]]:
    return [
        ("id", "INTEGER"),
        ("branch_id", "TEXT"),
        ("user_id", "TEXT"),
        ("comment", "TEXT"),
    ]


def inspect_sqlite(sqlite_path: Path) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.execute("PRAGMA table_info(reviews)")
        cols = [(r[1], r[2]) for r in cur.fetchall()]
        logger.info("PRAGMA table_info(reviews): %s", cols)

        expected = _expected_columns()
        names_got = [c[0] for c in cols]
        names_exp = [c[0] for c in expected]
        if names_got != names_exp:
            logger.error("Esquema distinto al esperado. Esperado %s, obtenido %s", names_exp, names_got)
            raise SystemExit(2)

        cur = conn.execute("SELECT COUNT(*) FROM reviews")
        n = cur.fetchone()[0]
        logger.info("Filas en reviews: %s", n)

        cur = conn.execute("SELECT id, branch_id, user_id, comment FROM reviews ORDER BY id LIMIT 5")
        for row in cur.fetchall():
            logger.info("Muestra: id=%s branch_id=%r user_id=%r comment=%r", *row)
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validar ingesta de reseñas.")
    parser.add_argument("--excel", type=Path, help="Ruta al .xlsx (si se omite, se crea muestra)")
    parser.add_argument("--sqlite", type=Path, help="Ruta al .sqlite destino (default: temporal)")
    args = parser.parse_args()

    import pandas as pd

    from data_processing.load_reviews import fetch_all_comments, load_reviews_excel_to_sqlite

    if args.excel and args.excel.is_file():
        excel_path = args.excel.resolve()
        logger.info("Usando Excel existente: %s", excel_path)
    else:
        if args.excel:
            logger.error("No existe el Excel: %s", args.excel)
            return 1
        raw_dir = ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        excel_path = raw_dir / "_ingest_validation.xlsx"
        df = pd.DataFrame(
            {
                "sede": ["BOG-01", "BOG-01", "MED-02", "BOG-01"],
                "usuario": ["u1", "u2", "u3", "u1"],
                "comentario": [
                    "Excelente atención en ventanilla",
                    "",
                    "Demora excesiva",
                    "Excelente atención en ventanilla",
                ],
            }
        )
        df.to_excel(excel_path, index=False)
        logger.info("Creado Excel de prueba: %s (4 filas, 1 comentario vacío, 1 duplicado)", excel_path)

    if args.sqlite:
        sqlite_path = args.sqlite.resolve()
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp = tempfile.mkstemp(suffix=".sqlite")
        import os

        os.close(fd)
        sqlite_path = Path(tmp)
        logger.info("Usando SQLite temporal: %s", sqlite_path)

    try:
        load_reviews_excel_to_sqlite(str(excel_path), str(sqlite_path))
    except Exception as e:
        logger.exception("Fallo load_reviews_excel_to_sqlite: %s", e)
        return 3

    inspect_sqlite(sqlite_path)

    tuples = fetch_all_comments(str(sqlite_path))
    logger.info("fetch_all_comments: %s tuplas", len(tuples))

    if args.excel and args.excel.is_file():
        pass
    else:
        if len(tuples) != 2:
            logger.error("Validación muestra: se esperaban 2 filas (vacío y duplicado fuera), hay %s", len(tuples))
            return 4
        branches = {t[0] for t in tuples}
        if branches != {"BOG-01", "MED-02"}:
            logger.error("Sedes inesperadas: %s", branches)
            return 4

    logger.info("Validación OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
