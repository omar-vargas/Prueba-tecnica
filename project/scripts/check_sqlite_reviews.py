"""Inspección rápida de bank_reviews.sqlite (uso puntual)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "bank_reviews.sqlite"


def main() -> int:
    print("Ruta:", DB.resolve())
    print("Existe:", DB.is_file())
    if not DB.is_file():
        return 1

    conn = sqlite3.connect(str(DB))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in cur.fetchall()]
        print("Tablas:", tables)

        cur = conn.execute("SELECT COUNT(*) FROM reviews")
        total = cur.fetchone()[0]
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM reviews
            WHERE comment IS NOT NULL AND TRIM(comment) != ''
            """
        )
        with_comment = cur.fetchone()[0]
        print("Filas totales en reviews:", total)
        print("Filas con comentario no vacío:", with_comment)

        cur = conn.execute(
            """
            SELECT COUNT(*) FROM reviews
            WHERE UPPER(IFNULL(branch_id,'')) LIKE '%CHAPINERO%'
            """
        )
        print("Filas con branch_id tipo CHAPINERO:", cur.fetchone()[0])

        cur = conn.execute(
            """
            SELECT branch_id, substr(comment, 1, 70)
            FROM reviews
            WHERE comment IS NOT NULL AND TRIM(comment) != ''
            LIMIT 5
            """
        )
        print("Muestra (5 filas):")
        for bid, prev in cur.fetchall():
            print(f"  {bid} | {prev!r}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
