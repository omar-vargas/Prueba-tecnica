"""
Cliente SQLite para consultas sobre datos estructurados (p. ej. reseñas desde Excel ingestado).

Gestiona conexión por ruta de archivo y expone un context manager para sesiones seguras.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Tuple


class SQLiteClient:
    """Encapsula la conexión a una base SQLite en disco."""

    def __init__(self, db_path: Path) -> None:
        """
        Args:
            db_path: Ruta absoluta o relativa al archivo .sqlite / .db.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        """Ruta del archivo de base de datos."""
        return self._db_path

    def connect(self) -> sqlite3.Connection:
        """
        Abre una conexión nueva (el llamador debe cerrarla o usar `session`).

        Returns:
            Conexión SQLite con `row_factory` en modo fila accesible por nombre.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager que confirma transacciones al salir sin excepción.

        Yields:
            Conexión activa.
        """
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_fetchall(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """
        Ejecuta una consulta de lectura y devuelve todas las filas.

        Args:
            sql: Sentencia SQL parametrizada.
            params: Tupla de parámetros (evita SQL injection).

        Returns:
            Lista de filas tipo sqlite3.Row.
        """
        with self.session() as conn:
            cur = conn.execute(sql, params or ())
            return list(cur.fetchall())

    def executemany(
        self,
        sql: str,
        seq_of_params: Iterable[Tuple[Any, ...]],
    ) -> None:
        """
        Ejecuta la misma sentencia con múltiples conjuntos de parámetros (ingesta por lotes).

        Args:
            sql: Sentencia SQL.
            seq_of_params: Secuencia de tuplas de parámetros.
        """
        with self.session() as conn:
            conn.executemany(sql, seq_of_params)
