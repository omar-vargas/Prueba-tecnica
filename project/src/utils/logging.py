"""
Configuración centralizada de logging para la aplicación.

Usa el estándar `logging` de Python con formato legible y nivel configurable por variable de entorno.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    name: str = "orchestrator",
) -> logging.Logger:
    """
    Configura el root logger (si aún no tiene handlers) y devuelve un logger con nombre dado.

    Args:
        level: Nivel textual (DEBUG, INFO, ...). Si es None, usa LOG_LEVEL o INFO.
        name: Nombre del logger hijo a devolver.

    Returns:
        Logger configurado listo para usar.
    """
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.setLevel(numeric)

    logger = logging.getLogger(name)
    logger.setLevel(numeric)
    return logger
