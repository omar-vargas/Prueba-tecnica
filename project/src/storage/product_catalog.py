"""
Catálogo de productos bancarios en JSON: guardado, carga y consultas ligeras.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Catálogo en memoria por ruta canónica (precarga al arranque de FastAPI).
_CATALOG_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def save_product_catalog(products: List[Dict[str, Any]], output_path: str) -> None:
    """
    Persiste la lista de productos en JSON con indentación UTF-8.

    Args:
        products: Lista de dicts de producto.
        output_path: Ruta del archivo de salida.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"products": products, "count": len(products)}
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Catálogo guardado: %s (%s productos).", path.resolve(), len(products))
    except Exception:
        logger.exception("Error al escribir catálogo en %s", output_path)
        raise


def load_product_catalog(input_path: str) -> List[Dict[str, Any]]:
    """
    Carga productos desde un JSON en disco.

    Acepta:

    * lista en la raíz ``[ {...}, ... ]``
    * objeto con clave ``products``

    Args:
        input_path: Ruta al archivo JSON.

    Returns:
        Lista de productos (vacía si no existe el archivo).
    """
    path = Path(input_path)
    if not path.is_file():
        logger.warning("No existe el catálogo: %s", input_path)
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Error al leer JSON: %s", input_path)
        raise

    if isinstance(raw, list):
        return [dict(p) for p in raw]
    if isinstance(raw, dict) and isinstance(raw.get("products"), list):
        return [dict(p) for p in raw["products"]]

    logger.warning("Estructura JSON no reconocida en %s", input_path)
    return []


def clear_product_catalog_cache() -> None:
    """Vacía la caché en memoria (p. ej. tras regenerar el JSON)."""
    _CATALOG_CACHE.clear()


def load_product_catalog_cached(input_path: str) -> List[Dict[str, Any]]:
    """
    Igual que ``load_product_catalog`` pero con una sola lectura de disco por ruta resuelta.

    Args:
        input_path: Ruta al JSON.

    Returns:
        Lista de productos.
    """
    key = str(Path(input_path).resolve())
    if key not in _CATALOG_CACHE:
        _CATALOG_CACHE[key] = load_product_catalog(key)
    return list(_CATALOG_CACHE[key])


class ProductCatalog:
    """Lee un archivo JSON de productos y expone consultas de alto nivel."""

    def __init__(self, catalog_path: Path) -> None:
        """
        Args:
            catalog_path: Ruta al JSON del portafolio (lista de objetos o dict con clave 'products').
        """
        self._path = Path(catalog_path)
        self._data: Optional[Dict[str, Any]] = None
        self._products: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Carga y normaliza el contenido del JSON en memoria."""
        self._products = load_product_catalog(str(self._path))
        self._data = {"products": self._products} if self._products else {}

    @property
    def products(self) -> List[Dict[str, Any]]:
        """Lista de productos cargados (vacía si no hay archivo o estructura desconocida)."""
        return list(self._products)

    def find_by_name_substring(self, substring: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Filtra productos cuyo campo 'name' o 'nombre' contiene la subcadena (case insensitive).

        Args:
            substring: Texto a buscar.
            limit: Máximo de resultados.

        Returns:
            Lista de dicts de producto.
        """
        if not self._products:
            self.load()
        sub = substring.lower()
        out: List[Dict[str, Any]] = []
        for p in self._products:
            name = str(p.get("name") or p.get("nombre") or "")
            if sub in name.lower():
                out.append(p)
            if len(out) >= limit:
                break
        return out
