"""Carga de configuración YAML compartida por todos los componentes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_config


def load_yaml(path: str | Path) -> dict[str, Any]:
    ruta = resolve_config(path)
    with ruta.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"El YAML está vacío: {ruta}")
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un objeto: {ruta}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Alias en inglés."""
    return load_yaml(path)


def cargar_config(path: str | Path) -> dict[str, Any]:
    """Alias en español para compatibilidad con el código existente."""
    return load_yaml(path)


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    ruta = Path(path)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return ruta
