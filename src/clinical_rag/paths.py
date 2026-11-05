"""Rutas del proyecto.

`PROJECT_ROOT` permite apuntar los datos y artefactos a otra ubicación
(por ejemplo, Google Drive en Colab). Si no está definido, se usa la raíz
del repositorio.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DOCS_DIR = REPO_ROOT / "docs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def project_root() -> Path:
    """Raíz de datos/artefactos. Por defecto, la raíz del repositorio."""
    return Path(os.environ.get("PROJECT_ROOT", REPO_ROOT)).expanduser().resolve()


def resolve_project(path: str | Path) -> Path:
    """Resuelve una ruta relativa contra PROJECT_ROOT."""
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (project_root() / p).resolve()


def resolve_repo(path: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def resolve_config(path: str | Path) -> Path:
    """Resuelve un YAML: absoluto, relativo al repo o dentro de configs/."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    candidate = REPO_ROOT / p
    if candidate.exists():
        return candidate
    return CONFIG_DIR / p


def as_project_path(path: str | Path) -> str:
    """Devuelve una ruta relativa a PROJECT_ROOT cuando sea posible."""
    p = resolve_project(path)
    try:
        return str(p.relative_to(project_root()))
    except ValueError:
        return str(p)
