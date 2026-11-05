"""Utilidades de configuración específicas del corpus."""
from __future__ import annotations

import re

from clinical_rag.config import cargar_config, load_yaml

__all__ = ["cargar_config", "load_yaml", "construir_patron_front_matter", "construir_patrones_categoria"]


def construir_patron_front_matter(titulos_excluidos: list[str]) -> re.Pattern:
    alternativas = "|".join(re.escape(t) for t in titulos_excluidos)
    return re.compile(rf"^({alternativas})\b", re.IGNORECASE)


def construir_patrones_categoria(categorias_seccion: dict) -> dict[str, re.Pattern]:
    return {
        categoria: re.compile("|".join(re.escape(p) for p in palabras), re.IGNORECASE)
        for categoria, palabras in categorias_seccion.items()
    }
