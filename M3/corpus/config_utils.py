# config_utils.py — nuevo módulo, sin dependencias de Docling/Chroma/embeddings
import re
import yaml


def cargar_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def construir_patron_front_matter(titulos_excluidos: list[str]) -> re.Pattern:
    alternativas = "|".join(re.escape(t) for t in titulos_excluidos)
    return re.compile(rf"^({alternativas})\b", re.IGNORECASE)


def construir_patrones_categoria(categorias_seccion: dict) -> dict[str, re.Pattern]:
    return {
        categoria: re.compile("|".join(re.escape(p) for p in palabras), re.IGNORECASE)
        for categoria, palabras in categorias_seccion.items()
    }