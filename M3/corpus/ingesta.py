import yaml, json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import TableItem
import re


def cargar_fuentes(path_yaml: str) -> list[dict]:
    with open(path_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["fuentes"]


def ingerir_documento(fuente: dict, converter: DocumentConverter) -> dict:
    """PDF -> lista de secciones {titulo, texto}, preservando la jerarquía real del documento."""
    resultado = converter.convert(fuente["archivo_local"])
    doc = resultado.document
    secciones, actual = [], {"titulo": None, "texto": ""}
    for item, _ in doc.iterate_items():
        if item.label == "section_header":
            if actual["texto"]:
                secciones.append(actual)
            actual = {"titulo": item.text, "texto": ""}
        elif isinstance(item, TableItem):
            # las tablas no tienen .text — se exportan a markdown, preservando filas/columnas
            actual["texto"] += item.export_to_markdown(doc) + "\n"
        elif hasattr(item, "text"):
            # red de seguridad: cualquier item de texto no contemplado explícitamente (ej. captions)
            actual["texto"] += item.text + "\n"
        # items sin .text y que no son tabla (ej. imágenes) se ignoran silenciosamente
    if actual["texto"]:
        secciones.append(actual)
    return {"doc_id": fuente["doc_id"], "secciones": secciones}


def es_front_matter(seccion: dict, patron: re.Pattern, min_palabras: int) -> bool:
    titulo = (seccion["titulo"] or "").strip()
    if patron.match(titulo):
        return True
    if len(seccion["texto"].split()) < min_palabras:
        return True
    return False


def filtrar_secciones(secciones: list[dict], patron: re.Pattern, min_palabras: int) -> list[dict]:
    return [s for s in secciones if not es_front_matter(s, patron, min_palabras)]