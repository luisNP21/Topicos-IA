"""Carga del manifiesto de casos con imagen (M4 · Agustín).

El eval set con imágenes lo construyen Luis (set sintético) y Pau (casos límite
reales). Este módulo solo define cómo lo consume la visión: un CSV o JSON con
una fila por caso y una columna `imagen`. No genera casos ni imágenes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from contratos import CasoVision


def _parsear_esperado(valor):
    """En CSV, `esperado` puede venir como JSON serializado."""
    if not isinstance(valor, str):
        return valor
    texto = valor.strip()
    if texto.startswith(("{", "[")):
        try:
            return json.loads(texto)
        except json.JSONDecodeError:
            return valor
    return valor


def _normalizar_caso(raw: dict, indice: int) -> CasoVision:
    doc_id = raw.get("id") or raw.get("doc_id") or raw.get("caso") or f"caso_{indice:02d}"
    imagen = (
        raw.get("imagen")
        or raw.get("image")
        or raw.get("imagen_path")
        or raw.get("path")
    )
    if not imagen:
        raise ValueError(
            f"El caso {doc_id!r} no tiene columna de imagen "
            "(se aceptan: imagen, image, imagen_path, path)"
        )

    caso: CasoVision = {
        "id": str(doc_id),
        "pregunta": str(raw.get("pregunta") or raw.get("question") or ""),
        "imagen": str(imagen),
        "entidad": raw.get("entidad") or raw.get("entity") or None,
        "esperado": _parsear_esperado(raw.get("esperado") or raw.get("ground_truth")),
        "criterio": raw.get("criterio") or None,
        "tipo": raw.get("tipo") or raw.get("tipo_caso") or "normal",
    }
    return caso


def cargar_manifiesto(path: str | Path) -> list[CasoVision]:
    """Carga un CSV o un JSON (lista de objetos) con los casos de imagen."""
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {ruta}")

    if ruta.suffix.lower() == ".csv":
        with open(ruta, "r", encoding="utf-8-sig", newline="") as f:
            filas = list(csv.DictReader(f))
    elif ruta.suffix.lower() in (".json", ".jsonl"):
        if ruta.suffix.lower() == ".jsonl":
            with open(ruta, "r", encoding="utf-8") as f:
                filas = [json.loads(line) for line in f if line.strip()]
        else:
            with open(ruta, "r", encoding="utf-8") as f:
                filas = json.load(f)
        if isinstance(filas, dict):
            filas = filas.get("casos") or filas.get("results") or []
    else:
        raise ValueError(
            f"Formato de manifiesto no soportado: {ruta.suffix}. Use .csv o .json/.jsonl"
        )

    if not isinstance(filas, list) or not filas:
        raise ValueError(f"El manifiesto está vacío o no es una lista: {ruta}")

    casos = [_normalizar_caso(fila, i) for i, fila in enumerate(filas, 1)]
    ids = [c["id"] for c in casos]
    repetidos = sorted({x for x in ids if ids.count(x) > 1})
    if repetidos:
        raise ValueError(f"doc_id repetidos en el manifiesto: {repetidos}")
    return casos


def resolver_ruta_imagen(
    caso: CasoVision,
    imagenes_dir: str | Path | None,
    project_root: str | Path,
) -> str:
    """Resuelve la ruta de la imagen sin descargarla.

    Prioridad: URL -> ruta absoluta -> `imagenes_dir` -> relativa a PROJECT_ROOT.
    """
    imagen = str(caso["imagen"])
    if imagen.startswith(("http://", "https://")):
        return imagen

    ruta = Path(imagen)
    if ruta.is_absolute():
        return str(ruta)
    if imagenes_dir:
        return str(Path(imagenes_dir) / ruta)
    return str(Path(project_root) / ruta)


def caso_desde_imagen(
    imagen: str | Path,
    pregunta: str = "",
    doc_id: str = "imagen_01",
    entidad: str | None = None,
) -> CasoVision:
    """Caso único para el CLI (`--imagen`), sin construir un eval set."""
    return {
        "id": doc_id,
        "pregunta": pregunta or "Transcribe la nota clínica de la imagen.",
        "imagen": str(imagen),
        "entidad": entidad,
        "esperado": None,
        "criterio": None,
        "tipo": "normal",
    }
