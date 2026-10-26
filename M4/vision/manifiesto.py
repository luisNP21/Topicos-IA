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
        "nivel": (str(raw.get("nivel")).strip() if raw.get("nivel") else None),
        "grupo": (str(raw.get("grupo")).strip() if raw.get("grupo") else None),
        "doc_id_original": (
            str(raw.get("doc_id")).strip() if raw.get("doc_id") else None
        ),
    }
    return caso


def _leer_filas(ruta: Path) -> list[dict]:
    if ruta.suffix.lower() == ".csv":
        with open(ruta, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if ruta.suffix.lower() in (".json", ".jsonl"):
        if ruta.suffix.lower() == ".jsonl":
            with open(ruta, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(ruta, "r", encoding="utf-8") as f:
            filas = json.load(f)
        if isinstance(filas, dict):
            filas = filas.get("casos") or filas.get("results") or []
        return filas
    raise ValueError(
        f"Formato de manifiesto no soportado: {ruta.suffix}. Use .csv o .json/.jsonl"
    )


def _valores(filas: list[dict], campo: str) -> list[str]:
    valores = {str(f.get(campo)).strip() for f in filas if f.get(campo)}
    return sorted(v for v in valores if v)


def cargar_manifiesto(
    path: str | Path,
    nivel: str | None = None,
    grupo: str | None = None,
) -> list[CasoVision]:
    """Carga un CSV/JSON de casos de imagen, opcionalmente filtrado por nivel/grupo.

    El set sintético completo tiene un `id` por página (`<doc_id>__p<k>`) que se
    repite entre niveles de degradación. Para procesarlo hay que elegir UN nivel
    (`--nivel`) o usar un CSV por nivel; si no, el cache mezclaría imágenes de
    niveles distintos bajo el mismo doc_id.
    """
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {ruta}")

    filas = _leer_filas(ruta)
    if not isinstance(filas, list) or not filas:
        raise ValueError(f"El manifiesto está vacío o no es una lista: {ruta}")

    niveles = _valores(filas, "nivel")
    grupos = _valores(filas, "grupo")

    if nivel:
        filas = [f for f in filas if str(f.get("nivel", "")).strip() == nivel]
    if grupo:
        filas = [f for f in filas if str(f.get("grupo", "")).strip() == grupo]

    if not filas:
        detalle = []
        if nivel:
            detalle.append(f"nivel={nivel!r} (disponibles: {niveles or 'sin columna nivel'})")
        if grupo:
            detalle.append(f"grupo={grupo!r} (disponibles: {grupos or 'sin columna grupo'})")
        raise ValueError(
            "El filtro no dejó casos en el manifiesto: " + "; ".join(detalle)
        )

    casos = [_normalizar_caso(fila, i) for i, fila in enumerate(filas, 1)]
    ids = [c["id"] for c in casos]
    repetidos = sorted({x for x in ids if ids.count(x) > 1})
    if repetidos:
        pista = (
            f" Use --nivel/--grupo o un CSV por nivel. Niveles disponibles: {niveles}."
            if not nivel and not grupo and niveles
            else " Revisar el manifiesto: el filtro aplicado no dejó ids únicos."
        )
        raise ValueError(f"doc_id repetidos en el manifiesto: {repetidos}.{pista}")
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
