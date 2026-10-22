"""Transcripción de imágenes a texto (M4 · Agustín).

Expone la función `ver()` del laboratorio S13 y una variante con metadatos
(`transcribir`) que alimenta el cache y el scorecard.

La salida siempre es texto plano listo para el NER de M1/M2. Si el backend no
encuentra texto, devuelve la válvula de escape literal y marca `valvula=True`.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

from backends import crear_backend
from contratos import Transcripcion
from prompts import (
    INSTRUCCION_TRANSCRIPCION,
    MENSAJE_SIN_TEXTO,
    es_valvula,
)


def cargar_imagen(src) -> Image.Image:
    """Carga una imagen local, una URL o una PIL.Image ya abierta."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")

    texto = str(src)
    if texto.startswith(("http://", "https://")):
        respuesta = requests.get(texto, timeout=30)
        respuesta.raise_for_status()
        return Image.open(io.BytesIO(respuesta.content)).convert("RGB")

    ruta = Path(texto).expanduser()
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")
    return Image.open(ruta).convert("RGB")


def sha256_imagen(imagen: Image.Image) -> str:
    """Huella estable del contenido visual (mismos píxeles -> mismo hash)."""
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def limpiar_salida(texto: str | None, instruccion: str | None = None) -> str:
    """Quita formato Markdown y ecos del prompt; conserva el contenido clínico."""
    if not texto:
        return ""

    limpio = str(texto)

    # Un VLM a veces repite la instrucción antes de transcribir. Se elimina la
    # primera línea si es el eco del prompt, sin tocar el resto de la nota.
    if instruccion:
        prefijo = instruccion.strip().lower()[:40]
        primera, _, resto = limpio.partition("\n")
        if primera.strip().lower().startswith(prefijo):
            if resto.strip():
                limpio = resto
            else:
                # El VLM pudo responder en una sola línea: quitar la instrucción
                # completa si aparece y conservar lo que venga después.
                idx = primera.lower().find(instruccion.strip().lower())
                limpio = (
                    primera[idx + len(instruccion.strip()) :].lstrip(" :-\t")
                    if idx != -1
                    else ""
                )
    limpio = re.sub(r"^\s*transcribe literalmente[^\n]*\n?", "", limpio, flags=re.IGNORECASE)

    # Bloques de código y marcas de formato de Docling/Markdown.
    limpio = re.sub(r"```(?:markdown|text|md)?", "", limpio, flags=re.IGNORECASE)
    limpio = re.sub(r"^\s*#{1,6}\s*", "", limpio, flags=re.MULTILINE)
    limpio = limpio.replace("**", "").replace("__", "").replace("`", "")
    limpio = limpio.replace("|", " ")

    # Espacios y saltos de línea razonables para el tokenizer del NER.
    limpio = re.sub(r"[ \t]+", " ", limpio)
    limpio = re.sub(r"\n\s*\n\s*\n+", "\n\n", limpio)
    return limpio.strip()


def transcribir(
    imagen: Image.Image,
    *,
    backend=None,
    backend_nombre: str | None = None,
    instruccion: str | None = None,
    ruta: str | None = None,
    cfg: dict | None = None,
    doc_id: str | None = None,
) -> Transcripcion:
    """Imagen -> Transcripcion. Nunca lanza por un fallo del backend: lo registra."""
    if backend is None:
        backend = crear_backend(backend_nombre, cfg)

    instruccion = instruccion or INSTRUCCION_TRANSCRIPCION
    inicio = time.perf_counter()
    error = None
    crudo = ""

    try:
        crudo = backend.transcribir(imagen, instruccion=instruccion, ruta=ruta)
    except Exception as exc:  # se reporta explícitamente; no se silencia
        error = f"{type(exc).__name__}: {exc}"

    texto = limpiar_salida(crudo, instruccion)
    valvula = es_valvula(texto)
    if valvula:
        texto = MENSAJE_SIN_TEXTO

    segundos = round(time.perf_counter() - inicio, 3)
    return {
        "doc_id": doc_id or "",
        "texto": texto,
        "backend": getattr(backend, "nombre", backend_nombre or "desconocido"),
        "segundos": segundos,
        "valvula": valvula,
        "error": error,
        "imagen": str(ruta or ""),
        "sha256": sha256_imagen(imagen),
    }


def ver(
    imagen: Image.Image,
    *,
    backend=None,
    backend_nombre: str | None = None,
    instruccion: str | None = None,
    ruta: str | None = None,
    cfg: dict | None = None,
) -> tuple[str, None]:
    """Interfaz del lab S13: imagen -> (texto, None).

    La transcripción no devuelve un JSON estructurado (a diferencia de la
    extracción de campos del lab); el segundo valor es None para respetar la
    firma `texto, json_o_None`. Los metadatos (backend, latencia, válvula,
    error, hash) están en `transcribir()`, que es la interfaz rica que usa
    `run_vision.py` y el scorecard.
    """
    t = transcribir(
        imagen,
        backend=backend,
        backend_nombre=backend_nombre,
        instruccion=instruccion,
        ruta=ruta,
        cfg=cfg,
    )
    return t["texto"], None


def transcribir_auto(
    imagen: Image.Image,
    cfg: dict,
    *,
    instruccion: str | None = None,
    ruta: str | None = None,
    doc_id: str | None = None,
) -> Transcripcion:
    """Prueba backends en orden hasta que uno funcione.

    Solo hace fallback ante un ERROR del backend (dependencia ausente, GPU,
    binario). Si un backend funciona pero no ve texto, esa es una respuesta
    válida (válvula) y NO se prueba otro motor.
    """
    orden = cfg.get("vision", {}).get(
        "fallback_order", ["vlm", "ocr_docling", "ocr_tesseract"]
    )
    intentados: list[str] = []
    inicio = time.perf_counter()

    for nombre in orden:
        intentados.append(nombre)
        try:
            backend = crear_backend(nombre, cfg)
        except Exception as exc:
            intentados[-1] = f"{nombre} ({type(exc).__name__})"
            continue

        t = transcribir(
            imagen,
            backend=backend,
            instruccion=instruccion,
            ruta=ruta,
            doc_id=doc_id,
        )
        if t["error"] is None:
            t["backend"] = backend.nombre
            t["segundos"] = round(time.perf_counter() - inicio, 3)
            t["meta"] = {"auto": True, "intentados": intentados}
            return t
        intentados[-1] = f"{nombre} ({t['error'][:80]})"

    total = round(time.perf_counter() - inicio, 3)
    return {
        "doc_id": doc_id or "",
        "texto": MENSAJE_SIN_TEXTO,
        "backend": "auto",
        "segundos": total,
        "valvula": True,
        "error": "Ningún backend de visión pudo transcribir la imagen: "
        + "; ".join(intentados),
        "imagen": str(ruta or ""),
        "sha256": sha256_imagen(imagen),
        "meta": {"auto": True, "intentados": intentados},
    }
