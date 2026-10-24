"""Contratos de la pieza de visión (M4 · Agustín).

Este archivo NO importa torch, transformers ni docling: solo declara la forma
de los datos y las interfaces. Eso permite probar el módulo con un backend
falso y permite que Isabella conecte el cache al harness sin cargar modelos.
"""

from __future__ import annotations

from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

from PIL import Image


class Transcripcion(TypedDict):
    """Salida de un backend de transcripción para un caso."""

    doc_id: str
    texto: str              # texto limpio que entra al NER / RAG
    backend: str            # "vlm" | "ocr_docling" | "ocr_tesseract" | "fake"
    segundos: float         # latencia medida de la llamada al backend
    valvula: bool           # True si el backend dijo "no se aprecia texto"
    error: str | None       # None si no hubo fallo; mensaje explícito si lo hubo
    imagen: str             # ruta o URL original (trazabilidad)
    sha256: str             # huella del contenido visual (reproducibilidad)
    meta: NotRequired[dict[str, Any]]


class CasoVision(TypedDict):
    """Un caso del manifiesto de imágenes (formato propio de M4)."""

    id: str
    pregunta: str
    imagen: str
    entidad: NotRequired[str | None]   # si el eval set ya trae la entidad gold
    esperado: NotRequired[Any]
    criterio: NotRequired[str | None]
    tipo: NotRequired[str | None]      # normal | adversarial | ...


class ResultadoCaso(TypedDict):
    """Resumen por caso que escribe `run(cfg, project_root)`."""

    id: str
    imagen: str
    backend: str
    segundos: float
    valvula: bool
    error: str | None
    sha256: str
    texto_preview: str


@runtime_checkable
class BackendTranscripcion(Protocol):
    """Interfaz mínima de un backend: imagen -> texto crudo."""

    nombre: str

    def transcribir(
        self,
        imagen: Image.Image,
        *,
        instruccion: str | None = None,
        ruta: str | None = None,
    ) -> str: ...


@runtime_checkable
class ResponderM3(Protocol):
    """Firma del sistema de M3 que consume el texto transcrito.

    Es la misma firma que usa el lab de S13 para que el enchufe sea directo.
    """

    def __call__(self, pregunta: str, contexto_extra: str = "") -> str: ...


@runtime_checkable
class ExtractorEntidades(Protocol):
    """Firma opcional para extraer entidades del texto transcrito (NER de M1/M2)."""

    def __call__(self, texto: str) -> list[str]: ...
