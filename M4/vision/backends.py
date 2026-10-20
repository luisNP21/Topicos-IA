"""Backends de transcripción para M4 (Agustín).

Tres implementaciones reales y una falsa para pruebas:

- `BackendVLMQwen`: VLM Qwen2-VL-2B-Instruct (patrón *input* de S11/S13).
- `BackendOCRDocling`: OCR/parseo con Docling (la herramienta que ya usa M3
  para el corpus y que recomendó Isabella).
- `BackendOCRTesseract`: OCR clásico de respaldo, útil cuando Docling no está
  disponible o para comparar motores.
- `BackendFake`: texto fijo, sin GPU ni dependencias; solo para tests.

Los imports pesados son perezosos: importar este archivo no descarga modelos.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image

from prompts import INSTRUCCION_TRANSCRIPCION, SYSTEM_VISION


class BackendError(RuntimeError):
    """Error explícito de un backend (dependencia ausente, binario, etc.)."""


class BackendVLMQwen:
    """Transcribe una imagen con Qwen2-VL-2B-Instruct (S11/S13)."""

    nombre = "vlm"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self._model = None
        self._proc = None
        self._torch = None

    def _cargar(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - depende del entorno
            raise BackendError(
                "El backend VLM requiere torch y transformers>=4.45. "
                "Instalar con: pip install -r M4/vision/requirements.txt"
            ) from exc

        model_id = self.cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")
        device = self.cfg.get("device", "auto")
        dtype_name = self.cfg.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_name, torch.float16)
        # En CPU, float16 no es seguro para todas las operaciones del VLM.
        if dtype_name == "float16" and not torch.cuda.is_available():
            dtype = torch.float32

        kwargs = {"torch_dtype": dtype}
        if device == "auto":
            # En Colab T4: reparte el modelo entre GPU/CPU si hace falta.
            kwargs["device_map"] = "auto"

        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if device not in ("auto",):
                self._model = self._model.to(device)
            self._model.eval()
            self._proc = AutoProcessor.from_pretrained(model_id)
        except Exception as exc:  # pragma: no cover - descarga/GPU
            raise BackendError(
                f"No se pudo cargar el VLM '{model_id}'. "
                "Verificar GPU, memoria y conexión a Hugging Face."
            ) from exc
        self._torch = torch

    def transcribir(
        self,
        imagen: Image.Image,
        *,
        instruccion: str | None = None,
        ruta: str | None = None,
    ) -> str:
        if self._model is None:
            self._cargar()

        torch = self._torch
        instruccion = instruccion or self.cfg.get("instruccion") or INSTRUCCION_TRANSCRIPCION
        system = self.cfg.get("system") or SYSTEM_VISION
        max_new_tokens = int(self.cfg.get("max_new_tokens", 512))

        msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": imagen},
                    {"type": "text", "text": instruccion},
                ],
            },
        ]

        texto_prompt = self._proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        # S11 usa qwen-vl-utils; si no está, S13 pasa la PIL directo.
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        if process_vision_info is not None:
            im_in, vid_in = process_vision_info(msgs)
            entradas = self._proc(
                text=[texto_prompt],
                images=im_in,
                videos=vid_in,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)
        else:
            entradas = self._proc(
                text=[texto_prompt],
                images=[imagen],
                return_tensors="pt",
            ).to(self._model.device)

        with torch.no_grad():
            salida = self._model.generate(
                **entradas,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # reproducible: sin muestreo
            )

        nuevos = salida[:, entradas["input_ids"].shape[1] :]
        return self._proc.batch_decode(nuevos, skip_special_tokens=True)[0].strip()


class BackendOCRDocling:
    """OCR/parseo con Docling (recomendación de Isabella, ya usado en M3)."""

    nombre = "ocr_docling"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self._converter = None

    def _cargar(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:  # pragma: no cover - depende del entorno
            raise BackendError(
                "Docling no está instalado. Instalar con: "
                "pip install 'docling>=2.0.0'. Si el entorno no lo soporta, "
                "usar --backend ocr_tesseract."
            ) from exc

        # Docling usa EasyOCR/Tesseract por debajo. Configuramos español de
        # forma best-effort: la API cambia entre versiones 2.x, así que si algo
        # no existe se conserva el pipeline por defecto.
        self._converter = DocumentConverter()
        if self.cfg.get("configurar_ocr", True):
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    EasyOcrOptions,
                    PdfPipelineOptions,
                )
                from docling.document_converter import PdfFormatOption

                opciones = PdfPipelineOptions()
                opciones.do_ocr = True
                opciones.ocr_options = EasyOcrOptions(
                    lang=[str(self.cfg.get("lang", "es"))]
                )

                format_options = {}
                try:
                    from docling.document_converter import ImageFormatOption

                    format_options[InputFormat.IMAGE] = ImageFormatOption(
                        pipeline_options=opciones
                    )
                except Exception:
                    format_options[InputFormat.IMAGE] = PdfFormatOption(
                        pipeline_options=opciones
                    )
                self._converter = DocumentConverter(format_options=format_options)
            except Exception:
                # Versión de Docling con otra API: se usa el pipeline por defecto.
                self._converter = DocumentConverter()

    def transcribir(
        self,
        imagen: Image.Image,
        *,
        instruccion: str | None = None,
        ruta: str | None = None,
    ) -> str:
        # Docling consume una ruta, no una PIL.Image. Si el caso trae ruta local,
        # se usa; si no, se guarda una copia temporal y se borra al terminar.
        if self._converter is None:
            self._cargar()

        temporal = None
        if ruta and Path(ruta).exists():
            path = Path(ruta)
        else:
            fd, nombre = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            imagen.save(nombre, format="PNG")
            path = Path(nombre)
            temporal = nombre

        try:
            try:
                resultado = self._converter.convert(str(path))
            except Exception as exc:
                raise BackendError(
                    "Docling no pudo procesar la imagen. Verificar que el OCR esté "
                    "activo (Docling reciente lo incluye) o usar --backend "
                    f"ocr_tesseract. Detalle: {exc}"
                ) from exc
            doc = resultado.document
            export = str(self.cfg.get("export", "markdown")).lower()
            if export == "text" and hasattr(doc, "export_to_text"):
                return doc.export_to_text()
            return doc.export_to_markdown()
        finally:
            if temporal is not None:
                try:
                    os.unlink(temporal)
                except OSError:
                    pass


class BackendOCRTesseract:
    """OCR clásico con Tesseract como respaldo/comparación."""

    nombre = "ocr_tesseract"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def transcribir(
        self,
        imagen: Image.Image,
        *,
        instruccion: str | None = None,
        ruta: str | None = None,
    ) -> str:
        try:
            import pytesseract
        except Exception as exc:  # pragma: no cover - depende del entorno
            raise BackendError(
                "pytesseract no está instalado. Instalar con: pip install pytesseract "
                "y el binario tesseract-ocr (+ tesseract-ocr-spa en Colab/Linux)."
            ) from exc

        lang = self.cfg.get("lang", "spa+eng")
        psm = int(self.cfg.get("psm", 6))
        oem = int(self.cfg.get("oem", 3))
        config = f"--psm {psm} --oem {oem}"
        try:
            return pytesseract.image_to_string(imagen, lang=lang, config=config)
        except Exception as exc:  # pragma: no cover - binario ausente
            if "tesseract" in str(exc).lower() or "TesseractNotFound" in type(exc).__name__:
                raise BackendError(
                    "El binario de Tesseract no está disponible. En Colab/Linux: "
                    "apt-get install -y tesseract-ocr tesseract-ocr-spa"
                ) from exc
            raise


class BackendFake:
    """Backend determinista para tests: no carga modelos ni usa red."""

    nombre = "fake"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.texto = self.cfg.get(
            "texto",
            "NOTA CLÍNICA SINTÉTICA. Paciente con DM2 e HTA. "
            "Antecedente de EPOC. Glucemia 142 mg/dL.",
        )

    def transcribir(
        self,
        imagen: Image.Image,
        *,
        instruccion: str | None = None,
        ruta: str | None = None,
    ) -> str:
        return self.texto


def crear_backend(nombre: str | None = None, cfg: dict | None = None):
    """Fábrica de backends. `cfg` es el YAML completo de M4/vision."""
    cfg = cfg or {}
    elegido = (nombre or cfg.get("vision", {}).get("backend") or "vlm").lower()

    if elegido in ("vlm", "qwen2-vl", "qwen2_vl", "qwen"):
        return BackendVLMQwen(cfg.get("vlm", {}))
    if elegido in ("ocr_docling", "docling", "ocr"):
        return BackendOCRDocling(cfg.get("ocr", {}).get("docling", {}))
    if elegido in ("ocr_tesseract", "tesseract"):
        return BackendOCRTesseract(cfg.get("ocr", {}).get("tesseract", {}))
    if elegido == "fake":
        return BackendFake(cfg.get("fake", {}))

    raise ValueError(
        f"Backend de visión desconocido: '{elegido}'. "
        "Use vlm | ocr_docling | ocr_tesseract | auto | fake."
    )
