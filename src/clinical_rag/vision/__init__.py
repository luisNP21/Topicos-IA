"""Componente de visión: imagen -> texto para el pipeline clínico."""

from clinical_rag.vision.integration import crear_responder_m3, sistema_multimodal
from clinical_rag.vision.transcription import transcribir, ver

__all__ = ["ver", "transcribir", "sistema_multimodal", "crear_responder_m3"]
