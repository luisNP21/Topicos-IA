"""Prompts y constantes de la transcripción visual (M4, Agustín).

Se separan de los backends para que OCR y VLM compartan el mismo contrato de
salida y la misma válvula de escape. El texto de la válvula es literal: es lo
que el harness y el juez visual pueden detectar sin ambigüedad.
"""

from __future__ import annotations

import re

# Válvula de escape: misma idea que S07/S10/S11. Si no hay texto legible, el
# componente NO inventa una nota clínica.
MENSAJE_SIN_TEXTO = "No se aprecia texto legible en la imagen."

# El VLM describe/transcribe; NO diagnostica. Es la regla ética de S13.
SYSTEM_VISION = (
    "Eres un transcriptor clínico. Transcribes literalmente lo que se ve en la "
    "imagen. No diagnosticas, no interpretas, no resumas y no inventas datos."
)

INSTRUCCION_TRANSCRIPCION = (
    "Transcribe literalmente el texto visible en la imagen, en español. "
    "Conserva tildes, la letra ñ, siglas, abreviaturas, números, dosis y saltos "
    "de línea, en el mismo orden en que aparecen. No interpretes, no resumas, "
    "no corrijas y no agregues información que no esté en la imagen. "
    "Devuelve únicamente el texto transcrito, sin comentarios ni explicaciones. "
    f'Si la imagen no contiene texto legible, responde exactamente: "{MENSAJE_SIN_TEXTO}"'
)

# Solo se considera válvula si la respuesta EMPIEZA con una negativa de lectura.
# Así no se confunde con una nota real que mencione "no se aprecia..." dentro
# de una frase clínica.
PATRON_VALVULA = re.compile(
    r"^\s*(no se aprecia|no se puede leer|no hay texto|no puedo|"
    r"no es legible|ilegible|sin texto)",
    re.IGNORECASE,
)


def es_valvula(texto: str | None) -> bool:
    """True si la salida del backend equivale a 'no hay texto legible'."""
    if not texto or not texto.strip():
        return True
    return bool(PATRON_VALVULA.match(texto))
