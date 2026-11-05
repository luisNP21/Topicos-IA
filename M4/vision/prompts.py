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
    r"no es legible|ilegible|sin texto|lo siento|"
    r"i cannot|i can'?t|i'?m sorry|there is no text|no (legible |readable )?text)",
    re.IGNORECASE,
)

# Respuestas que COMPLETAS son solo un comentario sobre la imagen, no una nota.
# Caso típico (revisión de M4): el VLM contesta «Se aprecia texto legible», es decir,
# pierde el "No" de la válvula o describe la imagen en vez de transcribirla. Con el
# patrón de arriba eso pasaba como si fuera la nota. Aquí se exige que sea TODA la
# respuesta, para no descartar una nota real que empiece con «Se aprecia edema…».
PATRON_SOLO_COMENTARIO = re.compile(
    r"^(?:(?:en )?la imagen )?(?:no )?"
    r"(?:se aprecia|se aprecian|se observa|se ve|se lee|hay|contiene|tiene|muestra)"
    r"(?: (?:ningún|ningun|algún|algun|un|el))? texto"
    r"(?: (?:legible|visible|escrito|ilegible|claro))?"
    r"(?: (?:en|de) (?:la|esta) (?:imagen|nota|foto))?"
    r"|(?:el )?texto (?:es )?(?:ilegible|no legible|no es legible)"
    r"|(?:imagen )?(?:en blanco|vacía|vacia|ilegible)"
    r"|(?:the image (?:contains|has|shows) )?(?:no )?(?:legible |readable |visible )?text"
    r"(?: (?:is )?(?:visible|legible))?(?: in the image)?",
    re.IGNORECASE,
)


def _normalizar_respuesta(texto: str) -> str:
    """Quita comillas, viñetas y puntuación de los bordes; colapsa espacios."""
    t = " ".join(texto.split())
    return t.strip(" \t\"'«»“”‘’*_-.:;!¡¿?()[]").strip()


def es_valvula(texto: str | None) -> bool:
    """True si la salida del backend equivale a 'no hay texto legible'.

    Tres casos: respuesta vacía; respuesta que EMPIEZA con una negativa de lectura
    («No se aprecia…», «Ilegible…»); o respuesta que ENTERA es solo un comentario
    sobre la imagen sin transcribir nada («Se aprecia texto legible»)."""
    if not texto or not texto.strip():
        return True
    normalizado = _normalizar_respuesta(texto)
    if not normalizado:
        return True
    return bool(PATRON_VALVULA.match(normalizado)
                or PATRON_SOLO_COMENTARIO.fullmatch(normalizado))
