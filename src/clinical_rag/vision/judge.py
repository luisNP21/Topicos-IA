"""Juez visual: apoyo de Agustín a la Tarea 5a de Luis.

La métrica completa (CER/WER y tasa de entidades inventadas) es de Luis. Aquí
queda SOLO el primitivo que mira la imagen y decide si una afirmación se
sostiene en ella. Luis lo integra cuando tenga las transcripciones cacheadas.

No se inventan números: esta función devuelve 1.0/0.0 por caso y el dueño de
la métrica decide cómo agregarlos.
"""

from __future__ import annotations


def juez_fidelidad_visual(
    afirmacion: str,
    imagen,
    preguntar_fn,
    *,
    max_new_tokens: int = 3,
) -> float:
    """1.0 si el VLM considera que la afirmación se sostiene en la imagen.

    `preguntar_fn(imagen, prompt, max_new_tokens=...) -> str` es la misma
    función del backend VLM (o un mock en tests).
    """
    prompt = (
        f'Afirmación: "{str(afirmacion)[:600]}". '
        "¿Se sostiene en la imagen? Responde solo SI o NO."
    )
    veredicto = preguntar_fn(imagen, prompt, max_new_tokens=max_new_tokens)
    return float(str(veredicto).strip().upper().startswith("S"))
