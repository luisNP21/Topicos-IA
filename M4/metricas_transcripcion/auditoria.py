"""Auditoria de entidades (tarea 5a).

Dos preguntas distintas:

1. Lectura (no depende del NER): de las entidades del gold que se ven en la imagen,
   cuantas aparecen escritas tal cual en la transcripcion. Mide si la vision conservo el
   texto clinico, aunque luego el NER falle.

2. Origen de cada entidad predicha (depende del NER):
   - acierto: esta en el gold.
   - fp_ner: no esta en el gold pero si en el texto original. Es un error del NER, que
     tambien puede pasar leyendo texto limpio (ver `en_sistema_texto`).
   - texto_extra: no esta en el original pero si en un sello, nota o tachadura de la imagen.
     La vision lo leyo bien; no es alucinacion.
   - mal_leida: no esta en el original, pero se parece a un fragmento del original
     (p. ej. "diabetis" por "diabetes"). Error de lectura.
   - inventada: no esta en el original ni se parece a nada de el. Candidata a alucinacion.
   - fuera_de_transcripcion: tampoco esta en la transcripcion (inconsistencia del cache NER).

`mal_leida` e `inventada` salen de un umbral de parecido. Son las candidatas a revisar con
el juez visual de M4/vision/juez.py, que mira la imagen.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache

from rapidfuzz import fuzz

from cer_wer import normalizar

CATEGORIAS = ["acierto", "fp_ner", "texto_extra", "mal_leida", "inventada", "fuera_de_transcripcion"]
_NORM_TEXTO = {"unicode": "NFC", "minusculas": True, "colapsar_espacios": True, "quitar_puntuacion": False}


def normalizar_entidad(e: str) -> str:
    """Igual que M2/common.normalizar_entidad, y ademas colapsa espacios internos."""
    e = normalizar(e, _NORM_TEXTO)
    return e.strip().strip(string.punctuation + " ")


def normalizar_texto(t: str) -> str:
    return normalizar(t, _NORM_TEXTO)


@lru_cache(maxsize=20000)
def _patron(entidad: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(entidad) + r"(?!\w)")


def aparece(entidad: str, texto: str) -> bool:
    """La entidad aparece como cadena completa (sin cortar palabras) en el texto."""
    return bool(entidad) and _patron(entidad).search(texto) is not None


def parecido(entidad: str, texto: str) -> tuple[float, str]:
    """Mejor parecido (0-100) de la entidad con algun fragmento del texto, y ese fragmento."""
    if not entidad or not texto:
        return 0.0, ""
    al = fuzz.partial_ratio_alignment(entidad, texto)
    if al is None:
        return 0.0, ""
    return float(al.score), texto[al.dest_start:al.dest_end]


def lectura_entidades(gold_legibles: list[str], transcripcion_n: str, cfg: dict) -> dict:
    exactas, aproximadas, perdidas = 0, 0, []
    for e in gold_legibles:
        if aparece(e, transcripcion_n):
            exactas += 1
            continue
        score = parecido(e, transcripcion_n)[0] if len(e) >= cfg["min_caracteres_parecido"] else 0.0
        if score >= cfg["umbral_mal_leida"]:
            aproximadas += 1
        else:
            perdidas.append(e)
    return {"legibles": len(gold_legibles), "exactas": exactas, "aproximadas": aproximadas,
            "perdidas": len(perdidas), "lista_perdidas": perdidas}


def clasificar_predichas(predichas: list[str], gold_n: set[str], original_n: str, extra_n: str,
                         transcripcion_n: str, ner_texto_n: set[str] | None, cfg: dict) -> list[dict]:
    filas = []
    for e in sorted({normalizar_entidad(p) for p in predichas} - {""}):
        fila = {"entidad": e, "parecido": None, "fragmento_original": None,
                "en_sistema_texto": (e in ner_texto_n) if ner_texto_n is not None else None}
        if e in gold_n:
            fila["categoria"] = "acierto"
        elif aparece(e, original_n):
            fila["categoria"] = "fp_ner"
        elif extra_n and aparece(e, extra_n):
            fila["categoria"] = "texto_extra"
        elif not aparece(e, transcripcion_n):
            fila["categoria"] = "fuera_de_transcripcion"
        else:
            score, frag = parecido(e, original_n) if len(e) >= cfg["min_caracteres_parecido"] else (0.0, "")
            fila.update(parecido=round(score, 1), fragmento_original=frag or None)
            fila["categoria"] = "mal_leida" if score >= cfg["umbral_mal_leida"] else "inventada"
        filas.append(fila)
    return filas
