"""Normalizacion de texto y CER/WER (tarea 5a).

CER = distancia de edicion (Levenshtein) en caracteres / caracteres de la referencia.
WER = lo mismo en palabras. Ambos pueden superar 1 si la transcripcion tiene mucho texto extra.
La distancia la calcula rapidfuzz (C++). Las operaciones (sustitucion, borrado, insercion)
salen de una alineacion optima; hay varias posibles, asi que su reparto es indicativo y
solo la suma (la distancia) es exacta.
"""

from __future__ import annotations

import re
import string
import unicodedata

from rapidfuzz.distance import Levenshtein

PUNTUACION = string.punctuation + "¡¿«»“”‘’…–—·"
_RE_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str, cfg: dict) -> str:
    """Normalizacion para CER. Ver la seccion `normalizacion` de config.yaml."""
    t = unicodedata.normalize(cfg.get("unicode", "NFC"), texto or "")
    if cfg.get("minusculas", True):
        t = t.lower()
    if cfg.get("quitar_puntuacion", False):
        t = t.translate(str.maketrans({c: " " for c in PUNTUACION}))
    if cfg.get("colapsar_espacios", True):
        t = _RE_ESPACIOS.sub(" ", t)
    return t.strip()


def sin_diacriticos(texto: str) -> str:
    """Quita tildes, dieresis y la virgulilla de la enie (diagnostico, no la metrica principal)."""
    nfd = unicodedata.normalize("NFD", texto)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def palabras(texto_normalizado: str, cfg: dict) -> list[str]:
    toks = texto_normalizado.split()
    if cfg.get("wer_quitar_puntuacion_bordes", True):
        toks = [t.strip(PUNTUACION) for t in toks]
        toks = [t for t in toks if t]
    return toks


def _operaciones(ref, hip) -> dict:
    conteo = {"sustituciones": 0, "borrados": 0, "inserciones": 0}
    for op in Levenshtein.editops(ref, hip):
        if op.tag == "replace":
            conteo["sustituciones"] += 1
        elif op.tag == "delete":
            conteo["borrados"] += 1
        elif op.tag == "insert":
            conteo["inserciones"] += 1
    return conteo


def comparar(referencia: str, hipotesis: str, cfg: dict) -> dict:
    """Errores absolutos de una hipotesis contra una referencia (para sumar despues)."""
    ref = normalizar(referencia, cfg)
    hip = normalizar(hipotesis, cfg)
    ref_p, hip_p = palabras(ref, cfg), palabras(hip, cfg)
    ref_sd, hip_sd = sin_diacriticos(ref), sin_diacriticos(hip)
    return {
        "caracteres_ref": len(ref),
        "caracteres_hip": len(hip),
        "errores_caracter": Levenshtein.distance(ref, hip),
        "errores_caracter_sin_diacriticos": Levenshtein.distance(ref_sd, hip_sd),
        "palabras_ref": len(ref_p),
        "palabras_hip": len(hip_p),
        "errores_palabra": Levenshtein.distance(ref_p, hip_p),
        "ops_caracter": _operaciones(ref, hip),
        "ops_palabra": _operaciones(ref_p, hip_p),
    }


def tasas(c: dict) -> dict:
    """CER y WER a partir de conteos (de un documento o sumados sobre varios)."""
    def div(a, b):
        return round(a / b, 4) if b else None
    return {
        "cer": div(c["errores_caracter"], c["caracteres_ref"]),
        "cer_sin_diacriticos": div(c["errores_caracter_sin_diacriticos"], c["caracteres_ref"]),
        "wer": div(c["errores_palabra"], c["palabras_ref"]),
    }


def sumar(conteos: list[dict]) -> dict:
    total = {"caracteres_ref": 0, "caracteres_hip": 0, "errores_caracter": 0,
             "errores_caracter_sin_diacriticos": 0, "palabras_ref": 0, "palabras_hip": 0,
             "errores_palabra": 0,
             "ops_caracter": {"sustituciones": 0, "borrados": 0, "inserciones": 0},
             "ops_palabra": {"sustituciones": 0, "borrados": 0, "inserciones": 0}}
    for c in conteos:
        for k, v in c.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    total[k][kk] += vv
            else:
                total[k] += v
    return total
