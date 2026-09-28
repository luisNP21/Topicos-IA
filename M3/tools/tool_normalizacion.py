"""
tool_normalizacion.py
Tool de normalizacion terminologica (SNOMED CT via BioPortal) -- M3

Dado un nombre de enfermedad en texto libre (salida del encoder de M1/M2),
intenta normalizarlo contra SNOMED CT. Secretos via .env (python-dotenv),
con fallback a Colab Secrets si no esta en .env.
"""

import os
import re
import unicodedata

import requests
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str | None:
    key = os.environ.get("BIOPORTAL_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata
        return userdata.get("BIOPORTAL_API_KEY")
    except Exception:
        return None


BIOPORTAL_SEARCH_URL = "https://data.bioontology.org/search"


def _tokens(texto: str) -> set:
    """Tokens significativos (minusculas, sin tildes, >2 letras) para comparar etiquetas."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                         if unicodedata.category(c) != "Mn")
    return {t for t in re.findall(r"\w+", sin_tildes) if len(t) > 2}


def _plausible(entidad: str, etiqueta: str) -> bool:
    """True si la etiqueta de BioPortal comparte algun token significativo con la entidad.

    Evita aceptar resultados sin relacion: para texto clinico en espanol BioPortal a
    veces devuelve etiquetas en ingles o basura (p. ej. 'asma bronquial' ->
    'Smooth muscle antibody'), que al reescribir la query arruinan el retrieval.
    """
    return bool(_tokens(entidad) & _tokens(etiqueta))


def normalizar_entidad(entidad: str, ontologia: str = "SNOMEDCT", api_key: str | None = None) -> dict:
    """
    Normaliza una entidad clinica contra SNOMED CT via BioPortal.

    entidad   : texto crudo del encoder, ej. "diabetes tipo 2"
    ontologia : ontologia BioPortal a consultar (por defecto SNOMEDCT)
    api_key   : clave opcional, si no se pasa busca en entorno o Colab Secrets

    Devuelve:
    {
        "entidad_original"    : str,
        "entidad_normalizada" : str,
        "source_terminology"  : str | None,
        "normalization_failed": bool
    }
    """
    key = api_key or get_api_key()
    if not key:
        return _normalizar_mock(entidad)
    return _normalizar_real(entidad, ontologia, api_key=key)


def _normalizar_real(entidad: str, ontologia: str, api_key: str) -> dict:
    params = {
        "q": entidad,
        "apikey": api_key,
        "ontologies": ontologia,
        "pagesize": 1,
    }
    try:
        resp = requests.get(BIOPORTAL_SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        resultados = resp.json().get("collection", [])
    except Exception as e:
        print(f"[tool_normalizacion] error consultando BioPortal: {e}")
        resultados = []

    if not resultados:
        return {
            "entidad_original": entidad,
            "entidad_normalizada": entidad,
            "source_terminology": None,
            "normalization_failed": True,
        }

    etiqueta = resultados[0]["prefLabel"]
    sin_aporte = etiqueta.strip().lower() == entidad.strip().lower()
    if not _plausible(entidad, etiqueta) or sin_aporte:
        # Resultado sin relacion con la entidad, o que la repite tal cual (no aporta):
        # fallo explicito (se mantiene la entidad original) en vez de reescribir la
        # query con un termino ajeno.
        motivo = "sin relacion" if not _plausible(entidad, etiqueta) else "sin aporte"
        print(f"[tool_normalizacion] descartado: '{entidad}' -> '{etiqueta}' ({motivo})")
        return {
            "entidad_original": entidad,
            "entidad_normalizada": entidad,
            "source_terminology": None,
            "normalization_failed": True,
        }

    return {
        "entidad_original": entidad,
        "entidad_normalizada": etiqueta,
        "source_terminology": f"{ontologia} (via BioPortal)",
        "normalization_failed": False,
    }


def _normalizar_mock(entidad: str) -> dict:
    """Respuestas fijas para correr sin API key -- solo para smoke-test."""
    diccionario_falso = {
        "diabetes tipo 2": "Diabetes mellitus, tipo 2",
        "hipertension": "Hipertension esencial",
    }
    normalizada = diccionario_falso.get(entidad.lower())
    if normalizada is None:
        return {
            "entidad_original": entidad,
            "entidad_normalizada": entidad,
            "source_terminology": None,
            "normalization_failed": True,
        }
    return {
        "entidad_original": entidad,
        "entidad_normalizada": normalizada,
        "source_terminology": "SNOMED CT (mock)",
        "normalization_failed": False,
    }


def run(cfg: dict, project_root) -> dict:
    """
    Punto de entrada estandar del proyecto.
    Mismo contrato run(cfg, project_root) -> dict que usa M2/harness.

    cfg debe contener la clave 'tool_normalizacion' con:
        ontologia        : str
        entidades_prueba : list[str]
    """
    tool_cfg = cfg["tool_normalizacion"]
    entidades_prueba = tool_cfg.get("entidades_prueba", [])
    ontologia = tool_cfg["ontologia"]

    api_key = get_api_key()
    modo = "real" if api_key else "mock"
    if modo == "mock":
        print(
            "[tool_normalizacion] BIOPORTAL_API_KEY no encontrada "
            "(.env ni Colab Secrets) -- corriendo en MOCK_MODE."
        )
    else:
        print(f"[tool_normalizacion] BIOPORTAL_API_KEY detectada -- consultando BioPortal ({ontologia}).")

    resultados = [normalizar_entidad(e, ontologia, api_key=api_key) for e in entidades_prueba]
    n_fallidas = sum(1 for r in resultados if r["normalization_failed"])

    return {
        "modo": modo,
        "ontologia": ontologia,
        "n_evaluadas": len(resultados),
        "n_normalizacion_fallida": n_fallidas,
        "resultados": resultados,
    }


if __name__ == "__main__":
    print(normalizar_entidad("diabetes tipo 2"))
    print(normalizar_entidad("una enfermedad rarisima inventada"))
