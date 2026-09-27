"""
tool_normalizacion.py
Tool de normalizacion terminologica (SNOMED CT via BioPortal) -- M3, Luis

Dado un nombre de enfermedad en texto libre (salida del encoder de M1/M2),
intenta normalizarlo contra SNOMED CT. Secretos via .env (python-dotenv),
con fallback a Colab Secrets si no esta en .env.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def _get_api_key() -> str | None:
    key = os.environ.get("BIOPORTAL_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata
        return userdata.get("BIOPORTAL_API_KEY")
    except Exception:
        return None


BIOPORTAL_API_KEY = _get_api_key()
BIOPORTAL_SEARCH_URL = "https://data.bioontology.org/search"
MOCK_MODE = BIOPORTAL_API_KEY is None

if MOCK_MODE:
    print(
        "[tool_normalizacion] BIOPORTAL_API_KEY no encontrada "
        "(.env ni Colab Secrets) -- corriendo en MOCK_MODE."
    )


def normalizar_entidad(entidad: str, ontologia: str = "SNOMEDCT") -> dict:
    """
    Normaliza una entidad clinica contra SNOMED CT via BioPortal.

    entidad   : texto crudo del encoder, ej. "diabetes tipo 2"
    ontologia : ontologia BioPortal a consultar (por defecto SNOMEDCT)

    Devuelve:
    {
        "entidad_original"    : str,
        "entidad_normalizada" : str,
        "source_terminology"  : str | None,
        "normalization_failed": bool
    }
    """
    if MOCK_MODE:
        return _normalizar_mock(entidad)
    return _normalizar_real(entidad, ontologia)


def _normalizar_real(entidad: str, ontologia: str) -> dict:
    params = {
        "q": entidad,
        "apikey": BIOPORTAL_API_KEY,
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

    return {
        "entidad_original": entidad,
        "entidad_normalizada": resultados[0]["prefLabel"],
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

    resultados = [normalizar_entidad(e, ontologia) for e in entidades_prueba]
    n_fallidas = sum(1 for r in resultados if r["normalization_failed"])

    return {
        "modo": "mock" if MOCK_MODE else "real",
        "ontologia": ontologia,
        "n_evaluadas": len(resultados),
        "n_normalizacion_fallida": n_fallidas,
        "resultados": resultados,
    }


if __name__ == "__main__":
    print(normalizar_entidad("diabetes tipo 2"))
    print(normalizar_entidad("una enfermedad rarisima inventada"))
