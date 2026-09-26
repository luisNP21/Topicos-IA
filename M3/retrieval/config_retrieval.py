"""
Lectura de config_retrieval.yaml: rutas, modelos, parámetros del retrieval, la intención y la
orquestación. Las rutas relativas se resuelven desde el directorio de trabajo (la raíz del
repositorio). La ruta del YAML puede darse como argumento o con M3_RETRIEVAL_CONFIG.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

import retrieval

RUTA_POR_DEFECTO = Path(__file__).with_name("config_retrieval.yaml")


def cargar_config(ruta: str | Path | None = None) -> dict:
    ruta = Path(ruta or os.environ.get("M3_RETRIEVAL_CONFIG") or RUTA_POR_DEFECTO)
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if cfg["corpus"] not in cfg["rutas"]:
        raise ValueError(f"corpus debe ser uno de {list(cfg['rutas'])}")
    retrieval.configurar_intencion(cfg["intencion"]["plantillas"], cfg["intencion"]["pregunta"])
    return cfg


def rutas(cfg: dict) -> dict:
    return cfg["rutas"][cfg["corpus"]]


def indice_desde_config(cfg: dict, **kw) -> retrieval.IndiceRAG:
    r, m, rt = rutas(cfg), cfg["modelos"], cfg["retrieval"]
    kw.setdefault("modelo_reranker", m["reranker"])
    return retrieval.construir_indice(r["chroma_dir"], r["manifest"], modelo_embeddings=m["embeddings"],
                                      n_candidatos=rt["n_candidatos"], krrf=rt["krrf"],
                                      enriquecer=rt["encabezado_contextual"], **kw)


def parametros_orquestacion(cfg: dict) -> dict:
    """Parámetros para resolver_query. Si el YAML no fija los umbrales, se usan los calibrados
    por el experimento (<salida>/umbrales.json)."""
    o = cfg["orquestacion"]
    umbral, umbral_evidencia = o["umbral"], o["umbral_evidencia"]
    if umbral is None:
        archivo = Path(rutas(cfg)["salida"]) / "umbrales.json"
        if not archivo.exists():
            raise FileNotFoundError(f"No hay umbrales en el YAML ni calibrados en {archivo}; corra el experimento.")
        final = json.loads(archivo.read_text(encoding="utf-8"))["final"]
        umbral, umbral_evidencia = final["umbral"], final["umbral_evidencia"]
    return {"umbral": umbral, "umbral_evidencia": umbral_evidencia,
            "forzar_por_sigla": o["forzar_por_sigla"], "k": cfg["retrieval"]["k"]}


def configurar_pipeline(cfg: dict) -> retrieval.IndiceRAG:
    """Construye el índice y lo deja listo para retrieve_naive y retrieve_advanced."""
    indice = indice_desde_config(cfg)
    retrieval.configurar(indice, cfg["retrieval"]["modo_pipeline"])
    return indice
