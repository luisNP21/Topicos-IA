"""
Lectura de config_retrieval.yaml: rutas, modelos, parámetros del retrieval, la intención, la
orquestación y la herramienta de normalización.

Las rutas relativas se resuelven desde la raíz del repositorio (dos niveles sobre este archivo, o
M3_REPO_ROOT si está definida), no desde el directorio de trabajo; así funcionan igual desde el
notebook, desde un script o desde la generación. La ruta del YAML puede darse como argumento o
con M3_RETRIEVAL_CONFIG.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

import retrieval

RUTA_POR_DEFECTO = Path(__file__).with_name("config_retrieval.yaml")
RAIZ_REPO = Path(os.environ.get("M3_REPO_ROOT") or Path(__file__).resolve().parents[2])


def resolver(ruta: str | Path | None) -> str | None:
    if ruta is None:
        return None
    ruta = Path(ruta)
    return str(ruta if ruta.is_absolute() else RAIZ_REPO / ruta)


def cargar_config(ruta: str | Path | None = None) -> dict:
    ruta = Path(ruta or os.environ.get("M3_RETRIEVAL_CONFIG") or RUTA_POR_DEFECTO)
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if cfg["corpus"] not in cfg["rutas"]:
        raise ValueError(f"corpus debe ser uno de {list(cfg['rutas'])}")
    retrieval.configurar_intencion(cfg["intencion"]["plantillas"], cfg["intencion"]["pregunta"])
    return cfg


def rutas(cfg: dict) -> dict:
    """Rutas del corpus activo, ya resueltas a rutas absolutas."""
    return {k: resolver(v) if isinstance(v, str) else v for k, v in cfg["rutas"][cfg["corpus"]].items()}


def verificar_modelo_del_corpus(cfg: dict) -> None:
    """El embedder de las consultas debe ser el mismo con el que la ingesta construyó el índice."""
    archivo = resolver(cfg.get("corpus_pipeline", {}).get("config"))
    if not archivo or not Path(archivo).exists():
        return
    modelo_corpus = yaml.safe_load(Path(archivo).read_text(encoding="utf-8"))["embeddings"]["model_name"]
    if modelo_corpus != cfg["modelos"]["embeddings"]:
        raise RuntimeError(f"El corpus se indexó con {modelo_corpus} y el retrieval usa "
                           f"{cfg['modelos']['embeddings']}; deben coincidir.")


def preparar_indice(cfg: dict) -> str:
    """Si el perfil define chroma_origen (por ejemplo, el índice en Drive), lo copia a chroma_dir
    en el disco local la primera vez: Chroma usa SQLite y puede fallar sobre el disco de Drive."""
    r = rutas(cfg)
    origen, destino = r.get("chroma_origen"), r["chroma_dir"]
    if origen and not Path(destino).exists():
        if not Path(origen).exists():
            raise FileNotFoundError(f"No existe el índice en {origen}. ¿Está montado Drive y corrió la ingesta?")
        shutil.copytree(origen, destino)
    return destino


def indice_desde_config(cfg: dict, **kw) -> retrieval.IndiceRAG:
    verificar_modelo_del_corpus(cfg)
    r, m, rt = rutas(cfg), cfg["modelos"], cfg["retrieval"]
    kw.setdefault("modelo_reranker", m["reranker"])
    return retrieval.construir_indice(preparar_indice(cfg), r["manifest"], modelo_embeddings=m["embeddings"],
                                      n_candidatos=rt["n_candidatos"], krrf=rt["krrf"],
                                      enriquecer=rt["encabezado_contextual"], **kw)


def parametros_orquestacion(cfg: dict) -> dict:
    """Parámetros para resolver_query. Orden de prioridad de los umbrales: los fijados en el YAML,
    los calibrados por el experimento (<salida>/umbrales.json) y, si aún no hay calibración, los
    provisionales del YAML."""
    o = cfg["orquestacion"]
    umbral, umbral_evidencia = o["umbral"], o["umbral_evidencia"]
    if umbral is None:
        archivo = Path(rutas(cfg)["salida"]) / "umbrales.json"
        if archivo.exists():
            final = json.loads(archivo.read_text(encoding="utf-8"))["final"]
            umbral, umbral_evidencia = final["umbral"], final["umbral_evidencia"]
        elif o.get("umbrales_provisionales"):
            umbral = o["umbrales_provisionales"]["umbral"]
            umbral_evidencia = o["umbrales_provisionales"]["umbral_evidencia"]
            print(f"Aviso: no hay umbrales calibrados en {archivo}; se usan los provisionales del YAML.")
        else:
            raise FileNotFoundError(f"No hay umbrales en el YAML ni calibrados en {archivo}; corra el experimento.")
    return {"umbral": umbral, "umbral_evidencia": umbral_evidencia,
            "forzar_por_sigla": o["forzar_por_sigla"], "k": cfg["retrieval"]["k"]}


def normalizador_desde_config(cfg: dict):
    """Normalizador simulado si el perfil tiene mapa_normalizacion; si no, la herramienta real."""
    mapa_ruta = rutas(cfg).get("mapa_normalizacion")
    if mapa_ruta:
        sys.path.insert(0, resolver("M3/generacion"))
        from mocks import crear_normalizador_mock
        mapa = json.loads(Path(mapa_ruta).read_text(encoding="utf-8"))
        return crear_normalizador_mock({k.strip().lower(): v for k, v in mapa.items()})

    n = cfg["normalizacion"]
    sys.path.insert(0, resolver(n["directorio"]))
    from tool_normalizacion import normalizar_entidad
    ontologia = yaml.safe_load(Path(resolver(n["config"])).read_text(encoding="utf-8"))["tool_normalizacion"]["ontologia"]
    return lambda entidad: normalizar_entidad(entidad, ontologia=ontologia)


def configurar_pipeline(cfg: dict) -> retrieval.IndiceRAG:
    """Construye el índice y lo deja listo para retrieve_naive y retrieve_advanced."""
    indice = indice_desde_config(cfg)
    retrieval.configurar(indice, cfg["retrieval"]["modo_pipeline"])
    return indice
