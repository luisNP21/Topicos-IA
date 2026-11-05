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
from collections import Counter
from pathlib import Path

import yaml

from clinical_rag.config import load_yaml
from clinical_rag.paths import CONFIG_DIR, REPO_ROOT, resolve_config, resolve_repo
from clinical_rag.retrieval import index as retrieval

RUTA_POR_DEFECTO = CONFIG_DIR / "retrieval.yaml"
RAIZ_REPO = REPO_ROOT


def resolver(ruta: str | Path | None) -> str | None:
    if ruta is None:
        return None
    return str(resolve_repo(ruta))


def cargar_config(ruta: str | Path | None = None) -> dict:
    ruta = Path(ruta or os.environ.get("M3_RETRIEVAL_CONFIG") or RUTA_POR_DEFECTO)
    cfg = load_yaml(ruta)
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


class IndiceInconsistente(RuntimeError):
    """El índice Chroma no contiene exactamente los chunks JSON del corpus."""


def cargar_chunks_corpus(cfg: dict) -> list[dict]:
    """Chunks JSON del perfil activo: la fuente de verdad del índice y del gold set."""
    ruta = rutas(cfg)["corpus_json"]
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No existen los chunks en {ruta}. ¿Está montado Drive y corrió la ingesta?")
    chunks = retrieval.cargar_chunks_json(ruta)
    if not chunks:
        raise ValueError(f"No hay chunks en {ruta}.")
    invalidos = [c.get("chunk_id") for c in chunks if not {"chunk_id", "doc_id", "texto"} <= c.keys()]
    if invalidos:
        raise ValueError(f"Chunks sin chunk_id, doc_id o texto: {invalidos[:5]}")
    repetidos = sorted(k for k, n in Counter(c["chunk_id"] for c in chunks).items() if n > 1)
    if repetidos:
        raise ValueError(f"chunk_id repetidos en {ruta}: {repetidos[:5]}")
    return [{**c, "seccion": c.get("seccion")} for c in chunks]


def verificar_indice(coleccion, chunks: list[dict]) -> None:
    """Falla si el índice no tiene exactamente los mismos ids y textos que los chunks JSON."""
    todo = coleccion.get(include=["documents"])
    en_indice = dict(zip(todo["ids"], todo["documents"]))
    en_json = {c["chunk_id"]: c["texto"] for c in chunks}
    sobran, faltan = sorted(en_indice.keys() - en_json.keys()), sorted(en_json.keys() - en_indice.keys())
    distintos = sorted(k for k in en_indice.keys() & en_json.keys() if en_indice[k] != en_json[k])
    problemas = []
    if len(en_indice) != len(en_json):
        problemas.append(f"el índice tiene {len(en_indice)} chunks y la carpeta de chunks {len(en_json)}")
    if sobran:
        problemas.append(f"{len(sobran)} ids del índice no están en los JSON (p. ej. {sobran[:3]})")
    if faltan:
        problemas.append(f"{len(faltan)} chunks JSON no están en el índice (p. ej. {faltan[:3]})")
    if distintos:
        problemas.append(f"{len(distintos)} chunks tienen otro texto en el índice (p. ej. {distintos[:3]})")
    if problemas:
        raise IndiceInconsistente("El índice no coincide con los chunks JSON: " + "; ".join(problemas) + ".")


def _abrir_coleccion(persist_dir: str):
    import chromadb
    return chromadb.PersistentClient(path=persist_dir).get_collection(retrieval.COLECCION)


def _construir_con_ingesta(chunks: list[dict], persist_dir: str, modelo_embeddings: str) -> None:
    """Mismo código de la ingesta, sin Docling: embeddings y colección Chroma."""
    from clinical_rag.corpus.embeddings import cargar_modelo_embeddings, embeber_chunks
    from clinical_rag.corpus.store import construir_indice_chroma

    modelo = cargar_modelo_embeddings(modelo_embeddings)
    construir_indice_chroma(chunks, embeber_chunks(chunks, modelo), persist_dir=persist_dir)


def preparar_indice(cfg: dict, construir_fn=None, abrir_fn=None, forzar: bool = False) -> str:
    """Deja en chroma_dir (disco local) un índice con exactamente los chunks de corpus_json.

    Si ya existe y coincide con los JSON se reutiliza; si no, se borra y se reconstruye con el
    código de la ingesta. Nunca se copia un índice ya construido: puede arrastrar ids de ingestas
    anteriores o venir de otra versión de Chroma."""
    construir_fn = construir_fn or _construir_con_ingesta
    abrir_fn = abrir_fn or _abrir_coleccion
    destino = rutas(cfg)["chroma_dir"]
    if "/content/drive/" in Path(destino).as_posix() + "/":
        raise ValueError(f"chroma_dir ({destino}) debe estar en el disco local: el índice se borra y reconstruye.")
    chunks = cargar_chunks_corpus(cfg)
    if Path(destino).exists() and not forzar:
        try:
            verificar_indice(abrir_fn(destino), chunks)
            return destino
        except Exception as e:
            print(f"Aviso: se reconstruye el índice de {destino} ({e})")
    if Path(destino).exists():
        shutil.rmtree(destino)
    construir_fn(chunks, destino, cfg["modelos"]["embeddings"])
    verificar_indice(abrir_fn(destino), chunks)
    print(f"Índice reconstruido en {destino}: {len(chunks)} chunks, "
          f"{len({c['doc_id'] for c in chunks})} guías.")
    return destino


def filtrar_manifest(indice: retrieval.IndiceRAG) -> list[str]:
    """Deja en el manifiesto solo las guías del índice y devuelve las que se descartaron."""
    docs = {c["doc_id"] for c in indice.chunks}
    descartadas = sorted(set(indice.manifest) - docs)
    indice.manifest = {d: v for d, v in indice.manifest.items() if d in docs}
    return descartadas


def indice_desde_config(cfg: dict, **kw) -> retrieval.IndiceRAG:
    verificar_modelo_del_corpus(cfg)
    r, m, rt = rutas(cfg), cfg["modelos"], cfg["retrieval"]
    kw.setdefault("modelo_reranker", m["reranker"])
    indice = retrieval.construir_indice(preparar_indice(cfg), r["manifest"], modelo_embeddings=m["embeddings"],
                                        n_candidatos=rt["n_candidatos"], krrf=rt["krrf"],
                                        enriquecer=rt["encabezado_contextual"], **kw)
    if r.get("solo_guias_indexadas"):
        descartadas = filtrar_manifest(indice)
        if descartadas:
            print(f"Aviso: el manifiesto lista guías que no están en el índice y se ignoran: {descartadas}")
    return indice


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
        from clinical_rag.generation.mocks import crear_normalizador_mock
        mapa = json.loads(Path(mapa_ruta).read_text(encoding="utf-8"))
        return crear_normalizador_mock({k.strip().lower(): v for k, v in mapa.items()})

    n = cfg["normalizacion"]
    from clinical_rag.tools.normalization import normalizar_entidad
    ontologia = load_yaml(n["config"])["tool_normalizacion"]["ontologia"]
    return lambda entidad: normalizar_entidad(entidad, ontologia=ontologia)


def configurar_pipeline(cfg: dict) -> retrieval.IndiceRAG:
    """Construye el índice y lo deja listo para retrieve_naive y retrieve_advanced."""
    indice = indice_desde_config(cfg)
    retrieval.configurar(indice, cfg["retrieval"]["modo_pipeline"])
    return indice
