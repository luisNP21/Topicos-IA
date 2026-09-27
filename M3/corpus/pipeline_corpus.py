"""
Orquestador de la etapa de corpus para M3.
Unico responsable de: cargar config.yaml, recorrer fuentes.yaml, invocar
ingesta -> chunking -> embeddings -> store por cada fuente, y devolver
estadisticas + manifest.

Sigue el mismo contrato que el resto del proyecto: run(cfg, project_root) -> dict
No contiene logica de negocio propia -- todo vive en ingesta.py, chunking.py,
embeddings.py, corpus_store.py y config_utils.py. Este archivo solo encadena.
"""

import json
import traceback
from pathlib import Path
from datetime import date

from docling.document_converter import DocumentConverter

from config_utils import cargar_config, construir_patron_front_matter, construir_patrones_categoria
from ingesta import cargar_fuentes, ingerir_documento, filtrar_secciones
from chunking import chunk_seccion, renumerar_chunks, clasificar_seccion, cargar_tokenizer
from embeddings import cargar_modelo_embeddings, embeber_chunks
from corpus_store import construir_indice_chroma, guardar_chunks_json


def _procesar_fuente(
    fuente: dict,
    project_root: str,
    converter: DocumentConverter,
    modelo_embed,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    patron_front_matter,
    min_palabras: int,
    patrones_categoria: dict,
) -> dict:
    ruta_completa = str(Path(project_root) / fuente["archivo_local"])
    if not Path(ruta_completa).exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta_completa}")

    fuente_resuelta = {**fuente, "archivo_local": ruta_completa}
    doc_ingerido = ingerir_documento(fuente_resuelta, converter)

    secciones_filtradas = filtrar_secciones(doc_ingerido["secciones"], patron_front_matter, min_palabras)
    if not secciones_filtradas:
        raise ValueError(f"Todas las secciones de '{fuente['doc_id']}' fueron filtradas como front matter")

    chunks = []
    for seccion in secciones_filtradas:
        nuevos = chunk_seccion(doc_ingerido["doc_id"], seccion, tokenizer, max_tokens, overlap_tokens)
        categoria = clasificar_seccion(seccion["titulo"], patrones_categoria)
        for c in nuevos:
            c["categoria_seccion"] = categoria
        chunks.extend(nuevos)
    chunks = renumerar_chunks(doc_ingerido["doc_id"], chunks)

    embeddings = embeber_chunks(chunks, modelo_embed)

    return {
        "doc_id": doc_ingerido["doc_id"],
        "chunks": chunks,
        "embeddings": embeddings,
        "n_secciones_originales": len(doc_ingerido["secciones"]),
        "n_secciones_filtradas": len(doc_ingerido["secciones"]) - len(secciones_filtradas),
    }

def run(cfg: dict, project_root: str) -> dict:
    """
    cfg esperado:
        {
            "fuentes_yaml": str,    # ruta a fuentes.yaml, relativa o absoluta
            "config_yaml": str,     # ruta a config.yaml, relativa o absoluta
            "chunks_dir": str,      # ej. "data/guias_clinicas/chunks"
            "chroma_dir": str,      # ej. "data/chroma_guias"
            "manifest_path": str,   # ej. "data/guias_clinicas/corpus_manifest.json"
        }

    Devuelve:
        {
            "n_fuentes_ok": int,
            "n_fuentes_fallidas": int,
            "n_documentos": int,
            "n_chunks": int,
            "n_secciones_filtradas": int,
            "errores": [{"doc_id": str, "error": str}],
        }
    """
    fuentes_yaml_path = cfg["fuentes_yaml"]
    fuentes_yaml_path = fuentes_yaml_path if Path(fuentes_yaml_path).is_absolute() else str(Path(project_root) / fuentes_yaml_path)
    fuentes = cargar_fuentes(fuentes_yaml_path)

    config_path = cfg["config_yaml"]
    config_path = config_path if Path(config_path).is_absolute() else str(Path(project_root) / config_path)
    config = cargar_config(config_path)

    patron_front_matter = construir_patron_front_matter(config["front_matter"]["titulos_excluidos"])
    min_palabras = config["front_matter"]["min_palabras"]
    patrones_categoria = construir_patrones_categoria(config["categorias_seccion"])
    modelo_embed = cargar_modelo_embeddings(config["embeddings"]["model_name"])
    tokenizer = cargar_tokenizer(config["embeddings"]["model_name"])
    max_tokens = config["chunking"]["max_tokens"]
    overlap_tokens = config["chunking"]["overlap_tokens"]

    converter = DocumentConverter()

    todos_los_chunks = []
    todos_los_embeddings = []
    fuentes_ok = []
    errores = []
    n_secciones_filtradas_total = 0

    for fuente in fuentes:
        try:
            resultado = _procesar_fuente(
                fuente, project_root, converter, modelo_embed, tokenizer, max_tokens, overlap_tokens,
                patron_front_matter, min_palabras, patrones_categoria,
            )
            todos_los_chunks.extend(resultado["chunks"])
            todos_los_embeddings.extend(resultado["embeddings"])
            n_secciones_filtradas_total += resultado["n_secciones_filtradas"]
            fuentes_ok.append(fuente)
            print(f"OK {fuente['doc_id']}: {len(resultado['chunks'])} chunks")
        except Exception as e:
            errores.append({"doc_id": fuente.get("doc_id", "desconocido"), "error": str(e)})
            print(f"ERROR {fuente.get('doc_id', 'desconocido')}: {e}")
            print(traceback.format_exc())
            continue  # una fuente fallida no debe detener el resto del lote

    if not todos_los_chunks:
        raise RuntimeError(
            f"Ninguna fuente se proceso exitosamente. Errores: {errores}"
        )

    # Persistencia -- solo con lo que si se proceso bien
    chunks_dir = str(Path(project_root) / cfg["chunks_dir"])
    chroma_dir = str(Path(project_root) / cfg["chroma_dir"])
    manifest_path = str(Path(project_root) / cfg["manifest_path"])

    guardar_chunks_json(todos_los_chunks, out_dir=chunks_dir)
    coleccion = construir_indice_chroma(todos_los_chunks, todos_los_embeddings, persist_dir=chroma_dir)

    manifest = [{
        "doc_id": f["doc_id"],
        "titulo": f["titulo"],
        "fuente_url": f["fuente_url"],
        "licencia": f["licencia"],
        "fecha_publicacion": f["fecha_publicacion"],
        "fecha_indexado": date.today().isoformat(),
        "responsable": f["responsable"],
    } for f in fuentes_ok]

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    stats = {
        "n_fuentes_ok": len(fuentes_ok),
        "n_fuentes_fallidas": len(errores),
        "n_documentos": len(fuentes_ok),
        "n_chunks": coleccion.count(),
        "n_secciones_filtradas": n_secciones_filtradas_total,
        "errores": errores,
    }

    if errores:
        print(f"AVISO: {len(errores)} fuente(s) fallaron -- revisa stats['errores'] antes de considerar el corpus completo")

    return stats