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
from corpus_store import construir_indice_chroma, guardar_chunks_json, cargar_coleccion


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


def _chunks_existen(doc_id: str, chunks_dir: str) -> bool:
    """
    True si ya hay al menos un chunk guardado en disco para este doc_id.
    Regla del equipo: una nueva version de un PDF se sube con un doc_id/nombre
    distinto, nunca sobrescribe el mismo doc_id -- por eso basta con chequear
    existencia, sin comparar hash ni fecha de modificacion del PDF.
    """
    return any(Path(chunks_dir).glob(f"{doc_id}_chunk*.json"))


def run(cfg: dict, project_root: str) -> dict:
    """
    cfg esperado:
        {
            "fuentes_yaml": str,
            "config_yaml": str,
        }

    Fuentes cuyo doc_id ya tiene chunks guardados en chunks_dir se saltan por
    completo (no se re-ingieren, no se re-chunkean, no se re-embeben) -- se
    asume que ya estan indexadas en Chroma de una corrida anterior. Solo se
    reprocesan fuentes nuevas o renombradas.
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

    chunks_dir = str(Path(project_root) / config["paths"]["chunks_dir"])
    chroma_dir = str(Path(project_root) / config["paths"]["chroma_dir"])
    manifest_path = str(Path(project_root) / config["paths"]["manifest_path"])

    converter = DocumentConverter()

    todos_los_chunks = []
    todos_los_embeddings = []
    fuentes_ok = []          # todas las que quedan bien indexadas (nuevas + cacheadas)
    fuentes_cacheadas = []   # NUEVO: subset de fuentes_ok que se salto por cache
    errores = []
    n_secciones_filtradas_total = 0

    for fuente in fuentes:
        doc_id = fuente["doc_id"]

        if _chunks_existen(doc_id, chunks_dir):  # NUEVO
            print(f"CACHE {doc_id}: chunks ya existen en disco, se omite reprocesamiento")
            fuentes_ok.append(fuente)
            fuentes_cacheadas.append(doc_id)
            continue

        try:
            resultado = _procesar_fuente(
                fuente, project_root, converter, modelo_embed, tokenizer, max_tokens, overlap_tokens,
                patron_front_matter, min_palabras, patrones_categoria,
            )
            todos_los_chunks.extend(resultado["chunks"])
            todos_los_embeddings.extend(resultado["embeddings"])
            n_secciones_filtradas_total += resultado["n_secciones_filtradas"]
            fuentes_ok.append(fuente)
            print(f"OK {doc_id}: {len(resultado['chunks'])} chunks")
        except Exception as e:
            errores.append({"doc_id": fuente.get("doc_id", "desconocido"), "error": str(e)})
            print(f"ERROR {fuente.get('doc_id', 'desconocido')}: {e}")
            print(traceback.format_exc())
            continue

    # NUEVO: ya no es un error que no haya chunks nuevos -- puede ser que todo
    # ya estuviera cacheado. Solo es error real si ademas no hay nada cacheado.
    if not todos_los_chunks and not fuentes_cacheadas:
        raise RuntimeError(f"Ninguna fuente se proceso exitosamente. Errores: {errores}")

    guardar_chunks_json(todos_los_chunks, out_dir=chunks_dir)  # no-op si la lista viene vacia

    # NUEVO: si hay chunks nuevos, se construye/actualiza el indice; si todo
    # estaba cacheado, solo se abre la coleccion existente para poder contar.
    if todos_los_chunks:
        coleccion = construir_indice_chroma(todos_los_chunks, todos_los_embeddings, persist_dir=chroma_dir)
    else:
        coleccion = cargar_coleccion(persist_dir=chroma_dir)

    # NUEVO: manifest se fusiona con el existente en vez de sobrescribirlo --
    # las fuentes cacheadas conservan su fecha_indexado original.
    manifest_existente = {}
    if Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_existente = {m["doc_id"]: m for m in json.load(f)}

    manifest = []
    for f in fuentes_ok:
        if f["doc_id"] in fuentes_cacheadas and f["doc_id"] in manifest_existente:
            manifest.append(manifest_existente[f["doc_id"]])  # conserva entrada previa intacta
        else:
            manifest.append({
                "doc_id": f["doc_id"], "titulo": f["titulo"], "fuente_url": f["fuente_url"],
                "licencia": f["licencia"], "fecha_publicacion": f["fecha_publicacion"],
                "fecha_indexado": date.today().isoformat(), "responsable": f["responsable"],
            })

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    stats = {
        "n_fuentes_ok": len(fuentes_ok),
        "n_fuentes_cacheadas": len(fuentes_cacheadas),  # NUEVO
        "n_fuentes_nuevas": len(fuentes_ok) - len(fuentes_cacheadas),  # NUEVO
        "n_fuentes_fallidas": len(errores),
        "n_documentos": len(fuentes_ok),
        "n_chunks": coleccion.count(),
        "n_secciones_filtradas": n_secciones_filtradas_total,
        "errores": errores,
    }

    if errores:
        print(f"AVISO: {len(errores)} fuente(s) fallaron -- revisa stats['errores'] antes de considerar el corpus completo")

    return stats