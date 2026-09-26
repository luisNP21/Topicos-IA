import chromadb
import json
import Path
from datetime import date


def cargar_coleccion(persist_dir: str, collection_name: str = "guias_clinicas"):
    """
    Para uso de Pau: abre la colección ya construida, sin reconstruirla.
    Lanza si la colección no existe todavía — mejor fallar explícito que crear una vacía por accidente.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)  # get_collection (no get_or_create) — falla si no existe


def guardar_chunks_json(chunks: list[dict], out_dir: str = "data/guias_clinicas/chunks"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for c in chunks:
        with open(f"{out_dir}/{c['chunk_id']}.json", "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=2)


def construir_indice_chroma(chunks: list[dict], embeddings: list[list[float]], persist_dir: str = "data/chroma_guias"):
    client = chromadb.PersistentClient(path=persist_dir)
    coleccion = client.get_or_create_collection("guias_clinicas", metadata={"hnsw:space": "cosine"})
    coleccion.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["texto"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"], "seccion": c["seccion"] or ""} for c in chunks]
    )
    return coleccion


def construir_manifest(fuentes: list[dict]) -> list[dict]:
    return [{
        "doc_id": f["doc_id"], "titulo": f["titulo"], "fuente_url": f["fuente_url"],
        "licencia": f["licencia"], "fecha_publicacion": f["fecha_publicacion"],
        "fecha_indexado": date.today().isoformat(), "responsable": f["responsable"]
    } for f in fuentes]