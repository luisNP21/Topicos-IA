"""
Versión de referencia de corpus_utils.py, el módulo compartido de ingesta del corpus.
El notebook la copia a la raíz del repositorio solo si corpus_utils.py no existe todavía.
"""
import chromadb

EMBED_MODEL = "intfloat/multilingual-e5-base"


def construir_indice_chroma(chunks: list[dict], embeddings: list[list[float]], persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    coleccion = client.get_or_create_collection("guias_clinicas", metadata={"hnsw:space": "cosine"})
    coleccion.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["texto"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"], "seccion": c["seccion"] or ""} for c in chunks],
    )
    return coleccion


def cargar_coleccion(persist_dir: str):
    return chromadb.PersistentClient(path=persist_dir).get_collection("guias_clinicas")
