"""
Construye un índice Chroma a partir del corpus de prueba, con el mismo código de ingesta del corpus
real (corpus_utils.py). Las rutas salen de config_retrieval.yaml (perfil "mock"). Para usar el
corpus real no hace falta este script: su índice lo construye la ingesta.

    python M3/retrieval/construir_indice_mock.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_retrieval import cargar_config  # noqa: E402
from corpus_utils import EMBED_MODEL, construir_indice_chroma  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default=None)
a = p.parse_args()
cfg = cargar_config(a.config)
rutas = cfg["rutas"]["mock"]
if cfg["modelos"]["embeddings"] != EMBED_MODEL:
    raise RuntimeError("El modelo de embeddings del YAML no coincide con el de corpus_utils.py.")

chunks = json.loads(Path(rutas["corpus_json"]).read_text(encoding="utf-8"))
invalidos = [c.get("chunk_id") for c in chunks if not {"chunk_id", "doc_id", "texto"} <= c.keys()]
if invalidos:
    raise ValueError(f"Chunks sin los campos obligatorios: {invalidos}")

modelo = SentenceTransformer(EMBED_MODEL)
embeddings = modelo.encode([f"passage: {c['texto']}" for c in chunks], normalize_embeddings=True).tolist()
coleccion = construir_indice_chroma(chunks, embeddings, persist_dir=rutas["chroma_dir"])
print(f"{coleccion.count()} chunks indexados en {rutas['chroma_dir']} ({len({c['doc_id'] for c in chunks})} documentos)")
