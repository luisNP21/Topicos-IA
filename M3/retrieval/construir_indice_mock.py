"""
Construye el índice Chroma del corpus de prueba con el mismo código de la ingesta real
(M3/corpus: embeddings.py y corpus_store.py). Las rutas y el modelo salen de config_retrieval.yaml
(perfil "mock"). Para el corpus real no hace falta este script: su índice lo construye la ingesta.

    python M3/retrieval/construir_indice_mock.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_retrieval import cargar_config, resolver, verificar_modelo_del_corpus  # noqa: E402

sys.path.insert(0, resolver("M3/corpus"))

from corpus_store import construir_indice_chroma  # noqa: E402
from embeddings import cargar_modelo_embeddings, embeber_chunks  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default=None)
a = p.parse_args()
cfg = cargar_config(a.config)
verificar_modelo_del_corpus(cfg)
rutas = {k: resolver(v) if isinstance(v, str) else v for k, v in cfg["rutas"]["mock"].items()}

chunks = json.loads(Path(rutas["corpus_json"]).read_text(encoding="utf-8"))
invalidos = [c.get("chunk_id") for c in chunks if not {"chunk_id", "doc_id", "texto"} <= c.keys()]
if invalidos:
    raise ValueError(f"Chunks sin los campos obligatorios: {invalidos}")

modelo = cargar_modelo_embeddings(cfg["modelos"]["embeddings"])
coleccion = construir_indice_chroma(chunks, embeber_chunks(chunks, modelo), persist_dir=rutas["chroma_dir"])
print(f"{coleccion.count()} chunks indexados en {rutas['chroma_dir']} ({len({c['doc_id'] for c in chunks})} documentos)")
