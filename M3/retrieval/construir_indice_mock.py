"""
Construye el índice Chroma del corpus de prueba con el mismo código de la ingesta real
(M3/corpus: embeddings.py y corpus_store.py). Las rutas y el modelo salen de config_retrieval.yaml
(perfil "mock"). El índice del corpus real se reconstruye igual, desde sus chunks JSON, al llamar a
configurar_pipeline o indice_desde_config.

    python M3/retrieval/construir_indice_mock.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_retrieval import cargar_config, preparar_indice, verificar_modelo_del_corpus  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default=None)
a = p.parse_args()
cfg = cargar_config(a.config)
cfg["corpus"] = "mock"
verificar_modelo_del_corpus(cfg)
preparar_indice(cfg, forzar=True)
