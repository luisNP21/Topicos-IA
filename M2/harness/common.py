import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m2_root_common", ROOT / "common.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

for _name in [
    "log",
    "fijar_seeds",
    "normalizar_entidad",
    "strip_chunk_suffix",
    "bio_to_entity_set",
    "aggregate_entities_by_original_doc",
    "micro_prf1_by_doc",
    "chunk_words",
]:
    globals()[_name] = getattr(MODULE, _name)
