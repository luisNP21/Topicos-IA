"""
Secciones 1 y 2 del harness: resolucion de rutas del proyecto y
carga/validacion del gold set.
"""

import json
from pathlib import Path

import yaml


def resolver_rutas(cfg: dict, project_root: Path) -> dict:
    """Seccion 1: centraliza todas las rutas derivadas de PROJECT_ROOT + config.yaml."""
    gold_set_path = project_root / cfg["rutas"]["gold_set"]
    inference_config_path = Path(__file__).resolve().parent / cfg["rutas"]["inference_config"]
    with open(inference_config_path, "r", encoding="utf-8") as f:
        inference_cfg = yaml.safe_load(f)
    predictions_path = project_root / inference_cfg["output_path"]
    output_dir = project_root / cfg["rutas"]["outputs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rutas = {
        "gold_set_path": gold_set_path,
        "predictions_path": predictions_path,
        "output_dir": output_dir,
        "dim1_path": output_dir / cfg["archivos_salida"]["dimension1"],
        "dim1b_path": output_dir / cfg["archivos_salida"]["dimension1b"],
        "dim3_path": output_dir / cfg["archivos_salida"]["dimension3"],
        "scorecard_json_path": output_dir / cfg["archivos_salida"]["scorecard_json"],
        "scorecard_md_path": output_dir / cfg["archivos_salida"]["scorecard_md"],
    }

    print("Proyecto en:", project_root)
    print("Gold set:", rutas["gold_set_path"], "| existe:", rutas["gold_set_path"].exists())
    print("Predicciones:", rutas["predictions_path"], "| existe:", rutas["predictions_path"].exists())
    print("Output:", rutas["output_dir"])

    return rutas


def cargar_gold_set(gold_set_path: Path) -> list[dict]:
    """Seccion 2: carga y valida el JSONL de gold examples."""
    assert gold_set_path.exists(), (
        f"Gold set no encontrado en {gold_set_path}\n"
        f"Correr la celda de generacion de gold set en 01_dataset_preparation.ipynb primero."
    )

    gold_examples = []
    with open(gold_set_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_examples.append(json.loads(line))

    assert len(gold_examples) > 0, "El gold set esta vacio"

    records = []
    for i, ex in enumerate(gold_examples):
        text = ex.get("input", ex.get("text", ""))
        entities = ex.get("esperado", ex.get("entities_gold", []))
        assert isinstance(entities, list), (
            f"esperado/entities_gold debe ser list[str], vino como {type(entities)}"
        )
        records.append({
            "doc_id": ex.get("doc_id", f"ex_{i}"),
            "text": text,
            "entities_gold": entities,
            "criterio": ex.get("criterio", ""),
        })

    print(f"Gold set cargado: {len(records)} ejemplos")
    print("Formato valido")
    print("Primer ejemplo:")
    print("  text[:120]:", records[0]["text"][:120], "...")
    print("  entities_gold:", records[0]["entities_gold"])
    print("  n_entidades:", len(records[0]["entities_gold"]))

    return records


def cargar_eval_set(gold_set_path: Path) -> list[dict]:
    return cargar_gold_set(gold_set_path)