"""
Secciones 1 y 2 del harness: resolucion de rutas del proyecto y
carga/validacion del gold set.
"""

import json
from pathlib import Path


def resolver_rutas(cfg: dict, project_root: Path) -> dict:
    """Seccion 1: centraliza todas las rutas derivadas de PROJECT_ROOT + config.yaml."""
    model_dir = project_root / cfg["rutas"]["model_dir"]
    gold_set_path = project_root / cfg["rutas"]["gold_set"]
    output_dir = project_root / cfg["rutas"]["outputs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rutas = {
        "model_dir": model_dir,
        "gold_set_path": gold_set_path,
        "output_dir": output_dir,
        "dim1_path": output_dir / cfg["archivos_salida"]["dimension1"],
        "dim1b_path": output_dir / cfg["archivos_salida"]["dimension1b"],
        "dim3_path": output_dir / cfg["archivos_salida"]["dimension3"],
        "scorecard_json_path": output_dir / cfg["archivos_salida"]["scorecard_json"],
        "scorecard_md_path": output_dir / cfg["archivos_salida"]["scorecard_md"],
    }

    print("Proyecto en:", project_root)
    print("Modelo (Drive):", rutas["model_dir"])
    print("  adapter_config existe:", (rutas["model_dir"] / "adapter_config.json").exists())
    print("  adapter_model existe: ", (rutas["model_dir"] / "adapter_model.safetensors").exists())
    print("Gold set:", rutas["gold_set_path"], "| existe:", rutas["gold_set_path"].exists())
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
    assert "input" in gold_examples[0], "Falta campo input en el gold set"
    assert "esperado" in gold_examples[0], "Falta campo esperado en el gold set"
    assert isinstance(gold_examples[0]["esperado"], list), (
        f"esperado debe ser list[str], vino como {type(gold_examples[0]['esperado'])}"
    )

    print(f"Gold set cargado: {len(gold_examples)} ejemplos")
    print("Formato valido")
    print("Primer ejemplo:")
    print("  input[:120]:", gold_examples[0]["input"][:120], "...")
    print("  esperado:", gold_examples[0]["esperado"])
    print("  n_entidades:", len(gold_examples[0]["esperado"]))

    return gold_examples