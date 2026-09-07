"""
Harness de evaluacion M2 -- DisTEMIST / Clinical BERT
Orquesta las 4 piezas del equipo en un solo comando:

    python run_harness.py --config config.yaml

Requiere un .env (no versionado) con:
    PROJECT_ROOT=/ruta/a/TopicosIA/Proyecto-Salud
    GROQ_API_KEY=gsk_...

No monta Drive ni conoce Colab -- eso es responsabilidad del notebook
o entorno que invoque este script. Aca solo se resuelven rutas ya dadas.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv


# ------------------------------------------------------------------
# Utilidades compartidas entre dimensiones 
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from common import fijar_seeds, log  # noqa: E402


def cargar_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise RuntimeError(
            "PROJECT_ROOT no esta definido. Crear un .env con:\n"
            "  PROJECT_ROOT=/ruta/a/TopicosIA/Proyecto-Salud"
        )
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"PROJECT_ROOT no existe: {root_path}")
    return root_path


def verificar_prerequisitos(cfg: dict, project_root: Path):
    """Falla rapido y con mensaje claro si falta algo, en vez de a mitad de dimension3."""
    gold_set = project_root / cfg["rutas"]["gold_set"]
    model_dir = project_root / cfg["rutas"]["model_dir"]

    if not gold_set.exists():
        raise RuntimeError(f"Gold set no encontrado: {gold_set}")
    if not (model_dir / "adapter_model.safetensors").exists():
        raise RuntimeError(f"Adaptador LoRA no encontrado en: {model_dir}")
    if cfg["dimension3_llm_judge"]["judge_provider"] == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY no definida en .env -- necesaria para dimension3.")

    log("Prerequisitos verificados: gold set, adaptador LoRA, API key.")


def main():
    parser = argparse.ArgumentParser(description="Harness de evaluacion M2")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config.yaml")
    parser.add_argument(
        "--solo", choices=["exact", "semantica", "judge", "scorecard"], default=None,
        help="Correr solo una dimension (util para debugging). Por defecto corre todo."
    )
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    project_root = resolver_project_root()
    outputs_dir = project_root / cfg["rutas"]["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    fijar_seeds(cfg["proyecto"]["seed_global"])
    log(f"Seed global fijada: {cfg['proyecto']['seed_global']}")
    log(f"PROJECT_ROOT: {project_root}")

    verificar_prerequisitos(cfg, project_root)

    resultados = {}

    # --- Dimension 1: exact-match (Luis) --------------------------------
    if args.solo in (None, "exact"):
        log("=" * 60)
        log("DIMENSION 1 -- Exact-match")
        log("=" * 60)
        import metrics_exact
        resultados["exact"] = metrics_exact.run(cfg, project_root)
        log(f"  F1 = {resultados['exact']['metrics']['f1']:.4f}")

    # --- Dimension 1b: similitud semantica (Pau) ------------------------
    if args.solo in (None, "semantica"):
        log("=" * 60)
        log("DIMENSION 1b -- Similitud semantica")
        log("=" * 60)
        import metrics_semantic
        resultados["semantica"] = metrics_semantic.run(cfg, project_root)
        log(f"  F1 = {resultados['semantica']['metrics']['f1']:.4f}  "
            f"(umbral={resultados['semantica']['configuracion']['umbral']})")

    # --- Dimension 3: LLM-as-judge (Agustin) ----------------------------
    if args.solo in (None, "judge"):
        log("=" * 60)
        log("DIMENSION 3 -- LLM-as-judge")
        log("=" * 60)
        import metrics_judge
        resultados["judge"] = metrics_judge.run(cfg, project_root)
        log(f"  Score juez = {resultados['judge']['metrics']['score_juez_mean']:.4f}")
        log(f"  Delta posicion = {resultados['judge']['sesgo_posicion']['delta_mean']:.4f}")

    # --- Scorecard integrador (Isa) --------------------------------------
    if args.solo in (None, "scorecard"):
        log("=" * 60)
        log("SCORECARD INTEGRADOR")
        log("=" * 60)
        import scorecard
        resultado_scorecard = scorecard.build(cfg, project_root)
        log(f"  Scorecard guardado en: {outputs_dir / cfg['scorecard']['archivo_salida_json']}")
        debilidad_top = max(
            resultado_scorecard["distribucion_debilidad"].items(), key=lambda x: x[1]
        )
        log(f"  Debilidad dominante: {debilidad_top[0]} ({debilidad_top[1]} docs)")

    log("=" * 60)
    log("Harness completado.")
    log("=" * 60)


if __name__ == "__main__":
    main()