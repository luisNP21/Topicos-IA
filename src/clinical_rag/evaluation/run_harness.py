"""CLI wrapper para el harness de M2.

La logica de orquestacion real vive en `harness.py`, para mantener la
responsabilidad de evaluacion centralizada en ese modulo.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from clinical_rag.config import load_yaml
from clinical_rag.evaluation.cached_system import sistema_desde_cache
from clinical_rag.common import log
from clinical_rag.evaluation.gold import cargar_gold_set, resolver_rutas
from clinical_rag.evaluation.harness import harness
from clinical_rag.evaluation.scorecard import build as build_scorecard
from clinical_rag.paths import project_root as default_project_root, resolve_config


def cargar_config(config_path: str) -> dict:
    return load_yaml(config_path)


def resolver_project_root() -> Path:
    load_dotenv()
    return default_project_root()


def main():
    parser = argparse.ArgumentParser(description="Harness de evaluacion M2")
    parser.add_argument("--config", default="configs/harness.yaml", help="Ruta al config.yaml")
    parser.add_argument(
        "--solo",
        choices=["exact", "semantica", "judge", "scorecard"],
        default=None,
        help="Correr solo una dimension (util para debugging).",
    )
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    project_root = resolver_project_root()
    rutas = resolver_rutas(cfg, project_root)

    if args.solo == "scorecard":
        return build_scorecard(cfg, project_root)

    eval_set = cargar_gold_set(rutas["gold_set_path"])
    if not rutas["predictions_path"].exists():
        raise FileNotFoundError(
            f"No se encontro el cache de predicciones: {rutas['predictions_path']}. "
            "Ejecuta primero scripts/run_ner_inference.py."
        )
    sistema = sistema_desde_cache(rutas["predictions_path"], eval_set)
    metricas = harness(eval_set, sistema, cfg, project_root, solo=args.solo)

    if args.solo is None:
        build_scorecard(cfg, project_root)

    log(f"Metricas finales: {metricas}")
    return metricas


if __name__ == "__main__":
    main()
