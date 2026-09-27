"""CLI wrapper para el harness de M2.

La logica de orquestacion real vive en `harness.py`, para mantener la
responsabilidad de evaluacion centralizada en ese modulo.
"""

import argparse
import os
import sys
from pathlib import Path

M2_ROOT = str(Path(__file__).resolve().parents[1])
if M2_ROOT in sys.path:
    sys.path.remove(M2_ROOT)
sys.path.insert(1, M2_ROOT)

import yaml
from dotenv import load_dotenv

from cached_system import sistema_desde_cache
from common import log
from gold_loader import cargar_gold_set, resolver_rutas
from harness import harness


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


def main():
    parser = argparse.ArgumentParser(description="Harness de evaluacion M2")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config.yaml")
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
        import scorecard
        return scorecard.build(cfg, project_root)

    eval_set = cargar_gold_set(rutas["gold_set_path"])
    if not rutas["predictions_path"].exists():
        raise FileNotFoundError(
            f"No se encontro el cache de predicciones: {rutas['predictions_path']}. "
            "Ejecuta primero M2/run_inference/run_inference.py."
        )
    sistema = sistema_desde_cache(rutas["predictions_path"], eval_set)
    metricas = harness(eval_set, sistema, cfg, project_root, solo=args.solo)

    if args.solo is None:
        import scorecard
        scorecard.build(cfg, project_root)

    log(f"Metricas finales: {metricas}")
    return metricas


if __name__ == "__main__":
    main()