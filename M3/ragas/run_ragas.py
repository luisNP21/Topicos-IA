"""
run_ragas.py
Entry point para correr la evaluacion RAGAS desde linea de comandos.
Mismo patron que run_corpus.py de M3/corpus y run_tools.py de M3/tools:
codigo versionado en git, datos en Drive via PROJECT_ROOT (.env).

Uso:
    python run_ragas.py --config config.yaml
    python run_ragas.py --config config.yaml --modo real   # requiere GROQ_API_KEY
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from config_utils import cargar_config
from evaluacion_ragas import run as run_ragas
from cruce_harness import run as run_cruce

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluacion RAGAS + cruce con harness M2 -- Luis"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al config.yaml (relativa a este script o absoluta)",
    )
    parser.add_argument(
        "--modo",
        choices=["mock", "real"],
        default=None,
        help="Sobreescribe ragas.modo del config.yaml",
    )
    args = parser.parse_args()

    load_dotenv()
    project_root = os.environ.get("PROJECT_ROOT", "")

    config_path = SCRIPT_DIR / args.config
    yaml_cfg = cargar_config(str(config_path))

    cfg = {
        "config_yaml": str(config_path),
        "ragas": yaml_cfg.get("ragas", {}),
        "cruce_harness": yaml_cfg.get("cruce_harness", {}),
    }

    # El flag --modo sobreescribe lo que haya en el yaml
    if args.modo:
        cfg["ragas"]["modo"] = args.modo

    # 1. Calcular metricas RAGAS
    resultado_ragas = run_ragas(cfg=cfg, project_root=project_root)

    # 2. Cruzar con F1 de extraccion de M2
    resultado_cruce = run_cruce(
        cfg=cfg,
        project_root=project_root,
        metricas_ragas=resultado_ragas["metricas"],
    )

    # 3. Imprimir tabla de diagnostico
    print("\n" + "=" * 60)
    print("TABLA DE CRUCE RAGAS x HARNESS M2")
    print("=" * 60)
    print(resultado_cruce["tabla_markdown"])
    print("\nResumen de diagnosticos:")
    for tipo, conteo in resultado_cruce["resumen"].items():
        print(f"  {tipo}: {conteo} documentos")

    # 4. Guardar resultado completo
    salida = {
        "ragas": resultado_ragas,
        "cruce_harness": {
            "f1_global_m2": resultado_cruce["f1_global_m2"],
            "resumen_diagnosticos": resultado_cruce["resumen"],
        },
    }
    print("\nResultado completo:")
    print(json.dumps(salida, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

