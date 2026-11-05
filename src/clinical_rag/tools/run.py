"""
run_tools.py
Entry point para correr la tool de normalizacion desde linea de comandos.
Mismo patron que run_corpus.py de M3/corpus: codigo versionado en git,
datos y outputs en Drive via PROJECT_ROOT (.env). config.yaml se resuelve
relativo a este archivo para que funcione igual sin importar el runtime.

Uso:
    python run_tools.py --config config.yaml
"""
import argparse

from dotenv import load_dotenv

from clinical_rag.config import cargar_config
from clinical_rag.paths import CONFIG_DIR, project_root as default_project_root, resolve_config
from clinical_rag.tools.normalization import run


def main():
    parser = argparse.ArgumentParser(
        description="Tool de normalizacion terminologica -- M3"
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "tools.yaml"),
        help="Ruta al config.yaml",
    )
    args = parser.parse_args()

    load_dotenv()
    project_root = default_project_root()

    config_path = resolve_config(args.config)
    yaml_cfg = cargar_config(str(config_path))

    cfg = {
        "config_yaml": str(config_path),
        "tool_normalizacion": yaml_cfg["tool_normalizacion"],
    }

    stats = run(cfg=cfg, project_root=project_root)
    print(stats)


if __name__ == "__main__":
    main()
