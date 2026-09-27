import yaml


def cargar_config(path: str) -> dict:
    """Carga un archivo YAML y lo devuelve como dict.
    Mismo patron que M3/corpus/config_utils.py y M3/tools/config_utils.py."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

