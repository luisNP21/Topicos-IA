"""
run_ragas.py
Entry point para correr la evaluacion RAGAS desde linea de comandos.
codigo versionado en git, datos en Drive via PROJECT_ROOT (.env).

Uso:
    python run_ragas.py --config config.yaml
    python run_ragas.py --config config.yaml --modo real   # requiere GROQ_API_KEY
"""
import argparse
import json
import hashlib
import os
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        return False

from config_utils import cargar_config
from evaluacion_ragas import run as run_ragas
from cruce_harness import run as run_cruce

SCRIPT_DIR = Path(__file__).resolve().parent


def guardar_reporte(salida: dict, ruta: Path) -> None:
    """JSON estricto y sin sobrescribir una corrida previa."""
    contenido = json.dumps(salida, ensure_ascii=False, indent=2, allow_nan=False)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("x", encoding="utf-8") as archivo:
        archivo.write(contenido + "\n")


def versiones_evaluacion() -> dict:
    resultado = {}
    for paquete in ("ragas", "numpy", "langchain-groq", "sentence-transformers"):
        try:
            resultado[paquete] = version(paquete)
        except PackageNotFoundError:
            resultado[paquete] = "no_instalado"
    return resultado


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
    parser.add_argument("--salida", default=None, help="Archivo JSON nuevo; no sobrescribe reportes existentes.")
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

    ahora = datetime.now(timezone.utc)
    base_salida = Path(project_root) if project_root else SCRIPT_DIR.parents[1]
    ruta_salida = Path(args.salida) if args.salida else (
        base_salida / "M3/outputs" / f"evaluacion_m3_{ahora:%Y%m%dT%H%M%S%fZ}.json")
    if ruta_salida.exists():
        raise FileExistsError(f"El reporte ya existe: {ruta_salida}")
    meta = {"fecha_utc": ahora.isoformat(), "versiones": versiones_evaluacion(),
            "config": cfg, "esquema_reporte": 1,
            "codigo_evaluacion_sha256": {
                nombre: hashlib.sha256((SCRIPT_DIR / nombre).read_bytes()).hexdigest()
                for nombre in ("evaluacion_ragas.py", "cruce_harness.py", "run_ragas.py")}}

    # 1. Calcular métricas. Un error produce un artefacto de fallo, no números proxy.
    try:
        resultado_ragas = run_ragas(cfg=cfg, project_root=project_root)
    except Exception as exc:
        guardar_reporte({"meta": meta, "estado": "error",
                         "error_tipo": type(exc).__name__}, ruta_salida)
        raise

    # 2. Cruzar con F1 de extraccion de M2
    resultado_cruce = run_cruce(
        cfg=cfg,
        project_root=project_root,
        metricas_ragas=resultado_ragas["metricas"],
    )

    # 3. Imprimir tabla de diagnostico
    print("\n" + "=" * 60)
    print("RESULTADOS M2 Y M3: ALCANCE SEPARADO")
    print("=" * 60)
    print(resultado_cruce["tabla_markdown"])
    print(f"\nAlcance: {resultado_cruce['alcance']}")
    for tipo, conteo in resultado_cruce["resumen"].items():
        print(f"  {tipo}: {conteo} documentos")

    # 4. Conservar la evidencia por caso, el denominador y las versiones.
    salida = {
        "meta": meta,
        "ragas": resultado_ragas,
        "cruce_harness": {
            "f1_global_m2": resultado_cruce["f1_global_m2"],
            "resumen_diagnosticos": resultado_cruce["resumen"],
            "alcance": resultado_cruce["alcance"],
            "n_documentos_m2": resultado_cruce["n_documentos_m2"],
        },
    }
    print("\nResultado completo:")
    guardar_reporte(salida, ruta_salida)
    print(json.dumps(salida, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Reporte guardado: {ruta_salida}")


if __name__ == "__main__":
    main()

