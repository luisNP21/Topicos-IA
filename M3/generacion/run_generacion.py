"""
Entry point de la generacion RAG  (Agustin / M3).

Integra las piezas REALES del equipo cuando `generacion.usar_mocks: false`:
  - retrieval avanzado de Paula ....... M3/retrieval/retrieval.py
  - tool de normalizacion de Luis ..... M3/tools/tool_normalizacion.py
y escribe `M3/outputs/resultado_generacion.json`, el archivo que consume RAGAS
(M3/ragas/evaluacion_ragas.py) para faithfulness y answer relevancy.

Los parametros de la orquestacion (umbral, umbral_evidencia, forzar_por_sigla, k)
NO se duplican aqui: se leen del YAML de Paula (`M3/retrieval/config_retrieval.yaml`),
como pidio Isabella. La pregunta de generacion es la MISMA que usa el reranker
(`retrieval.pregunta_intencion`), para que el LLM responda sobre lo que se busco.

Uso:
    python run_generacion.py --config config.yaml            # piezas reales
    python run_generacion.py --config config.yaml --mocks    # sin dependencias externas
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from generacion import GroqGenerator, contextos_para_ragas, generar_respuesta  # noqa: E402
from orquestacion import resolver_query  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def cargar_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_project_root() -> Path:
    load_dotenv()
    root = os.environ.get("PROJECT_ROOT")
    if root and Path(root).exists():
        return Path(root)
    return Path(__file__).resolve().parents[2]   # M3/generacion -> M3 -> repo


def _cargar_entidades(ruta: Path) -> list[dict]:
    """
    Acepta .json (lista de strings o de dicts) o .jsonl (una consulta por linea).

    Devuelve [{entidad, ground_truth}]. Se admiten los nombres del equipo
    ('consulta') y los de RAGAS ('question'/'ground_truth').
    """
    if ruta.suffix == ".jsonl":
        registros = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        registros = json.loads(ruta.read_text(encoding="utf-8"))

    entidades = []
    for r in registros:
        if isinstance(r, str):
            entidades.append({"entidad": r, "ground_truth": None})
            continue
        entidad = r.get("consulta") or r.get("entidad") or r.get("question") or ""
        ground_truth = r.get("ground_truth") or r.get("respuesta_esperada") or r.get("esperado")
        if entidad:
            entidades.append({"entidad": entidad, "ground_truth": ground_truth})
    return entidades


def _piezas_mock(cfg: dict) -> tuple[Callable, Callable, dict, Callable]:
    """Mocks propios: corre sin las piezas del equipo ni dependencias externas."""
    from mocks import crear_normalizador_mock, crear_retriever_mock
    o = cfg["orquestacion_mock"]
    _log("Piezas MOCK (retrieval y normalizacion propias).")
    return (crear_retriever_mock(), crear_normalizador_mock(),
            {"umbral": o["umbral"], "umbral_evidencia": o["umbral_evidencia"],
             "forzar_por_sigla": False, "k": o["k"]},
            lambda e: f"¿Qué dice la guía clínica sobre «{e}»?")


def _piezas_reales(cfg: dict, project_root: Path) -> tuple[Callable, Callable, dict, Callable]:
    """
    Piezas del equipo: retrieval (Pau) y normalizacion (Luis).
    Falla con un mensaje claro si las ramas aun no estan merged o faltan dependencias.
    """
    mod = cfg["modulos"]
    retr_dir = project_root / mod["retrieval_dir"]
    tool_dir = project_root / mod["tool_dir"]
    retr_cfg_path = project_root / mod["retrieval_config"]
    tool_cfg_path = project_root / mod["tool_config"]

    for carpeta in (retr_dir, tool_dir):
        sys.path.insert(0, str(carpeta))
    os.environ["M3_RETRIEVAL_CONFIG"] = str(retr_cfg_path)

    try:
        import retrieval
        from config_retrieval import (cargar_config as cargar_retr, configurar_pipeline,
                                      parametros_orquestacion)
        from tool_normalizacion import normalizar_entidad
    except Exception as e:
        raise RuntimeError(
            "No se pudieron importar las piezas reales (M3/retrieval y M3/tools). "
            "Verifica que las ramas de Paula y Luis esten integradas y que las "
            f"dependencias del retrieval (chromadb, rank_bm25, corpus_utils) esten instaladas: {e}"
        ) from e

    tool_cfg = yaml.safe_load(tool_cfg_path.read_text(encoding="utf-8"))
    ontologia = tool_cfg["tool_normalizacion"]["ontologia"]

    retrieval_cfg = cargar_retr(str(retr_cfg_path))
    configurar_pipeline(retrieval_cfg)           # construye el indice (Chroma) y fija el modo
    params = parametros_orquestacion(retrieval_cfg)
    _log(f"Piezas REALES: retrieval ({mod['retrieval_config']}) + tool ({ontologia}).")

    return (
        retrieval.retrieve_advanced,
        lambda entidad: normalizar_entidad(entidad, ontologia=ontologia),
        params,
        retrieval.pregunta_intencion,
    )


def run(cfg: dict, project_root: Path) -> dict:
    random.seed(cfg["proyecto"].get("seed_global", 42))
    gen_cfg = cfg["generacion"]
    rutas = cfg["rutas"]

    outputs_dir = project_root / rutas["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    entidades = _cargar_entidades(project_root / rutas["entidades"])
    _log(f"Entidades: {len(entidades)} (desde {rutas['entidades']})")

    if cfg.get("_forzar_mocks") or gen_cfg.get("usar_mocks", False):
        retrieve_fn, normalizar_fn, params, pregunta_fn = _piezas_mock(cfg)
    else:
        retrieve_fn, normalizar_fn, params, pregunta_fn = _piezas_reales(cfg, project_root)

    generar_fn = GroqGenerator(
        model=gen_cfg["modelo"], temperature=gen_cfg["temperature"],
        max_tokens=gen_cfg["max_tokens"],
        pausa_entre_llamadas=gen_cfg.get("pausa_entre_llamadas", 3.0),
    )

    _log(f"Generando sobre {len(entidades)} entidades (umbral={params['umbral']}, "
         f"umbral_evidencia={params.get('umbral_evidencia')}, k={params['k']}, "
         f"modelo={gen_cfg['modelo']})...")

    resultados = []
    for i, item in enumerate(entidades, 1):
        entidad = item["entidad"]
        q = resolver_query(
            entidad, retrieve_fn=retrieve_fn, normalizar_fn=normalizar_fn,
            umbral=params["umbral"], k=params["k"],
            umbral_evidencia=params.get("umbral_evidencia"),
            forzar_por_sigla=params.get("forzar_por_sigla", False),
        )
        resp = generar_respuesta(q["query_final"], q["fragments"], generar_fn=generar_fn,
                                 pregunta=pregunta_fn(entidad))
        _log(f"  [{i}/{len(entidades)}] {entidad!r} -> query={q['query_final']!r} | "
             f"tool={q['tool_invoked']} | fragmentos={len(q['fragments'])} | "
             f"fallback={resp['fallback_used']} | fuentes={len(resp['sources_used'])}")

        resultados.append({
            "entidad": entidad,
            "query_final": q["query_final"],
            "tool_invoked": q["tool_invoked"],
            "tool_reason": q["tool_reason"],
            "fragments": [f["chunk_id"] for f in q["fragments"]],
            "scores": [f.get("score") for f in q["fragments"]],
            "score_tipo": (q["fragments"][0].get("score_tipo") if q["fragments"] else None),
            # `contexts` y `answer` son lo que consume RAGAS (Luis).
            "contexts": contextos_para_ragas(q["fragments"]),
            "answer": resp["answer"],
            "sources_used": resp["sources_used"],
            "fallback_used": resp["fallback_used"],
            "ground_truth": item.get("ground_truth"),
        })

    n = len(resultados) or 1
    resumen = {
        "modulo": "generacion_rag",
        "rol": "agustin",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "proveedor": gen_cfg["proveedor"],
        "modelo_generador": gen_cfg["modelo"],
        "usar_mocks": bool(cfg.get("_forzar_mocks") or gen_cfg.get("usar_mocks", False)),
        "orquestacion": {k: params.get(k) for k in ("umbral", "umbral_evidencia", "forzar_por_sigla", "k")},
        "n_entidades": len(resultados),
        "metricas_generacion": {
            "tool_invocada_pct": sum(r["tool_invoked"] for r in resultados) / n,
            "fallback_pct": sum(r["fallback_used"] for r in resultados) / n,
            "con_fuentes_pct": sum(bool(r["sources_used"]) for r in resultados) / n,
        },
        "resultados": resultados,
    }

    archivo = rutas.get("archivo_salida", "resultado_generacion.json")
    salida = outputs_dir / archivo
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    _log(f"Guardado en: {salida}")
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generacion RAG (M3) -- Agustin")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--mocks", action="store_true",
                        help="Fuerza los mocks propios (sin piezas del equipo ni dependencias).")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    cfg["_forzar_mocks"] = args.mocks
    project_root = resolver_project_root()
    _log(f"PROJECT_ROOT: {project_root}")
    run(cfg, project_root)


if __name__ == "__main__":
    main()
