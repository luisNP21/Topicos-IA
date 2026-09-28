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

    Devuelve [{id, entidad, pregunta, tipo_caso, ground_truth}]. Admita los nombres
    del eval set del equipo ('entidad', 'pregunta', 'esperado') y los de RAGAS
    ('question', 'ground_truth'). La `pregunta` del caso, si viene, se usa tal cual
    para la generacion; si no, se cae a la plantilla de intencion del retrieval.
    """
    if ruta.suffix == ".jsonl":
        registros = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        registros = json.loads(ruta.read_text(encoding="utf-8"))

    entidades = []
    for r in registros:
        if isinstance(r, str):
            entidades.append({"id": None, "entidad": r, "pregunta": None,
                              "tipo_caso": None, "ground_truth": None})
            continue
        entidad = r.get("entidad") or r.get("consulta") or r.get("question") or ""
        if not entidad:
            continue
        entidades.append({
            "id": r.get("id"),
            "entidad": entidad,
            "pregunta": r.get("pregunta") or r.get("question"),
            "tipo_caso": r.get("tipo_caso"),
            "ground_truth": r.get("ground_truth") or r.get("respuesta_esperada") or r.get("esperado"),
        })
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

    La normalizacion se toma con `normalizador_desde_config` de Pau, que elige el
    mapa simulado (corpus mock) o la tool real de Luis segun `config_retrieval.yaml`.
    Falla con un mensaje claro si las piezas no estan integradas o faltan dependencias.
    """
    mod = cfg["modulos"]
    retr_dir = project_root / mod["retrieval_dir"]
    retr_cfg_path = project_root / mod["retrieval_config"]

    sys.path.insert(0, str(retr_dir))
    # Si el entorno ya fijo M3_RETRIEVAL_CONFIG (por ejemplo, un override local que
    # arma el notebook), se respeta; si no, se usa el YAML del repo.
    os.environ.setdefault("M3_RETRIEVAL_CONFIG", str(retr_cfg_path))

    try:
        import retrieval
        from config_retrieval import (cargar_config as cargar_retr, configurar_pipeline,
                                      normalizador_desde_config, parametros_orquestacion)
    except Exception as e:
        raise RuntimeError(
            "No se pudieron importar las piezas reales (M3/retrieval y M3/tools). "
            "Verifica que las ramas de Pau y Luis esten integradas y que las "
            f"dependencias del retrieval (chromadb, rank_bm25) esten instaladas: {e}"
        ) from e

    retrieval_cfg = cargar_retr(os.environ["M3_RETRIEVAL_CONFIG"])   # respeta el override local si existe
    configurar_pipeline(retrieval_cfg)                         # indice Chroma + modo del pipeline
    params = parametros_orquestacion(retrieval_cfg)            # umbrales (calibrados o provisionales)
    normalizar_fn = normalizador_desde_config(retrieval_cfg)   # mapa mock o tool real, segun el YAML
    _log(f"Piezas REALES: retrieval ({mod['retrieval_config']}) + normalizacion "
         f"(corpus '{retrieval_cfg['corpus']}').")

    return retrieval.retrieve_advanced, normalizar_fn, params, retrieval.pregunta_intencion


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
        # Pregunta del caso si viene; si no, la plantilla de intencion del retrieval.
        pregunta = item.get("pregunta") or pregunta_fn(entidad)
        resp = generar_respuesta(q["query_final"], q["fragments"], generar_fn=generar_fn,
                                 pregunta=pregunta)
        _log(f"  [{i}/{len(entidades)}] {entidad!r} ({item.get('tipo_caso') or '-'}) -> "
             f"query={q['query_final']!r} | tool={q['tool_invoked']} | "
             f"fragmentos={len(q['fragments'])} | fallback={resp['fallback_used']} | "
             f"fuentes={len(resp['sources_used'])}")

        resultados.append({
            "id": item.get("id"),
            "entidad": entidad,
            # `question` es la pregunta real del caso; RAGAS deberia usarla (hoy usa `entidad`).
            "question": pregunta,
            "tipo_caso": item.get("tipo_caso"),
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
    parser.add_argument("--entidades", default=None,
                        help="Eval set a generar (ruta relativa a PROJECT_ROOT); sobreescribe "
                             "rutas.entidades del config. Util para el ejemplos.json del equipo.")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    cfg["_forzar_mocks"] = args.mocks
    if args.entidades:
        cfg["rutas"]["entidades"] = args.entidades
    project_root = resolver_project_root()
    _log(f"PROJECT_ROOT: {project_root}")
    run(cfg, project_root)


if __name__ == "__main__":
    main()
