from __future__ import annotations

from common import fijar_seeds, log


def run(cfg: dict, project_root) -> dict:
    """Orquesta la evaluacion completa exactamente igual que el main legacy,
    pero centralizando la logica en el modulo harness.py.
    """
    outputs_dir = project_root / cfg["rutas"]["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    fijar_seeds(cfg["proyecto"]["seed_global"])
    log(f"Seed global fijada: {cfg['proyecto']['seed_global']}")
    log(f"PROJECT_ROOT: {project_root}")

    resultados = {}
    solo = None
    if "solo" in cfg:
        solo = cfg["solo"]

    if solo in (None, "exact"):
        log("=" * 60)
        log("DIMENSION 1 -- Exact-match")
        log("=" * 60)
        import metrics_exact
        resultados["exact"] = metrics_exact.run(cfg, project_root)
        log(f"  F1 = {resultados['exact']['metrics']['f1']:.4f}")

    if solo in (None, "semantica"):
        log("=" * 60)
        log("DIMENSION 1b -- Similitud semantica")
        log("=" * 60)
        import metrics_semantic
        resultados["semantica"] = metrics_semantic.run(cfg, project_root)
        log(f"  F1 = {resultados['semantica']['metrics']['f1']:.4f}  "
            f"(umbral={resultados['semantica']['configuracion']['umbral']})")

    if solo in (None, "judge"):
        log("=" * 60)
        log("DIMENSION 3 -- LLM-as-judge")
        log("=" * 60)
        import metrics_judge
        resultados["judge"] = metrics_judge.run(cfg, project_root)
        log(f"  Score juez = {resultados['judge']['metrics']['score_juez_mean']:.4f}")
        log(f"  Aciertos de dominio = "
            f"{resultados['judge']['metrics']['aciertos_dominio']}/"
            f"{resultados['judge']['metrics']['total_dominio']}")
        log(f"  Sesgo de posicion (tasa empate A/B) = "
            f"{resultados['judge']['sesgo_posicion']['tasa_sesgo_posicion']:.4f}")

    if solo in (None, "scorecard"):
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
    return resultados
