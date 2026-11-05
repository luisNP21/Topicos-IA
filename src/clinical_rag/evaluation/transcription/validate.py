"""Compara lo que midio run_metricas con lo que se inyecto en la simulacion (tarea 5a).

- Errores de caracter: la distancia de edicion medida debe ser <= a las ediciones inyectadas
  (Levenshtein busca la alineacion minima) y cercana a ellas.
- Categorias de entidades: cada entidad inyectada debe caer en su categoria.
"""

from __future__ import annotations

import json
from pathlib import Path


def validar(salida: Path, resultados: list[dict]) -> dict:
    verdad = json.loads((salida / "caches" / "verdad_simulacion.json").read_text(encoding="utf-8"))
    informe = {}
    for r in resultados:
        v = verdad.get(r["nivel"])
        if v is None:
            continue
        inyectadas = sum(d["ediciones_inyectadas"] for d in v.values())
        tildes = sum(d["ediciones_tilde"] for d in v.values())
        medidas = r["visible"]["conteos"]["errores_caracter"]
        medidas_sd = r["visible"]["conteos"]["errores_caracter_sin_diacriticos"]
        esperado = {
            "acierto": sum(d["aciertos"] for d in v.values()),
            "mal_leida": sum(len(d["mal_leidas"]) for d in v.values()),
            "inventada": sum(len(d["inventadas"]) for d in v.values()),
            "fp_ner": sum(len(d["fp_ner"]) for d in v.values()),
            "texto_extra": sum(len(d["texto_extra"]) for d in v.values()),
            "fuera_de_transcripcion": 0,
        }
        a = r.get("auditoria_entidades") or {}
        medido = {k: a.get(k) for k in esperado}
        ratio = medidas / inyectadas if inyectadas else None
        informe[r["nivel"]] = {
            "ediciones_inyectadas": inyectadas, "errores_medidos": medidas,
            "ratio_medido_inyectado": None if ratio is None else round(ratio, 4),
            "ediciones_de_tilde_inyectadas": tildes,
            "errores_que_desaparecen_sin_diacriticos": medidas - medidas_sd,
            "entidades_esperadas": esperado, "entidades_medidas": medido,
            "ok_caracteres": ratio is not None and 0.85 <= ratio <= 1.0,
            "ok_entidades": esperado == medido,
        }
    informe["_todo_ok"] = all(x["ok_caracteres"] and x["ok_entidades"] for x in informe.values() if isinstance(x, dict))
    (salida / "validacion_simulacion.json").write_text(json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8")
    return informe
