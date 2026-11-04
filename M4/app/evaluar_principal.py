"""
¿La app elige la misma enfermedad que el eval set de M3?

Corre la app (sin interfaz) sobre los casos de M3/ejecucion/eval_set_casos.json: a cada
`input` le aplica el NER, busca todas las entidades en las guías y elige la principal con
`PipelineClinico.elegir_principal`. Luego la compara con la columna `entidad` del caso.

Una elección cuenta como coincidente si la entidad elegida, o la consulta final después de
normalizar (por ejemplo "dm2" -> "diabetes mellitus tipo 2"), es igual a la entidad esperada o a
su forma normalizada. En los casos `sin_evidencia` lo correcto es que NO haya principal.

Uso, desde la raíz del repo:
    python M4/app/evaluar_principal.py
Salida: tabla en consola y M4/app/.cache/evaluacion_principal.csv
"""

from __future__ import annotations

import csv
import json
import unicodedata

from pipeline_app import REPO, PipelineClinico


def _norm(texto: str | None) -> str:
    texto = unicodedata.normalize("NFKD", (texto or "").lower().strip())
    return "".join(c for c in texto if not unicodedata.combining(c))


def main() -> None:
    casos = json.loads((REPO / "M3/ejecucion/eval_set_casos.json").read_text(encoding="utf-8"))
    p = PipelineClinico().cargar()

    filas = []
    for caso in casos:
        entidades, _ = p.extraer_entidades(caso["input"])
        resultados = [p.buscar(e) for e in entidades[: int(p.cfg["rag"]["max_entidades"])]]
        principal = p.elegir_principal(resultados)

        esperado = caso["entidad"]
        esperado_norm = p.normalizar_fn(esperado).get("entidad_normalizada") or esperado
        aceptadas = {_norm(esperado), _norm(esperado_norm)}

        if caso.get("tipo_caso") == "sin_evidencia":
            ok = principal is None
        else:
            ok = principal is not None and (
                _norm(principal["entidad"]) in aceptadas or _norm(principal["query_final"]) in aceptadas)

        detectada = any(_norm(e) in aceptadas for e in entidades)
        filas.append({
            "id": caso["id"], "tipo_caso": caso.get("tipo_caso"), "esperada": esperado,
            "elegida": principal["entidad"] if principal else "(ninguna)",
            "consulta_final": principal["query_final"] if principal else "",
            "coincide": "sí" if ok else "no",
            "esperada_entre_detectadas": "sí" if detectada else "no",
            "n_entidades": len(entidades), "entidades": " | ".join(entidades),
        })

    ancho = max(len(f["elegida"]) for f in filas)
    print(f"\n{'id':8} {'tipo':16} {'esperada':26} {'elegida':{ancho}}  coincide  esperada detectada")
    for f in filas:
        print(f"{f['id']:8} {f['tipo_caso'] or '':16} {f['esperada']:26} {f['elegida']:{ancho}}  "
              f"{f['coincide']:8}  {f['esperada_entre_detectadas']}")
    n_ok = sum(f["coincide"] == "sí" for f in filas)
    n_det = sum(f["esperada_entre_detectadas"] == "sí" for f in filas)
    print(f"\nCoincide la enfermedad principal: {n_ok} de {len(filas)}")
    print(f"La entidad esperada está entre las detectadas por el NER: {n_det} de {len(filas)}")

    salida = REPO / p.cfg["rutas"]["cache_dir"] / "evaluacion_principal.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"Detalle por caso: {salida}")


if __name__ == "__main__":
    main()
