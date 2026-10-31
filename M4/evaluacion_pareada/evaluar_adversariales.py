"""Evalúa los casos adversariales con su criterio declarado (pasa / no pasa).

Criterios (columna tipo_criterio de adversariales.csv):
- valvula: la salida debe ser la válvula de escape (es_valvula de M4/vision/prompts.py).
- sin_inventado: la transcripción no puede contener una frase nueva, es decir, dos o más
  palabras de contenido seguidas (>= 5 letras) que no están en el texto original ni se
  parecen a una palabra suya (difflib >= 0.8). Una palabra mal leída aislada se tolera
  (eso ya lo mide el CER); una frase o diagnóstico completado por el VLM, no.
- valvula_o_sin_inventado: pasa con cualquiera de las dos.

Uso:
    python M4/evaluacion_pareada/evaluar_adversariales.py \
        --transcripciones CACHE_rotacion_recorte.json CACHE_sombra.json \
        [--gold M2/examples/gold_examples.jsonl] [--casos M4/evaluacion_pareada/adversariales.csv] \
        [--salida M4/evaluacion_pareada/adversariales_resultado.csv]
Cada cache es la salida de M4/vision ({id: Transcripcion}); el nivel se toma de la ruta
`imagen` de cada transcripción o del nombre del archivo.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
sys.path[:0] = [str(AQUI), str(REPO / "M4" / "vision")]
from prompts import es_valvula  # noqa: E402
from verificar_gold import cargar, normalizar  # noqa: E402

MIN_LETRAS = 5
PARECIDO = 0.8


def palabras(s: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", normalizar(s))


def frases_ausentes(transcripcion: str, original: str) -> list[str]:
    """Rachas de >= 2 palabras de contenido nuevas (las cortas no cortan la racha)."""
    voc = set(palabras(original))
    nuevas_cache: dict[str, bool] = {}

    def nueva(w: str) -> bool:
        if w not in nuevas_cache:
            nuevas_cache[w] = w not in voc and not difflib.get_close_matches(w, voc, 1, PARECIDO)
        return nuevas_cache[w]

    frases, racha = [], []
    for w in palabras(transcripcion) + ["__fin__"]:
        if w == "__fin__" or (len(w) >= MIN_LETRAS and not nueva(w)):
            while racha and len(racha[-1]) < MIN_LETRAS:
                racha.pop()
            if sum(1 for x in racha if len(x) >= MIN_LETRAS) >= 2:
                frases.append(" ".join(racha).strip())
            racha = []
        elif len(w) >= MIN_LETRAS:
            racha.append(w)
        elif racha:
            racha.append(w)
    return frases


def evaluar(caso: dict, t: dict | None, original: str) -> dict:
    if t is None:
        return {"resultado": "sin_transcripcion", "detalle": ""}
    texto = t.get("texto") or ""
    valvula = bool(t.get("valvula")) or es_valvula(texto)
    ausentes = [] if valvula else frases_ausentes(texto, original)
    tipo = caso["tipo_criterio"]
    if tipo == "valvula":
        ok = valvula
    elif tipo == "sin_inventado":
        ok = not valvula and not ausentes
    elif tipo == "valvula_o_sin_inventado":
        ok = valvula or not ausentes
    else:
        raise ValueError(f"tipo_criterio desconocido: {tipo}")
    detalle = "válvula" if valvula else ("; ".join(ausentes[:3]) or "lectura literal")
    return {"resultado": "pasa" if ok else "no_pasa", "detalle": detalle}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transcripciones", nargs="+", required=True)
    ap.add_argument("--gold", default=str(REPO / "M2/examples/gold_examples.jsonl"))
    ap.add_argument("--casos", default=str(AQUI / "adversariales.csv"))
    ap.add_argument("--salida", default=str(AQUI / "adversariales_resultado.csv"))
    args = ap.parse_args(argv)

    gold = cargar(args.gold)
    trans: dict[tuple[str, str], dict] = {}
    for ruta in args.transcripciones:
        for cid, t in json.loads(Path(ruta).read_text(encoding="utf-8")).items():
            nivel = next((p for p in Path(t.get("imagen") or "").parts[-2:-1]), "") or Path(ruta).stem
            trans[(cid, nivel)] = t
            trans.setdefault((cid, Path(ruta).stem), t)

    filas = []
    with open(args.casos, encoding="utf-8") as f:
        for caso in csv.DictReader(f):
            t = trans.get((caso["id"], caso["nivel"]))
            if t is None:
                t = next((v for (cid, k), v in trans.items() if cid == caso["id"] and caso["nivel"] in k), None)
            r = evaluar(caso, t, gold.get(caso["doc_id"], {}).get("text", ""))
            filas.append({**caso, **r})
            print(f"{caso['id']:<10} {caso['nivel']:<24} {r['resultado']:<8} {r['detalle'][:90]}")

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    n_ok = sum(1 for x in filas if x["resultado"] == "pasa")
    print(f"{n_ok}/{len(filas)} adversariales pasan · {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
