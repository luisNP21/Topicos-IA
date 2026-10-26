"""Transcripciones y NER simulados con errores conocidos (tarea 5a).

Sirve para comprobar que las metricas miden lo que deben antes de tener transcripciones
reales. NO produce resultados del sistema: todo lo que sale de aqui se marca como simulado.

Por pagina parte del texto visible del manifiesto e inyecta, con semilla:
- ruido de caracteres (sustitucion, borrado o insercion) fuera de las entidades,
- perdida de tildes y enies,
- una entidad del gold mal escrita (un caracter cambiado),
- frases con enfermedades que no estan en el original,
- copia del texto extra (sello, nota, tachadura),
- paginas declaradas sin texto (valvula).
Lleva la cuenta exacta de lo inyectado en `verdad_simulacion.json`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from auditoria import aparece, normalizar_entidad, normalizar_texto, parecido
from paginas import MENSAJE_SIN_TEXTO

LETRAS = "abcdefghijklmnopqrstuvwxyz"
SIN_TILDE = str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN")


def _semilla(*partes) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, partes)).encode()).digest()[:4], "big")


def _patron_libre(entidad: str) -> re.Pattern:
    """La entidad en el texto de una pagina, aunque cruce un salto de linea."""
    partes = [re.escape(p) for p in entidad.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(partes) + r"(?!\w)", re.IGNORECASE)


def _entero(valor, rng) -> int:
    a, b = valor
    return int(rng.integers(a, b + 1))


def _cambiar_letra(texto: str, rng) -> str:
    pos = [i for i, c in enumerate(texto) if c.isalpha() and 0 < i < len(texto) - 1]
    if not pos:
        return texto
    i = pos[int(rng.integers(len(pos)))]
    nueva = rng.choice([c for c in LETRAS if c != texto[i].lower()])
    return texto[:i] + str(nueva) + texto[i + 1:]


def _ruido(texto: str, protegido: np.ndarray, tasa: float, p_tilde: float, rng) -> tuple[str, int, int]:
    """Aplica ruido a los caracteres no protegidos. Devuelve (texto, ediciones, ediciones_de_tilde)."""
    salida, ediciones, tildes = [], 0, 0
    for i, c in enumerate(texto):
        if protegido[i] or c.isspace():
            salida.append(c)
            continue
        if c in "áéíóúñÁÉÍÓÚÑ" and rng.random() < p_tilde:
            salida.append(c.translate(SIN_TILDE))
            ediciones += 1
            tildes += 1
            continue
        if rng.random() < tasa:
            op = rng.random()
            ediciones += 1
            if op < 0.6:
                salida.append(str(rng.choice([l for l in LETRAS if l != c.lower()])))
            elif op < 0.8:
                pass                                   # borrado
            else:
                salida.append(c + str(rng.choice(list(LETRAS))))
            continue
        salida.append(c)
    return "".join(salida), ediciones, tildes


def generar(cfg: dict, manifest: dict, gold_por_doc: dict, salida: Path) -> list[dict]:
    """Escribe caches simulados por nivel y devuelve las entradas para run_metricas."""
    sim = cfg["simulacion"]
    seed = cfg["proyecto"]["seed_global"]
    salida.mkdir(parents=True, exist_ok=True)
    resumen_niveles = manifest["niveles"]
    entradas, verdad = [], {}

    for nivel in sim["niveles"]:
        if nivel not in resumen_niveles:
            continue
        severidad = resumen_niveles[nivel].get("severidad", "mixta")
        tasa = float(sim["tasa_caracter"].get(severidad, sim["tasa_caracter"]["mixta"]))
        cache, ner, verdad_nivel = {}, {}, {}

        for doc in manifest["documentos"]:
            info = doc["niveles"].get(nivel)
            if info is None:
                continue
            doc_id = doc["doc_id"]
            rng = np.random.default_rng(_semilla(seed, "simulacion", nivel, doc_id))
            original_n = normalizar_texto(gold_por_doc[doc_id]["text"])
            ocultas = {x["entidad"] for x in info["entidades"]["no_visibles"] if x["estado"] == "oculta"}
            gold = [e for e in dict.fromkeys(gold_por_doc[doc_id]["entities_gold"]) if e not in ocultas]
            paginas = list(info["texto_visible_por_pagina"])
            extras = [" ".join(x["texto"] for x in pg["texto_extra"]) for pg in info["paginas"]]

            # Entidades a escribir mal: del gold, visibles en alguna pagina.
            # >= 8 caracteres: con una letra cambiada, el parecido es 1 - 1/n >= 87,5 (sobre el umbral de 85).
            # En entidades de 6 o menos, un solo error ya cae como "inventada" (limitacion documentada).
            candidatas = [e for e in gold if len(normalizar_entidad(e)) >= 8
                          and any(_patron_libre(e).search(t) for t in paginas)]
            rng.shuffle(candidatas)
            # Sin solapes ("lesión" dentro de "lesión tumoral"): alterar una romperia la otra.
            mal = []
            for e in candidatas:
                if len(mal) >= _entero(sim["mal_leidas_por_doc"], rng):
                    break
                en = normalizar_entidad(e)
                if all(en not in normalizar_entidad(o) and normalizar_entidad(o) not in en for o in gold if o != e):
                    mal.append(e)

            # Fragmentos del original que el NER marcara sin ser gold (se protegen del ruido).
            gold_n = {normalizar_entidad(e) for e in gold_por_doc[doc_id]["entities_gold"]}
            fps = []
            for _ in range(_entero(sim["fp_ner_por_doc"], rng)):
                k = int(rng.integers(len(paginas)))
                pal = paginas[k].split()
                if len(pal) < 3:
                    continue
                j = int(rng.integers(len(pal) - 1))
                frag = " ".join(pal[j:j + 2])
                fn = normalizar_entidad(frag)
                # Debe estar tal cual en el original: en el texto visible dos palabras pueden
                # quedar juntas aunque en el original las separe texto oculto.
                if fn and fn not in gold_n and aparece(fn, original_n):
                    fps.append(frag)

            ediciones = tildes = 0
            mal_escritas, inventadas, valvulas, extra_copiado = [], [], [], []
            for k, texto in enumerate(paginas):
                if rng.random() < sim["p_valvula_pagina"]:
                    valvulas.append(k + 1)
                    ediciones += len(normalizar_texto(texto)) + 1
                    cache[f"{doc_id}__p{k + 1}"] = _registro(doc_id, k, MENSAJE_SIN_TEXTO, True, info)
                    continue
                for e in mal:
                    m = _patron_libre(e).search(texto)
                    if m and e not in [x["original"] for x in mal_escritas]:
                        nueva = _cambiar_letra(" ".join(m.group().split()), rng)
                        texto = texto[:m.start()] + nueva + texto[m.end():]
                        ediciones += 1    # los espacios se colapsan al normalizar: solo cuenta la letra
                        mal_escritas.append({"original": e, "escrita": nueva})
                protegido = np.zeros(len(texto), bool)
                for e in gold + fps + [x["escrita"] for x in mal_escritas]:
                    for m in _patron_libre(e).finditer(texto):
                        protegido[m.start():m.end()] = True
                texto, ed, ti = _ruido(texto, protegido, tasa, sim["p_perder_tilde"], rng)
                ediciones += ed
                tildes += ti
                if extras[k] and rng.random() < sim["p_transcribir_extra"]:
                    texto += "\n" + extras[k]
                    ediciones += len(normalizar_texto(extras[k])) + 1
                    extra_copiado.append(extras[k])
                cache[f"{doc_id}__p{k + 1}"] = _registro(doc_id, k, texto, False, info)

            # Enfermedades inventadas: no deben parecerse al original (si no, serian "mal_leida").
            n_inv = _entero(sim["inventadas_por_doc"], rng)
            no_valvula = [k for k in range(len(paginas)) if k + 1 not in valvulas]
            for enf in sim["enfermedades_inventadas"]:
                if len(inventadas) >= n_inv or not no_valvula:
                    break
                if parecido(normalizar_entidad(enf), original_n)[0] >= cfg["auditoria"]["umbral_mal_leida"]:
                    continue
                k = no_valvula[int(rng.integers(len(no_valvula)))]
                frase = f" Antecedentes de {enf}."
                cache[f"{doc_id}__p{k + 1}"]["texto"] += frase
                ediciones += len(frase)
                inventadas.append(enf)

            # NER simulado sobre la transcripcion unida.
            transcripcion_n = normalizar_texto("\n".join(
                cache[f"{doc_id}__p{k + 1}"]["texto"] for k in range(len(paginas)) if k + 1 not in valvulas))
            detectadas = [e for e in gold if re.search(r"(?<!\w)" + re.escape(normalizar_entidad(e)) + r"(?!\w)",
                                                       transcripcion_n) and rng.random() < sim["p_entidad_detectada"]]
            fps_ok = [f for f in fps if normalizar_entidad(f) in transcripcion_n]
            extra_ner = []
            for ex in extra_copiado:
                pal = [p for p in ex.split() if len(normalizar_entidad(p)) >= 4]
                if pal and not aparece(normalizar_entidad(pal[0]), original_n):
                    extra_ner.append(pal[0])
                    break
            ner[doc_id] = sorted(set(detectadas + [x["escrita"] for x in mal_escritas] + inventadas + fps_ok + extra_ner))
            verdad_nivel[doc_id] = {
                "ediciones_inyectadas": ediciones, "ediciones_tilde": tildes, "paginas_valvula": valvulas,
                "aciertos": len({normalizar_entidad(e) for e in detectadas}),
                "mal_leidas": [x["escrita"] for x in mal_escritas], "inventadas": inventadas,
                "fp_ner": fps_ok, "texto_extra": extra_ner,
            }

        ruta_t = salida / f"transcripciones_simulado_{nivel}.json"
        ruta_n = salida / f"ner_simulado_{nivel}.json"
        ruta_t.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        ruta_n.write_text(json.dumps(ner, ensure_ascii=False, indent=1), encoding="utf-8")
        entradas.append({"transcripciones": str(ruta_t), "ner": str(ruta_n), "nombre": "simulado"})
        verdad[nivel] = verdad_nivel

    (salida / "verdad_simulacion.json").write_text(json.dumps(verdad, ensure_ascii=False, indent=1), encoding="utf-8")
    return entradas


def _registro(doc_id: str, k: int, texto: str, valvula: bool, info: dict) -> dict:
    return {"doc_id": f"{doc_id}__p{k + 1}", "texto": texto, "backend": "simulado", "segundos": 0.0,
            "valvula": valvula, "error": None, "imagen": info["paginas"][k]["archivo"], "sha256": ""}
