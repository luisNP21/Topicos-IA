"""Que texto del gold sigue visible en la imagen final (tarea 3a).

Un token (palabra) es visible si sobrevive al menos `umbral` de sus puntos de muestra.
Una entidad del gold se localiza en el texto original (todas sus apariciones) y queda:
- visible: alguna aparicion tiene todas sus palabras visibles,
- parcial: ninguna completa, pero alguna con palabras visibles,
- oculta: todas sus palabras quedaron fuera de cuadro o tapadas,
- no_localizada: la cadena no aparece en el texto (no se puede evaluar).

Esto separa "el VLM leyo mal" de "el texto no estaba en la imagen". La legibilidad
(desenfoque, baja resolucion, letra dificil) NO cuenta como texto oculto: es lo que se mide.
"""

from __future__ import annotations

import numpy as np


def fraccion_visible_por_token(n_tokens: int, pt_token: list[np.ndarray], vivo: list[np.ndarray]) -> np.ndarray:
    total = np.zeros(n_tokens, np.float32)
    vivos = np.zeros(n_tokens, np.float32)
    for toks, v in zip(pt_token, vivo):
        if len(toks):
            np.add.at(total, toks, 1)
            np.add.at(vivos, toks, v.astype(np.float32))
    frac = np.ones(n_tokens, np.float32)
    con_puntos = total > 0
    frac[con_puntos] = vivos[con_puntos] / total[con_puntos]
    return frac


def _apariciones(texto: str, entidad: str) -> list[tuple[int, int]]:
    spans = []
    for hay, aguja in ((texto, entidad), (texto.lower(), entidad.lower())):
        ini = hay.find(aguja)
        while ini != -1 and aguja:
            spans.append((ini, ini + len(aguja)))
            ini = hay.find(aguja, ini + 1)
        if spans:
            break
    return spans


def estado_entidades(texto: str, entidades: list[str], tokens: list[dict], visibles: np.ndarray) -> dict:
    inis = np.array([t["ini"] for t in tokens])
    fines = np.array([t["fin"] for t in tokens])
    detalle, conteo = [], {"visible": 0, "parcial": 0, "oculta": 0, "no_localizada": 0}
    for ent in dict.fromkeys(entidades):
        spans = _apariciones(texto, ent)
        if not spans:
            estado = "no_localizada"
        else:
            mejor = "oculta"
            for a, b in spans:
                idx = np.where((inis < b) & (fines > a))[0]
                if len(idx) and visibles[idx].all():
                    mejor = "visible"
                    break
                if len(idx) and visibles[idx].any():
                    mejor = "parcial"
            estado = mejor
        conteo[estado] += 1
        if estado != "visible":
            detalle.append({"entidad": ent, "estado": estado})
    return {"total": len(dict.fromkeys(entidades)), **conteo, "no_visibles": detalle}
