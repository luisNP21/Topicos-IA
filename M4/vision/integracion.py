"""Enchufe del componente visual al sistema (M4 · Agustín).

Patrón elegido: **input**. La imagen se transcribe ANTES del RAG y el texto
entra al sistema de M3 como contexto. La justificación formal del patrón es la
Tarea 2 de Pau; aquí solo se implementa el enchufe.

`responder_m3` es inyectable:
- `responder_m3_sustituto`: marcado, para correr sin M3/Groq.
- `crear_responder_m3(...)`: adaptador real que reutiliza retrieval,
  normalización y generación de M3. Requiere `PROJECT_ROOT`, el índice de M3 y
  `GROQ_API_KEY`.

El sistema completo (app de Pau) puede inyectar su propio `responder_m3` y su
extractor de entidades (NER de M1/M2) sin reescribir esta pieza.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from contratos import ExtractorEntidades, ResponderM3
from transcripcion import cargar_imagen, ver


def responder_m3_sustituto(pregunta: str, contexto_extra: str = "") -> str:
    """Sustituto MARCADO del sistema de M3 (mismo rol que el del lab S13)."""
    return (
        "[SUSTITUTO M3] Pregunta: "
        f"{pregunta}. Contexto visual: {str(contexto_extra)[:300]}"
    )


def _parsear_contexto(contexto_extra) -> tuple[str, list[str]]:
    """Devuelve (texto, entidades) desde un string o un dict."""
    if isinstance(contexto_extra, dict):
        texto = str(contexto_extra.get("texto") or "")
        entidades = contexto_extra.get("entidades") or []
        return texto, [str(e) for e in entidades if str(e).strip()]

    texto = str(contexto_extra or "")
    if texto.strip().startswith("{"):
        try:
            obj = json.loads(texto)
            if isinstance(obj, dict):
                entidades = obj.get("entidades") or []
                return (
                    str(obj.get("texto") or ""),
                    [str(e) for e in entidades if str(e).strip()],
                )
        except json.JSONDecodeError:
            pass
    return texto, []


def _entidades_desde_contexto(
    contexto_extra,
    extractor: ExtractorEntidades | None,
    cfg: dict,
) -> list[str]:
    """Resuelve las entidades que alimentan el RAG de M3.

    Orden:
    1. `entidades` explícitas en el contexto (JSON del eval set o del app).
    2. `extractor` inyectado (NER de M1/M2).
    3. `integracion.entidad` del YAML (para correr un caso puntual).
    4. Si el contexto es corto y de una línea, se usa como entidad
       (útil para `--entidad DM2`); una nota larga NO se usa como consulta.
    """
    texto, entidades = _parsear_contexto(contexto_extra)
    if entidades:
        return entidades

    if extractor is not None:
        extraidas = extractor(texto)
        if extraidas:
            return [str(e) for e in extraidas if str(e).strip()]

    entidad_cfg = (cfg.get("integracion") or {}).get("entidad")
    if entidad_cfg:
        return [str(entidad_cfg)]

    limpio = texto.strip()
    if limpio and len(limpio) <= 80 and "\n" not in limpio:
        return [limpio]
    return []


def _cargar_cfg_m3(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def crear_responder_m3(
    cfg: dict | None = None,
    project_root: str | Path | None = None,
    extractor: ExtractorEntidades | None = None,
) -> ResponderM3:
    """Construye el `responder_m3(pregunta, contexto_extra)` real o sustituto.

    Si `integracion.usar_mocks` es True (o no hay `project_root`), devuelve el
    sustituto marcado. Si es False, carga las piezas reales de M3 una sola vez
    y las reutiliza en cada caso.
    """
    cfg = cfg or {}
    integracion = cfg.get("integracion") or {}

    if integracion.get("usar_mocks", True) or project_root is None:
        return responder_m3_sustituto

    project_root = Path(project_root)
    os.environ.setdefault("PROJECT_ROOT", str(project_root))

    generar_dir = project_root / "M3" / "generacion"
    if str(generar_dir) not in sys.path:
        sys.path.insert(0, str(generar_dir))

    try:
        from generacion import GroqGenerator, generar_respuesta
        from orquestacion import resolver_query
        from run_generacion import _piezas_reales
    except Exception as exc:  # pragma: no cover - depende del repo/entorno
        raise RuntimeError(
            "No se pudieron importar las piezas reales de M3 "
            "(M3/generacion, M3/retrieval, M3/tools). Verificar que el repo "
            f"esté completo y las dependencias instaladas: {exc}"
        ) from exc

    m3_cfg_path = project_root / integracion.get(
        "m3_generacion_config", "M3/generacion/config.yaml"
    )
    if not m3_cfg_path.exists():
        raise FileNotFoundError(f"No existe el config de M3: {m3_cfg_path}")

    try:
        cfg_m3 = _cargar_cfg_m3(m3_cfg_path)
        retrieve_fn, normalizar_fn, params, pregunta_fn = _piezas_reales(
            cfg_m3, project_root
        )
    except Exception as exc:  # pragma: no cover - depende del índice/Drive
        raise RuntimeError(
            "No se pudo inicializar el pipeline real de M3 (índice Chroma, "
            f"retrieval o tool de normalización): {exc}"
        ) from exc

    gen_cfg = cfg_m3.get("generacion", {})
    try:
        generar_fn = GroqGenerator(
            model=gen_cfg.get("modelo", "qwen/qwen3.8-27b"),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            max_tokens=int(gen_cfg.get("max_tokens", 700)),
            pausa_entre_llamadas=float(gen_cfg.get("pausa_entre_llamadas", 3.0)),
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo crear el generador de M3. Verificar `groq` y "
            f"GROQ_API_KEY en el entorno: {exc}"
        ) from exc
    max_entidades = int(integracion.get("max_entidades", 3))

    def responder(pregunta: str, contexto_extra: str = "") -> str:
        entidades = _entidades_desde_contexto(contexto_extra, extractor, cfg)
        if not entidades:
            return (
                "No pude identificar una entidad de enfermedad en el texto "
                "transcrito. Revisa la imagen o inyecta el extractor NER de M1/M2."
            )

        partes: list[str] = []
        for entidad in entidades[:max_entidades]:
            try:
                resuelta = resolver_query(
                    entidad,
                    retrieve_fn=retrieve_fn,
                    normalizar_fn=normalizar_fn,
                    umbral=params["umbral"],
                    k=params["k"],
                    umbral_evidencia=params.get("umbral_evidencia"),
                    forzar_por_sigla=params.get("forzar_por_sigla", False),
                )
                # Si el caso no trae pregunta, se usa la misma pregunta de
                # intención que alimenta el reranker de M3.
                respuesta = generar_respuesta(
                    entidad,
                    resuelta["fragments"],
                    generar_fn=generar_fn,
                    pregunta=pregunta or pregunta_fn(entidad),
                )
                etiqueta = f"[{entidad}] "
                if respuesta["fallback_used"]:
                    etiqueta += "(sin evidencia suficiente) "
                partes.append(etiqueta + respuesta["answer"])
            except Exception as exc:
                # Fallo explícito por entidad; no se oculta ni se inventa.
                partes.append(f"[{entidad}] Error al consultar M3: {type(exc).__name__}: {exc}")

        return "\n\n".join(partes)

    return responder


def sistema_multimodal(
    pregunta: str,
    imagen=None,
    *,
    entidad: str | None = None,
    responder_m3: ResponderM3 | None = None,
    extractor: ExtractorEntidades | None = None,
    backend=None,
    cfg: dict | None = None,
    project_root: str | Path | None = None,
    instruccion: str | None = None,
):
    """Pregunta (+ imagen) -> respuesta. Enchufe del patrón *input*.

    - Sin imagen: delega en `responder_m3(pregunta)` (el sistema de M3 intacto).
    - Con imagen: `ver()` la transcribe y el texto se suma al contexto del RAG.
    - `entidad`: si el eval set ya la trae, se pasa explícita para no depender
      del extractor NER en la demo.
    """
    cfg = cfg or {}
    if responder_m3 is None:
        responder_m3 = crear_responder_m3(cfg, project_root, extractor=extractor)

    if isinstance(imagen, (str, Path)):
        imagen = cargar_imagen(imagen)

    if imagen is None:
        if entidad:
            contexto = json.dumps(
                {"texto": "", "entidades": [entidad]}, ensure_ascii=False
            )
            return responder_m3(pregunta, contexto_extra=contexto)
        return responder_m3(pregunta)

    if isinstance(backend, str):
        texto, _meta = ver(
            imagen,
            backend_nombre=backend,
            instruccion=instruccion,
            cfg=cfg,
        )
    else:
        texto, _meta = ver(
            imagen,
            backend=backend,
            instruccion=instruccion,
            cfg=cfg,
        )

    if entidad:
        contexto = json.dumps(
            {"texto": texto, "entidades": [entidad]}, ensure_ascii=False
        )
    else:
        contexto = texto
    return responder_m3(pregunta, contexto_extra=contexto)
