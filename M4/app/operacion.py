"""
Operación de la app de M4: versión de Git, logs por consulta y health check.

Sugerencias de la revisión de M4 para las apps:
  - "Versión con tag de Git visible en la app"  -> version_git()
  - "Health check y logs por consulta: entrada, ruta, confianza y latencia"
                                               -> RegistroConsultas / Traza / salud()
  - "Versiones fijadas (requirements.txt)"     -> congelar_requirements()

Privacidad: el log NO guarda el texto de la nota ni las imágenes, solo su tamaño y un
hash (para reconocer la misma entrada repetida). Se puede cambiar en config_app.yaml
(logs.guardar_texto), pero en la demo no se deben ingresar datos reales de pacientes.

Uso por línea de comandos (desde la raíz del repo):
    python M4/app/operacion.py --version          # versión de Git que mostraría la app
    python M4/app/operacion.py --requirements     # requirements.txt fijado con el entorno actual
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parents[1]
INICIO_PROCESO = time.time()


# ---------------------------------------------------------------------------
# Versión de Git
# ---------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        salida = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                timeout=5, check=True)
        return salida.stdout.strip() or None
    except Exception:
        return None


_VERSION: dict | None = None


def version_git() -> dict:
    """{'version', 'commit', 'rama', 'tag', 'cambios_sin_commit', 'fecha_commit'}.

    `version` es lo que se muestra: el tag si el commit tiene uno (p. ej. v1.0-demo),
    si no `git describe` (tag-N-gcommit o el commit corto). `-dirty` = hay cambios
    locales sin commit, o sea que lo que corre no es exactamente lo publicado.
    Si no hay carpeta .git (p. ej. un Space que copia solo los archivos), se usa la
    variable de entorno APP_VERSION."""
    global _VERSION
    if _VERSION is None:
        describe = _git("describe", "--tags", "--always", "--dirty")
        _VERSION = {
            "version": describe or os.environ.get("APP_VERSION") or "desconocida",
            "commit": _git("rev-parse", "--short", "HEAD"),
            "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "tag": _git("describe", "--tags", "--exact-match"),
            "cambios_sin_commit": bool(describe and describe.endswith("-dirty")),
            "fecha_commit": _git("log", "-1", "--format=%cs"),
        }
    return _VERSION


def texto_version() -> str:
    v = version_git()
    partes = [f"versión `{v['version']}`"]
    if v["rama"] and v["rama"] != "HEAD":
        partes.append(f"rama `{v['rama']}`")
    if v["fecha_commit"]:
        partes.append(f"commit del {v['fecha_commit']}")
    return " · ".join(partes)


# ---------------------------------------------------------------------------
# Versiones de paquetes y requirements fijado
# ---------------------------------------------------------------------------
# Dependencias directas de la app y de las piezas que llama (NER de M2, visión de M4,
# retrieval y generación de M3). torch NO se fija: viene preinstalado con CUDA en
# Colab/Kaggle y reinstalarlo puede romper la GPU.
PAQUETES_APP = [
    "gradio", "transformers", "accelerate", "peft", "torchao", "qwen-vl-utils",
    "sentence-transformers", "chromadb", "rank-bm25", "groq", "docling", "pytesseract",
    "Pillow", "numpy", "PyYAML", "python-dotenv", "requests",
]
NO_FIJAR = ["torch"]


def versiones_paquetes(nombres: list[str] | None = None) -> dict[str, str | None]:
    salida = {}
    for n in (nombres or PAQUETES_APP + NO_FIJAR):
        try:
            salida[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            salida[n] = None
    return salida


def congelar_requirements() -> str:
    """requirements.txt con `==` para cada dependencia directa, según el entorno actual.

    Se corre UNA vez en el entorno donde la app funcionó de punta a punta (la sesión de
    Kaggle/Colab de la demo) y se versiona el resultado."""
    v = versiones_paquetes()
    lineas = [
        "# Dependencias de la app de M4, FIJADAS con el entorno donde se probó de punta a punta.",
        f"# Generado con: python M4/app/operacion.py --requirements  ({datetime.now():%Y-%m-%d})",
        "# Instalar desde la raíz del repo:  pip install -r M4/app/requirements.txt",
        f"# torch no se fija: se usa el preinstalado con CUDA (versión al generar: {v.get('torch') or 'no instalado'}).",
        "",
    ]
    faltan = []
    for n in PAQUETES_APP:
        if v.get(n):
            lineas.append(f"{n}=={v[n]}")
        else:
            faltan.append(n)
    if faltan:
        lineas += ["", "# No instalados en el entorno al generar (revisar antes de usar):"]
        lineas += [f"# {n}" for n in faltan]
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------------------
# Logs por consulta
# ---------------------------------------------------------------------------
def _hash(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()[:16]


def _hash_archivo(ruta: str) -> str | None:
    try:
        return _hash(Path(ruta).read_bytes())
    except OSError:
        return None


class Traza:
    """Acumula lo que pasa en UNA consulta: entrada, ruta, confianza y latencia por etapa."""

    def __init__(self, texto: str, imagenes: list[str], backend: str, max_entidades: int,
                 guardar_texto: bool = False):
        self._t0 = time.perf_counter()
        self.resultado: str | None = None      # lo fija la app al terminar cada camino
        self.registro: dict = {
            "id": uuid.uuid4().hex[:12],
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": version_git()["version"],
            "entrada": {
                "tipo": ("imagen+texto" if imagenes and texto else "imagen" if imagenes else "texto"),
                "n_caracteres_texto": len(texto),
                "hash_texto": _hash(texto.encode("utf-8")) if texto else None,
                "n_imagenes": len(imagenes),
                "hash_imagenes": [_hash_archivo(r) for r in imagenes],
                "backend_pedido": backend,
                "max_entidades": max_entidades,
            },
            "ruta": {},
            "confianza": {},
            "latencia_s": {},
            "resultado": None,
            "error": None,
        }
        if guardar_texto:
            self.registro["entrada"]["texto"] = texto

    @contextmanager
    def etapa(self, nombre: str):
        inicio = time.perf_counter()
        try:
            yield
        finally:
            lat = self.registro["latencia_s"]
            lat[nombre] = round(lat.get(nombre, 0.0) + time.perf_counter() - inicio, 3)

    def vision(self, transcripcion: dict) -> None:
        paginas = transcripcion.get("paginas", [])
        self.registro["ruta"]["vision"] = {
            "backend": transcripcion.get("backend"),
            "paginas": [{"segundos": p.get("segundos"), "valvula": p.get("valvula"),
                         "error": p.get("error")} for p in paginas],
            "n_caracteres_leidos": len(transcripcion.get("texto") or ""),
        }

    def entidades(self, entidades: list[str], origen: str, consultadas: int) -> None:
        self.registro["ruta"].update(n_entidades=len(entidades), origen_entidades=origen,
                                     n_consultadas=consultadas)

    def principal(self, resultados: list[dict], principal: dict | None,
                  umbral_evidencia: float | None) -> None:
        con_evidencia = [r for r in resultados if r["fragmentos"]]
        self.registro["ruta"]["n_con_evidencia"] = len(con_evidencia)
        self.registro["ruta"]["normalizadas"] = sum(1 for r in resultados if r.get("tool_invoked"))
        conf = self.registro["confianza"]
        conf["umbral_evidencia"] = umbral_evidencia
        if principal is None:
            conf["score_principal"] = None
            return
        scores = [f["score"] for f in principal["fragmentos"] if isinstance(f.get("score"), (int, float))]
        self.registro["ruta"].update(
            principal=principal["query_final"],
            normalizada=principal["query_final"] != principal["entidad"],
            chunk_ids=[f["chunk_id"] for f in principal["fragmentos"]],
        )
        conf.update(score_principal=round(max(scores), 4) if scores else None,
                    n_fragmentos=len(principal["fragmentos"]),
                    fallback=principal.get("fallback"),
                    error_llm=principal.get("error"))

    def terminar(self, resultado: str, error: str | None = None) -> dict:
        self.registro["resultado"] = resultado
        self.registro["error"] = error
        self.registro["latencia_s"]["total"] = round(time.perf_counter() - self._t0, 3)
        return self.registro


class RegistroConsultas:
    """Escribe una línea JSON por consulta (JSONL) y un resumen en la consola."""

    def __init__(self, carpeta: Path, guardar_texto: bool = False):
        self.carpeta = Path(carpeta)
        self.guardar_texto = guardar_texto
        self.archivo = self.carpeta / "consultas.jsonl"
        self.n = 0
        self.n_errores = 0
        self.ultima: dict | None = None

    def nueva(self, texto: str, imagenes: list[str], backend: str, max_entidades: int) -> Traza:
        return Traza(texto, imagenes, backend, max_entidades, self.guardar_texto)

    def guardar(self, registro: dict) -> None:
        self.n += 1
        self.n_errores += bool(registro.get("error"))
        self.ultima = registro
        try:
            self.carpeta.mkdir(parents=True, exist_ok=True)
            with open(self.archivo, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except OSError as exc:            # un log que falla no debe tumbar la consulta
            print(f"[log] No se pudo escribir {self.archivo}: {exc}", flush=True)
        lat = registro["latencia_s"]
        etapas = ", ".join(f"{k} {v:.2f}s" for k, v in lat.items() if k != "total")
        conf = registro["confianza"].get("score_principal")
        print(f"[consulta {registro['id']}] {registro['entrada']['tipo']} -> {registro['resultado']}"
              f" | principal: {registro['ruta'].get('principal') or '-'}"
              f" | score: {conf if conf is not None else '-'}"
              f" | {lat.get('total', 0):.2f}s ({etapas})"
              + (f" | ERROR: {registro['error']}" if registro.get("error") else ""), flush=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def _gpu() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"disponible": False}
        libre, total = torch.cuda.mem_get_info()
        return {"disponible": True, "nombre": torch.cuda.get_device_name(0),
                "memoria_libre_gb": round(libre / 1e9, 2), "memoria_total_gb": round(total / 1e9, 2)}
    except Exception as exc:
        return {"disponible": False, "error": str(exc)}


def salud(pipeline, registro: RegistroConsultas | None = None, profundo: bool = False) -> dict:
    """Estado de la app: 'ok', 'degradado' (funciona con limitaciones) o 'error'.

    profundo=True además hace una búsqueda real en las guías (~0,4 s, sin LLM) para
    comprobar que el índice y el reranker responden, no solo que están cargados."""
    chequeos: dict[str, dict] = {}

    def chequeo(nombre: str, ok: bool, critico: bool, detalle) -> None:
        chequeos[nombre] = {"ok": bool(ok), "critico": critico, "detalle": detalle}

    info = getattr(pipeline, "info", {}) or {}
    chequeo("indice_guias", info.get("n_chunks", 0) > 0, True,
            f"{info.get('n_chunks', 0)} chunks de {len(info.get('guias', []))} guías")
    chequeo("ner", getattr(pipeline, "ner", None) is not None, True, "Clinical BERT + LoRA")

    vision = getattr(pipeline, "vision", None)
    backend = info.get("backend_por_defecto")
    motor = getattr(vision, "_cache", {}).get(backend) if vision else None
    cargado = motor is not None and (getattr(motor, "_model", True) is not None
                                     and getattr(motor, "_converter", True) is not None)
    chequeo("vision", cargado, False,
            f"motor por defecto '{backend}' " + ("cargado" if cargado else "sin cargar (se carga con la 1.ª imagen)"))
    chequeo("llm", getattr(pipeline, "generar_fn", None) is not None, False,
            info.get("modelo_llm") or "sin GROQ_API_KEY: solo fragmentos, sin respuesta redactada")

    gpu = _gpu()
    chequeo("gpu", gpu.get("disponible", False), False, gpu)

    if profundo and chequeos["indice_guias"]["ok"]:
        try:
            r = pipeline.buscar("diabetes mellitus tipo 2")
            chequeo("busqueda_prueba", bool(r["fragmentos"]), True,
                    f"'diabetes mellitus tipo 2' -> {len(r['fragmentos'])} fragmentos en {r['segundos']} s")
        except Exception as exc:
            chequeo("busqueda_prueba", False, True, f"{type(exc).__name__}: {exc}")

    if any(not c["ok"] for c in chequeos.values() if c["critico"]):
        estado = "error"
    elif any(not c["ok"] for c in chequeos.values()):
        estado = "degradado"
    else:
        estado = "ok"

    resultado = {
        "estado": estado,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version_git(),
        "activa_hace_s": round(time.time() - INICIO_PROCESO),
        "chequeos": chequeos,
        "avisos_carga": list(getattr(pipeline, "avisos_carga", [])),
    }
    if registro is not None:
        resultado["consultas"] = {
            "total": registro.n, "con_error": registro.n_errores,
            "ultima": ({k: registro.ultima[k] for k in ("id", "fecha", "resultado", "error")}
                       | {"latencia_total_s": registro.ultima["latencia_s"].get("total")}
                       if registro.ultima else None),
            "archivo_log": str(registro.archivo),
        }
    return resultado


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Versión y dependencias de la app de M4")
    parser.add_argument("--version", action="store_true", help="Muestra la versión de Git")
    parser.add_argument("--requirements", action="store_true",
                        help="Imprime un requirements.txt fijado con el entorno actual")
    args = parser.parse_args()
    if args.requirements:
        print(congelar_requirements(), end="")
    else:
        print(json.dumps(version_git(), ensure_ascii=False, indent=2))
