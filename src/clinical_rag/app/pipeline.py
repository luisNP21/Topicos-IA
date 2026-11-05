"""
Pipeline del punto de entrada de M4: imagen o texto -> entidades + fragmentos de guía.

Reutiliza las piezas reales del equipo, sin reimplementarlas:
  - Visión ........ M4/vision/transcripcion.py y backends.py (VLM Qwen2-VL u OCR)
  - NER ........... M2/run_inference (Clinical BERT + LoRA de M1, mismo chunking que M2)
  - Retrieval ..... M3/retrieval (índice Chroma + híbrida + reranker, modo 'intencion')
  - Orquestación .. M3/generacion/orquestacion.py (cuándo se normaliza, compuerta de evidencia)
  - Generación .... M3/generacion/generacion.py (Groq, válvula de escape si no hay evidencia)

Uso como librería (sin interfaz):
    from clinical_rag.app.pipeline import PipelineClinico
    p = PipelineClinico(); p.cargar()
    r = p.analizar(texto="Paciente con diabetes mellitus tipo 2 ...")
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import yaml

from clinical_rag.config import load_yaml
from clinical_rag.generation.generate import GroqGenerator, generar_respuesta
from clinical_rag.generation.orchestration import resolver_query
from clinical_rag.paths import CONFIG_DIR, REPO_ROOT, project_root as default_project_root, resolve_project
from clinical_rag.retrieval import index as retrieval
from clinical_rag.retrieval.config import (
    cargar_config as cargar_retrieval_config,
    configurar_pipeline,
    normalizador_desde_config,
    parametros_orquestacion,
)
from clinical_rag.vision import backends as vision_backends
from clinical_rag.vision import prompts as vision_prompts
from clinical_rag.vision import transcription as vision_transcription

APP_DIR = Path(__file__).resolve().parent
REPO = REPO_ROOT
CONFIG_APP = CONFIG_DIR / "app.yaml"

Progreso = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[app {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Entorno y recursos
# ---------------------------------------------------------------------------
def cargar_entorno() -> Path:
    """Carga el .env (raíz del repo o carpeta actual) y devuelve PROJECT_ROOT validado."""
    from dotenv import find_dotenv, load_dotenv

    if (REPO / ".env").exists():
        load_dotenv(REPO / ".env")
    load_dotenv(find_dotenv(usecwd=True))

    return default_project_root()


def verificar_recursos(cfg: dict, project_root: Path) -> None:
    """Falla con una lista clara de lo que falta, antes de cargar modelos pesados."""
    r = cfg["rutas"]
    faltan = []
    adapter = resolve_project(r["adapter_ner"])
    if not (adapter / "adapter_config.json").exists():
        faltan.append(f"Adaptador LoRA del NER: {adapter} (debe tener adapter_config.json)")
    chunks = resolve_project(r["corpus_chunks"])
    if not chunks.is_dir() or not any(chunks.glob("*.json")):
        faltan.append(f"Chunks del corpus (un .json por chunk): {chunks}")
    manifest = resolve_project(r["corpus_manifest"])
    if not manifest.exists():
        faltan.append(f"Manifiesto del corpus: {manifest}")
    if faltan:
        raise FileNotFoundError(
            "Faltan recursos dentro de PROJECT_ROOT. Cópialos desde Drive "
            "(MyDrive/TopicosIA/Proyecto-Salud) a la misma ruta relativa:\n  - "
            + "\n  - ".join(faltan)
        )


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------
class ExtractorNER:
    """Texto -> entidades de enfermedad, en orden de aparición y sin repetir.

    Usa las mismas funciones que M2 (predict_word_level, chunk_words, normalizar_entidad).
    La única diferencia con `bio_to_entity_set` de M2 es que conserva el orden, para que
    la app consulte primero las entidades que aparecen primero en la nota.
    """

    def __init__(self, cfg_ner: dict, adapter_path: Path, project_root: Path):
        from clinical_rag.common import chunk_words, normalizar_entidad
        from clinical_rag.ner.inference import predict_word_level
        from clinical_rag.ner.model import cargar_modelo

        self._chunk_words = chunk_words
        self._normalizar = normalizar_entidad
        self._predecir = predict_word_level
        self.window = int(cfg_ner["window_words"])
        self.overlap = int(cfg_ner["overlap_words"])
        self.model, self.tokenizer, self.id2label = cargar_modelo(
            {"checkpoint": cfg_ner["checkpoint"], "adapter_path": str(adapter_path)}, None
        )

    def __call__(self, texto: str) -> list[str]:
        palabras = texto.split()
        if not palabras:
            return []
        menciones: list[str] = []
        for _, trozo in self._chunk_words(palabras, self.window, self.overlap):
            etiquetas = self._predecir(trozo, self.model, self.tokenizer, self.id2label)
            actual: list[str] = []
            for tok, tag in zip(trozo, etiquetas):
                if tag == "B-ENFERMEDAD":
                    if actual:
                        menciones.append(" ".join(actual))
                    actual = [tok]
                elif tag == "I-ENFERMEDAD" and actual:
                    actual.append(tok)
                else:
                    if actual:
                        menciones.append(" ".join(actual))
                    actual = []
            if actual:
                menciones.append(" ".join(actual))

        vistas, salida = set(), []
        for m in menciones:
            n = self._normalizar(m)
            if n and n not in vistas:
                vistas.add(n)
                salida.append(n)
        return salida


class Vision:
    """Imágenes -> texto con los backends de M4/vision. Cada backend se carga una sola vez."""

    def __init__(self, cfg_vision_app: dict):
        self.trans = vision_transcription
        self.backends = vision_backends
        self.prompts = vision_prompts
        self.cfg = load_yaml("configs/vision.yaml")
        self.cfg.setdefault("vlm", {})["max_new_tokens"] = int(cfg_vision_app["vlm_max_new_tokens"])
        self._cache: dict[str, object] = {}
        self.mensaje_sin_texto = self.prompts.MENSAJE_SIN_TEXTO

    @staticmethod
    def hay_gpu() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def resolver_backend(self, nombre: str) -> str:
        if nombre in (None, "", "auto"):
            return "vlm" if self.hay_gpu() else "ocr_docling"
        return nombre

    def _backend(self, nombre: str):
        if nombre not in self._cache:
            self._cache[nombre] = self.backends.crear_backend(nombre, self.cfg)
        return self._cache[nombre]

    def transcribir(self, rutas: list[str], backend: str) -> dict:
        nombre = self.resolver_backend(backend)
        motor = self._backend(nombre)
        paginas = []
        for i, ruta in enumerate(rutas, 1):
            imagen = self.trans.cargar_imagen(ruta)
            t = self.trans.transcribir(imagen, backend=motor, ruta=str(ruta), doc_id=f"p{i}")
            paginas.append(t)
        legibles = [p["texto"] for p in paginas if not p["valvula"] and not p["error"]]
        return {"backend": nombre, "texto": "\n".join(legibles), "paginas": paginas}


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
class PipelineClinico:
    def __init__(self, config_path: Path | str = CONFIG_APP):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.project_root: Path | None = None
        self.vision: Vision | None = None
        self._retr = retrieval
        self.ner: ExtractorNER | None = None
        self.generar_fn = None
        self.info: dict = {}
        self.avisos_carga: list[str] = []

    # -- carga ---------------------------------------------------------------
    def cargar(self, progreso: Progreso = _log) -> "PipelineClinico":
        self.project_root = cargar_entorno()
        progreso(f"PROJECT_ROOT: {self.project_root}")
        verificar_recursos(self.cfg, self.project_root)

        progreso("Preparando visión...")
        self.vision = Vision(self.cfg["vision"])

        progreso("Preparando índice de guías (la 1.ª vez descarga modelos y reconstruye Chroma)...")
        ruta_cfg, origen_umbrales = self._config_retrieval_app()
        os.environ["M3_RETRIEVAL_CONFIG"] = str(ruta_cfg)
        cfg_retr = cargar_retrieval_config(ruta_cfg)
        indice = configurar_pipeline(cfg_retr)
        self.params = parametros_orquestacion(cfg_retr)
        self.normalizar_fn = normalizador_desde_config(cfg_retr)
        self._retr = retrieval

        progreso("Cargando NER (Clinical BERT + LoRA de M1)...")
        self.ner = ExtractorNER(
            self.cfg["ner"],
            resolve_project(self.cfg["rutas"]["adapter_ner"]),
            self.project_root,
        )

        gen_cfg = load_yaml("configs/generation.yaml")["generacion"]
        try:
            self.generar_fn = GroqGenerator(
                model=gen_cfg["modelo"], temperature=float(gen_cfg["temperature"]),
                max_tokens=int(gen_cfg["max_tokens"]),
                pausa_entre_llamadas=float(gen_cfg.get("pausa_entre_llamadas", 3.0)),
            )
        except Exception as exc:
            self.generar_fn = None
            self.avisos_carga.append(
                f"Sin generación con LLM ({exc}). Se muestran entidades y fragmentos, sin respuesta redactada."
            )

        self.info = {
            "n_chunks": len(indice.chunks),
            "guias": sorted({indice.manifest.get(c["doc_id"], {}).get("titulo") or c["doc_id"]
                             for c in indice.chunks}),
            "umbral": self.params["umbral"],
            "umbral_evidencia": self.params.get("umbral_evidencia"),
            "origen_umbrales": origen_umbrales,
            "modelo_llm": gen_cfg["modelo"] if self.generar_fn else None,
            "gpu": Vision.hay_gpu(),
            "backend_por_defecto": self.vision.resolver_backend(self.cfg["vision"]["backend_por_defecto"]),
        }
        for a in self.avisos_carga:
            progreso("AVISO: " + a)
        progreso(f"Listo: {self.info['n_chunks']} chunks de {len(self.info['guias'])} guías | "
                 f"umbrales {self.info['umbral']}/{self.info['umbral_evidencia']} ({origen_umbrales}) | "
                 f"visión por defecto: {self.info['backend_por_defecto']}")
        return self

    def _config_retrieval_app(self) -> tuple[Path, str]:
        """Copia de config_retrieval.yaml con rutas locales (el original tiene rutas de Colab)."""
        cfg = load_yaml("configs/retrieval.yaml")
        r = self.cfg["rutas"]
        cfg["corpus"] = "real"
        real = cfg["rutas"]["real"]
        real["corpus_json"] = str(resolve_project(r["corpus_chunks"]))
        real["manifest"] = str(resolve_project(r["corpus_manifest"]))
        real["chroma_dir"] = str(resolve_project(r["chroma_local"]))

        u = self.cfg["umbrales"]
        umbrales_json = resolve_project(real["salida"]) / "umbrales.json"
        if u.get("preferir_umbrales_json", True) and umbrales_json.exists():
            cfg["orquestacion"]["umbral"] = None
            cfg["orquestacion"]["umbral_evidencia"] = None
            origen = f"calibrados en {umbrales_json.name}"
        else:
            cfg["orquestacion"]["umbral"] = float(u["umbral"])
            cfg["orquestacion"]["umbral_evidencia"] = float(u["umbral_evidencia"])
            origen = "config_app.yaml"

        destino = resolve_project(r["cache_dir"]) / "config_retrieval_app.yaml"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return destino, origen

    # -- pasos (la interfaz los llama uno a uno para mostrar el avance) --------
    def transcribir(self, imagenes: list[str], backend: str = "auto") -> dict:
        return self.vision.transcribir(imagenes, backend)

    def extraer_entidades(self, texto: str) -> tuple[list[str], str]:
        """Devuelve (entidades, origen). Una consulta corta sin entidades se usa tal cual."""
        entidades = self.ner(texto)
        if entidades:
            return entidades, "ner"
        limpio = texto.strip()
        if limpio and len(limpio) <= 80 and "\n" not in limpio:
            return [limpio.lower()], "consulta_directa"
        return [], "ner"

    def buscar(self, entidad: str) -> dict:
        """Una entidad -> orquestación de M3 (retrieval + normalización + compuerta) -> fragmentos.

        No llama al LLM: es rápido (~0,35 s) y se hace para todas las entidades, para poder
        elegir la enfermedad principal del caso antes de redactar."""
        inicio = time.perf_counter()
        q = resolver_query(
            entidad, retrieve_fn=self._retr.retrieve_advanced, normalizar_fn=self.normalizar_fn,
            umbral=self.params["umbral"], k=self.params["k"],
            umbral_evidencia=self.params.get("umbral_evidencia"),
            forzar_por_sigla=self.params.get("forzar_por_sigla", False),
        )
        fragmentos = [{
            "chunk_id": f.get("chunk_id"), "fuente": f.get("fuente") or f.get("doc_id"),
            "seccion": f.get("seccion"), "score": f.get("score"), "score_tipo": f.get("score_tipo"),
            "texto": f.get("texto", ""),
        } for f in q["fragments"]]
        return {
            "entidad": entidad, "query_final": q["query_final"],
            "tool_invoked": q["tool_invoked"], "tool_reason": q["tool_reason"],
            "fragmentos": fragmentos, "_fragments_m3": q["fragments"],
            "respuesta": None, "fallback": not q["fragments"], "fuentes_usadas": [],
            "pregunta": None, "error": None,
            "segundos": round(time.perf_counter() - inicio, 2),
        }

    @staticmethod
    def elegir_principal(resultados: list[dict]) -> dict | None:
        """La enfermedad principal del caso: la entidad con más evidencia en las guías.

        Criterio determinista (como la orquestación de M3, sin pedirle al LLM que decida):
        entre las entidades con al menos un fragmento sobre el umbral de evidencia, la de
        mayor score del reranker en su primer fragmento. En empate gana la que aparece
        primero en la nota. Si ninguna tiene evidencia, no hay principal (válvula de escape).
        Ojo: es la enfermedad sobre la que las guías pueden responder mejor, no un diagnóstico
        clínico principal decidido por el sistema."""
        con_evidencia = [r for r in resultados if r["fragmentos"]]
        if not con_evidencia:
            return None
        return max(con_evidencia, key=lambda r: max(
            (f["score"] for f in r["fragmentos"] if isinstance(f.get("score"), (int, float))), default=0.0))

    @staticmethod
    def extraer_pregunta(texto: str) -> str | None:
        """Si el usuario escribió una pregunta ("¿...?"), se usa para la generación, como la
        columna `pregunta` del eval set de M3. Si no, se usa la pregunta de intención."""
        import re
        preguntas = re.findall(r"¿[^?¿]{5,}\?", texto or "")
        return preguntas[-1].strip() if preguntas else None

    def redactar(self, resultado: dict, pregunta: str | None = None) -> dict:
        """Genera la respuesta con el LLM de M3 sobre los fragmentos de una entidad."""
        inicio = time.perf_counter()
        pregunta = pregunta or retrieval.pregunta_intencion(resultado["query_final"])
        resultado["pregunta"] = pregunta
        if self.generar_fn is None and resultado["_fragments_m3"]:
            return resultado                  # sin GROQ_API_KEY: solo fragmentos
        try:
            resp = generar_respuesta(
                resultado["query_final"], resultado["_fragments_m3"], generar_fn=self.generar_fn,
                pregunta=pregunta,
            )
            resultado.update(respuesta=resp["answer"], fallback=resp["fallback_used"],
                             fuentes_usadas=resp["sources_used"])
        except Exception as exc:              # error explícito, no se inventa respuesta
            resultado["error"] = f"{type(exc).__name__}: {exc}"
        resultado["segundos"] = round(resultado["segundos"] + time.perf_counter() - inicio, 2)
        return resultado

    def consultar(self, entidad: str, pregunta: str | None = None) -> dict:
        """Buscar + redactar para una sola entidad (uso como librería)."""
        return self.redactar(self.buscar(entidad), pregunta)

    # -- todo junto (uso como librería) ----------------------------------------
    def analizar(self, texto: str = "", imagenes: list[str] | None = None, backend: str = "auto",
                 max_entidades: int | None = None, pregunta: str | None = None) -> dict:
        """Nota (texto y/o imágenes) -> entidades -> una enfermedad principal con su respuesta."""
        imagenes = imagenes or []
        transcripcion = self.transcribir(imagenes, backend) if imagenes else None
        partes = [transcripcion["texto"]] if transcripcion else []
        if texto.strip():
            partes.append(texto.strip())
        texto_total = "\n".join(p for p in partes if p)

        entidades, origen = self.extraer_entidades(texto_total) if texto_total else ([], "ner")
        n = int(max_entidades or self.cfg["rag"]["max_entidades"])
        resultados = [self.buscar(e) for e in entidades[:n]]
        principal = self.elegir_principal(resultados)
        if principal is not None:
            self.redactar(principal, pregunta or self.extraer_pregunta(texto))
        return {
            "transcripcion": transcripcion, "texto": texto_total,
            "entidades": entidades, "origen_entidades": origen,
            "principal": principal, "resultados": resultados,
            "no_consultadas": entidades[n:],
        }

