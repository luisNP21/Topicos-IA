"""Pruebas de la pieza de visión (M4 · Agustín).

No cargan VLM, Docling ni Tesseract: usan `BackendFake` y mocks. Correr con:

    python M4/vision/test_vision.py
    # o: python -m unittest M4.vision.test_vision -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from backends import BackendFake, BackendOCRTesseract, crear_backend  # noqa: E402
from cache import (  # noqa: E402
    cache_desde_transcripciones,
    cargar_cache_transcripciones,
    guardar_cache_ner,
    guardar_cache_transcripciones,
)
from contratos import Transcripcion  # noqa: E402
from integracion import (  # noqa: E402
    _entidades_desde_contexto,
    crear_responder_m3,
    responder_m3_sustituto,
    sistema_multimodal,
)
from juez import juez_fidelidad_visual  # noqa: E402
from manifiesto import cargar_manifiesto, resolver_ruta_imagen  # noqa: E402
from ner_bridge import generar_cache_ner  # noqa: E402
from prompts import MENSAJE_SIN_TEXTO, es_valvula  # noqa: E402
from run_vision import run  # noqa: E402
from transcripcion import limpiar_salida, transcribir, ver  # noqa: E402


def _imagen(tmp: Path, nombre: str = "nota.png") -> Path:
    ruta = tmp / nombre
    Image.new("RGB", (240, 80), "white").save(ruta)
    return ruta


def _transcripcion(doc_id: str, texto: str = "DM2 e HTA", **extra) -> Transcripcion:
    base: Transcripcion = {
        "doc_id": doc_id,
        "texto": texto,
        "backend": "fake",
        "segundos": 0.1,
        "valvula": False,
        "error": None,
        "imagen": "nota.png",
        "sha256": "abc",
    }
    base.update(extra)
    return base


class TestPrompts(unittest.TestCase):
    def test_valvula_solo_al_inicio(self):
        self.assertTrue(es_valvula(""))
        self.assertTrue(es_valvula("No se aprecia texto legible en la imagen."))
        self.assertTrue(es_valvula("  No se puede leer el documento"))
        self.assertFalse(es_valvula("Paciente con DM2 e HTA."))
        # Una nota real puede mencionar la frase dentro de una oración.
        self.assertFalse(es_valvula("En la nota se dice que no se aprecia el sello."))

    def test_eco_sin_contenido_es_valvula(self):
        # Salida real de Qwen2-VL-2B en el reto de recorte ex_22_p2.
        self.assertTrue(es_valvula("Se aprecia texto legible en la imagen."))
        self.assertTrue(es_valvula("  hay texto legible en la imagen"))
        self.assertTrue(es_valvula("La imagen contiene texto."))

    def test_nota_que_empieza_con_se_aprecia_no_es_valvula(self):
        self.assertFalse(es_valvula("Se aprecia texto legible en la imagen: paciente con DM2."))
        self.assertFalse(es_valvula("Se aprecia edema en miembros inferiores y disnea."))
        self.assertFalse(es_valvula(
            "realizada y alopecia, ambos relacionados con la toxicidad a las medidas de soporte"
        ))

    def test_valvula_con_la_imagen_como_sujeto(self):
        # Salida real de Qwen2-VL-2B en rotacion_recorte_severo/caso_05_p1.
        self.assertTrue(es_valvula("La imagen no aprecia texto legible en la imagen."))
        self.assertTrue(es_valvula("En la imagen no se aprecia texto legible."))
        self.assertTrue(es_valvula("Esta imagen no contiene texto."))
        # Si después viene una lectura, no se descarta.
        self.assertFalse(es_valvula("La imagen no aprecia texto legible en la imagen. Paciente con DM2."))
        self.assertFalse(es_valvula("La imagen muestra una nota: paciente con DM2 e HTA."))


class TestLimpieza(unittest.TestCase):
    def test_quita_markdown_y_pipes(self):
        crudo = "```markdown\n# Nota clínica\n**DM2** | HTA\n```"
        limpio = limpiar_salida(crudo)
        self.assertNotIn("#", limpio)
        self.assertNotIn("**", limpio)
        self.assertNotIn("|", limpio)
        self.assertNotIn("```", limpio)
        self.assertIn("DM2", limpio)

    def test_quita_eco_del_prompt(self):
        instruccion = "Transcribe literalmente el texto visible en la imagen."
        crudo = instruccion + "\nPaciente con EPOC."
        self.assertEqual(limpiar_salida(crudo, instruccion), "Paciente con EPOC.")


class TestBackendFakeYVer(unittest.TestCase):
    def test_ver_devuelve_texto_y_none(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = _imagen(Path(d))
            texto, extra = ver(
                Image.open(ruta),
                backend_nombre="fake",
                cfg={"fake": {"texto": "DM2 e HTA"}},
                ruta=str(ruta),
            )
        self.assertEqual(texto, "DM2 e HTA")
        self.assertIsNone(extra)

    def test_transcribir_conserva_metadatos(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = _imagen(Path(d))
            t = transcribir(
                Image.open(ruta),
                backend_nombre="fake",
                cfg={"fake": {"texto": "DM2 e HTA"}},
                ruta=str(ruta),
                doc_id="doc_1",
            )
        self.assertEqual(t["doc_id"], "doc_1")
        self.assertEqual(t["backend"], "fake")
        self.assertFalse(t["valvula"])
        self.assertIsNone(t["error"])
        self.assertGreaterEqual(t["segundos"], 0)
        self.assertEqual(len(t["sha256"]), 64)

    def test_texto_vacio_activa_valvula(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = _imagen(Path(d))
            t = transcribir(
                Image.open(ruta),
                backend=BackendFake({"texto": "   "}),
                ruta=str(ruta),
                doc_id="caso_1",
            )
        self.assertTrue(t["valvula"])
        self.assertEqual(t["texto"], MENSAJE_SIN_TEXTO)

    def test_error_del_backend_se_registra(self):
        class BackendRoto:
            nombre = "roto"

            def transcribir(self, imagen, *, instruccion=None, ruta=None):
                raise RuntimeError("sin GPU")

        with tempfile.TemporaryDirectory() as d:
            ruta = _imagen(Path(d))
            t = transcribir(Image.open(ruta), backend=BackendRoto(), ruta=str(ruta))
        self.assertTrue(t["valvula"])
        self.assertIn("sin GPU", t["error"])
        self.assertEqual(t["texto"], MENSAJE_SIN_TEXTO)

    def test_crear_backend_desconocido(self):
        with self.assertRaises(ValueError):
            crear_backend("no_existe", {})

    def test_tesseract_configura_cmd_y_tessdata(self):
        fake = types.ModuleType("pytesseract")

        class TesseractNotFoundError(Exception):
            pass

        fake.TesseractNotFoundError = TesseractNotFoundError
        fake.pytesseract = types.SimpleNamespace(tesseract_cmd=None)

        def image_to_string(imagen, lang, config):
            return f"lang={lang};config={config};cmd={fake.pytesseract.tesseract_cmd}"

        fake.image_to_string = image_to_string
        with patch.dict(sys.modules, {"pytesseract": fake}), patch.dict(
            os.environ, {}, clear=False
        ):
            backend = BackendOCRTesseract(
                {
                    "lang": "spa+eng",
                    "psm": 6,
                    "oem": 3,
                    "tesseract_cmd": "C:/Tesseract/tesseract.exe",
                    "tessdata_dir": "C:/tessdata",
                }
            )
            salida = backend.transcribir(Image.new("RGB", (10, 10)))
            self.assertEqual(os.environ.get("TESSDATA_PREFIX"), "C:/tessdata")

        self.assertIn("lang=spa+eng", salida)
        self.assertIn("--psm 6 --oem 3", salida)
        self.assertIn("cmd=C:/Tesseract/tesseract.exe", salida)


class TestCache(unittest.TestCase):
    def test_roundtrip_transcripciones(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "transcripciones.json"
            t = _transcripcion("doc_1")
            guardar_cache_transcripciones([t], path)
            cache = cargar_cache_transcripciones(path)
        self.assertEqual(cache["doc_1"]["texto"], "DM2 e HTA")
        self.assertEqual(cache["doc_1"]["backend"], "fake")

    def test_doc_id_repetido_falla(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.json"
            with self.assertRaises(ValueError):
                guardar_cache_transcripciones(
                    [_transcripcion("x"), _transcripcion("x")], path
                )

    def test_cache_ner_valida_y_ordena(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ner.json"
            guardar_cache_ner({"doc_1": ["HTA", "DM2", "DM2"]}, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["doc_1"], ["DM2", "HTA"])

    def test_cache_ner_rechaza_formato_invalido(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                guardar_cache_ner({"doc_1": "no-es-lista"}, Path(d) / "x.json")

    def test_cache_desde_transcripciones_exige_los_mismos_ids(self):
        transcripciones = {"a": {"texto": "x"}, "b": {"texto": "y"}}
        with self.assertRaises(ValueError):
            cache_desde_transcripciones(transcripciones, {"a": ["DM2"]})


class TestManifiesto(unittest.TestCase):
    def test_csv_con_columnas_del_equipo(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "casos.csv"
            path.write_text(
                "id,pregunta,imagen,esperado,criterio,tipo,entidad\n"
                'img_01,"¿Qué dice?",nota.png,"[""DM2""]",exacto,normal,DM2\n',
                encoding="utf-8",
            )
            casos = cargar_manifiesto(path)
        self.assertEqual(casos[0]["id"], "img_01")
        self.assertEqual(casos[0]["imagen"], "nota.png")
        self.assertEqual(casos[0]["esperado"], ["DM2"])
        self.assertEqual(casos[0]["entidad"], "DM2")

    def test_ids_repetidos_fallan(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "casos.json"
            path.write_text(
                json.dumps(
                    [
                        {"id": "x", "imagen": "a.png"},
                        {"id": "x", "imagen": "b.png"},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                cargar_manifiesto(path)

    def test_filtra_por_nivel_y_evita_duplicados(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "casos.csv"
            path.write_text(
                "id,imagen,nivel,grupo\n"
                "ex_1__p1,a.png,limpio,nucleo\n"
                "ex_1__p1,b.png,sombra_severo,nucleo\n",
                encoding="utf-8",
            )
            casos = cargar_manifiesto(path, nivel="limpio")
            self.assertEqual(len(casos), 1)
            self.assertEqual(casos[0]["nivel"], "limpio")
            with self.assertRaises(ValueError):
                cargar_manifiesto(path)
            with self.assertRaises(ValueError):
                cargar_manifiesto(path, nivel="noexiste")

    def test_resolver_ruta_imagen(self):
        caso = {"imagen": "nota.png"}
        self.assertEqual(
            Path(resolver_ruta_imagen(caso, "/imagenes", "/repo")),
            Path("/imagenes/nota.png"),
        )
        self.assertEqual(
            resolver_ruta_imagen({"imagen": "https://x/y.png"}, None, "/repo"),
            "https://x/y.png",
        )


class TestIntegracion(unittest.TestCase):
    def test_sin_imagen_delega_sin_contexto(self):
        llamadas = []

        def responder(pregunta, contexto_extra=""):
            llamadas.append((pregunta, contexto_extra))
            return "ok"

        salida = sistema_multimodal(
            "¿Tratamiento?", imagen=None, responder_m3=responder, cfg={}
        )
        self.assertEqual(salida, "ok")
        self.assertEqual(llamadas, [("¿Tratamiento?", "")])

    def test_con_imagen_pasa_el_texto_y_la_entidad(self):
        capturado = {}

        def responder(pregunta, contexto_extra=""):
            capturado["contexto"] = json.loads(contexto_extra)
            return "respuesta"

        with tempfile.TemporaryDirectory() as d:
            ruta = _imagen(Path(d))
            salida = sistema_multimodal(
                "¿Tratamiento?",
                Image.open(ruta),
                entidad="DM2",
                responder_m3=responder,
                backend="fake",
                cfg={"fake": {"texto": "Paciente con DM2 e HTA."}},
            )
        self.assertEqual(salida, "respuesta")
        self.assertEqual(capturado["contexto"]["entidades"], ["DM2"])
        self.assertIn("DM2", capturado["contexto"]["texto"])

    def test_sustituto_esta_marcado(self):
        self.assertIn("[SUSTITUTO M3]", responder_m3_sustituto("x", "y"))

    def test_crear_responder_mock(self):
        responder = crear_responder_m3({"integracion": {"usar_mocks": True}}, None)
        self.assertIn("[SUSTITUTO M3]", responder("x", "y"))

    def test_entidades_desde_json(self):
        cfg = {"integracion": {}}
        self.assertEqual(
            _entidades_desde_contexto('{"texto":"x","entidades":["DM2"]}', None, cfg),
            ["DM2"],
        )

    def test_entidades_con_extractor(self):
        cfg = {"integracion": {}}
        self.assertEqual(
            _entidades_desde_contexto("nota larga", lambda t: ["HTA"], cfg), ["HTA"]
        )

    def test_nota_larga_sin_extractor_no_se_usa_como_consulta(self):
        cfg = {"integracion": {}}
        nota = "Paciente con DM2. " * 20
        self.assertEqual(_entidades_desde_contexto(nota, None, cfg), [])


class TestNerBridge(unittest.TestCase):
    def test_bridge_escribe_cache_compatible_y_respeta_valvula(self):
        modelo = types.ModuleType("model_loader")
        modelo.cargar_modelo = lambda cfg, root=None: (None, None, {})

        common = types.ModuleType("common")
        common.chunk_words = lambda words, window, overlap: [(0, words)]
        common.bio_to_entity_set = lambda tokens, tags: {
            tok.lower().strip(".,") for tok, tag in zip(tokens, tags)
            if tag == "B-ENFERMEDAD"
        }
        common.aggregate_entities_by_original_doc = lambda ids, sets: {
            doc: set().union(*[s for d, s in zip(ids, sets) if d == doc]) for doc in ids
        }

        inference = types.ModuleType("inference")
        inference.predict_word_level = lambda tokens, model, tok, labels: [
            "B-ENFERMEDAD" if t.lower().strip(".,") in ("dm2", "hta") else "O"
            for t in tokens
        ]

        cache = {
            "doc_1": {"texto": "Paciente con DM2 e HTA.", "valvula": False},
            "doc_2": {"texto": MENSAJE_SIN_TEXTO, "valvula": True},
        }
        with tempfile.TemporaryDirectory() as d, patch.dict(
            sys.modules,
            {"model_loader": modelo, "common": common, "inference": inference},
        ):
            salida = Path(d) / "ner.json"
            resumen = generar_cache_ner(cache, {}, d, salida)
            data = json.loads(salida.read_text(encoding="utf-8"))
        self.assertEqual(data["doc_1"], ["dm2", "hta"])
        self.assertEqual(data["doc_2"], [])
        self.assertEqual(resumen["n_documentos"], 2)


class TestJuez(unittest.TestCase):
    def test_juez_si_y_no(self):
        imagen = Image.new("RGB", (10, 10), "white")
        self.assertEqual(
            juez_fidelidad_visual(
                "DM2", imagen, lambda img, prompt, max_new_tokens=3: "SI"
            ),
            1.0,
        )
        self.assertEqual(
            juez_fidelidad_visual(
                "DM2", imagen, lambda img, prompt, max_new_tokens=3: "NO"
            ),
            0.0,
        )


class TestRun(unittest.TestCase):
    def _cfg(self, tmp: Path) -> dict:
        return {
            "vision": {"backend": "fake", "comparar": ["fake"]},
            "fake": {"texto": "Paciente con DM2 e HTA."},
            "rutas": {
                "manifest": str(tmp / "casos.csv"),
                "imagenes_dir": str(tmp),
                "transcripciones_cache": "predicciones/transcripciones_{backend}.json",
                "ner_cache": "predicciones/ner_{backend}.json",
                "salida": "salidas/resultado_{backend}.json",
            },
            "ner": {"enabled": False},
            "integracion": {"usar_mocks": True},
        }

    def test_run_escribe_cache_y_resumen(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _imagen(tmp, "nota.png")
            (tmp / "casos.csv").write_text(
                "id,pregunta,imagen,entidad\nimg_01,¿Qué dice?,nota.png,DM2\n",
                encoding="utf-8",
            )
            cfg = self._cfg(tmp)
            resumen = run(cfg, tmp, manifest=str(tmp / "casos.csv"))
            cache = cargar_cache_transcripciones(
                tmp / "predicciones" / "transcripciones_fake.json"
            )
            resumen_json = json.loads(
                (tmp / "salidas" / "resultado_fake.json").read_text(encoding="utf-8")
            )
        self.assertEqual(resumen["n_casos"], 1)
        self.assertEqual(resumen["n_ok"], 1)
        self.assertEqual(cache["img_01"]["texto"], "Paciente con DM2 e HTA.")
        self.assertEqual(resumen_json["backend"], "fake")

    def test_run_comparar_escribe_resumen_lado_a_lado(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfg = self._cfg(tmp)
            cfg["vision"]["comparar"] = ["fake", "ocr_docling"]
            resumen = run(cfg, tmp, demo=True, comparar=True)
            archivo = tmp / "M4/vision/outputs/comparacion_backends.json"
            self.assertTrue(archivo.exists())
            self.assertIn("archivo_comparacion", resumen)
            self.assertEqual(len(resumen["comparacion"]), 2)

    def test_run_filtra_nivel_y_sufija_cache(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _imagen(tmp, "a.png")
            _imagen(tmp, "b.png")
            (tmp / "casos.csv").write_text(
                "id,imagen,nivel,grupo\n"
                "ex_1__p1,a.png,limpio,nucleo\n"
                "ex_1__p1,b.png,sombra_severo,nucleo\n",
                encoding="utf-8",
            )
            cfg = self._cfg(tmp)
            resumen = run(
                cfg,
                tmp,
                manifest=str(tmp / "casos.csv"),
                nivel="limpio",
                backend="fake",
            )
            cache_path = tmp / "predicciones" / "transcripciones_fake_limpio.json"
            salida_path = tmp / "salidas" / "resultado_fake_limpio.json"
            self.assertTrue(cache_path.exists())
            self.assertTrue(salida_path.exists())
            self.assertEqual(resumen["nivel"], "limpio")
            cache = cargar_cache_transcripciones(cache_path)
            self.assertEqual(list(cache), ["ex_1__p1"])

    def test_run_demo_genera_imagen(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfg = self._cfg(tmp)
            cfg["rutas"]["transcripciones_cache"] = "predicciones/transcripciones_{backend}.json"
            resumen = run(cfg, tmp, demo=True, backend="fake")
            self.assertTrue((tmp / "M4/vision/data/demo/nota_sintetica.png").exists())
        self.assertEqual(resumen["n_casos"], 1)
        self.assertEqual(resumen["n_ok"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
