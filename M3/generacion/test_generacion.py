"""
Pruebas de la pieza de generacion/orquestacion (Agustin / M3).

NO llaman a la API: usan mocks inyectados, que es lo que permite el desacople
por contratos. Correr con:

    python test_generacion.py
    # o:  python -m unittest test_generacion -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generacion import MENSAJE_SIN_EVIDENCIA, contextos_para_ragas, generar_respuesta, recolectar_api_keys
from orquestacion import es_sigla, resolver_query
from run_generacion import _cargar_entidades, _piezas_reales


def _frag(chunk_id: str, score: float, score_tipo: str = "test", texto: str = "texto") -> dict:
    return {"chunk_id": chunk_id, "doc_id": "doc1", "texto": texto, "score": score,
            "score_tipo": score_tipo, "fuente": "Fuente de prueba"}


def retrieve_controlado(mapa_scores: dict[str, float]):
    """Retriever falso: devuelve un fragmento con el score de `mapa_scores[consulta]`."""
    def retrieve(consulta: str, k: int):
        score = mapa_scores.get(consulta, 0.0)
        return [_frag("c_top", score)] if score > 0 else []
    return retrieve


def normalizador_controlado(exito: bool, termino: str = "ENFERMEDAD CONTROLADA"):
    def normalizar(entidad: str):
        if exito:
            return {"entidad_original": entidad, "entidad_normalizada": termino,
                    "source_terminology": "SNOMED CT", "normalization_failed": False}
        return {"entidad_original": entidad, "entidad_normalizada": entidad,
                "source_terminology": None, "normalization_failed": True}
    return normalizar


class TestOrquestacion(unittest.TestCase):

    def test_no_invoca_tool_si_hay_evidencia(self):
        r = resolver_query("neumonia", retrieve_fn=retrieve_controlado({"neumonia": 0.9}),
                           normalizar_fn=normalizador_controlado(True), umbral=0.5, k=5)
        self.assertFalse(r["tool_invoked"])
        self.assertEqual(r["query_final"], "neumonia")
        self.assertEqual(r["tool_reason"], "")
        self.assertTrue(r["fragments"])

    def test_invoca_tool_y_normaliza_con_score_bajo(self):
        retrieve = retrieve_controlado({"epoc": 0.1, "ENFERMEDAD CONTROLADA": 0.8})
        r = resolver_query("epoc", retrieve_fn=retrieve,
                           normalizar_fn=normalizador_controlado(True), umbral=0.5, k=5)
        self.assertTrue(r["tool_invoked"])
        self.assertEqual(r["query_final"], "ENFERMEDAD CONTROLADA")
        self.assertIn("normalizada vía SNOMED CT", r["tool_reason"])
        self.assertTrue(r["fragments"])

    def test_tool_sin_match_mantiene_entidad_cruda(self):
        r = resolver_query("amiloidosis", retrieve_fn=retrieve_controlado({"amiloidosis": 0.0}),
                           normalizar_fn=normalizador_controlado(False), umbral=0.5, k=5)
        self.assertTrue(r["tool_invoked"])
        self.assertEqual(r["query_final"], "amiloidosis")   # no inventa un termino
        self.assertIn("SIN match", r["tool_reason"])
        self.assertEqual(r["fragments"], [])

    def test_compuerta_evidencia_deja_sin_fragmentos(self):
        # score por debajo del umbral de evidencia -> no se entrega contexto
        r = resolver_query("rara", retrieve_fn=retrieve_controlado({"rara": 0.3}),
                           normalizar_fn=normalizador_controlado(False), umbral=0.9,
                           k=5, umbral_evidencia=0.5)
        self.assertEqual(r["fragments"], [])
        self.assertIn("sin evidencia", r["tool_reason"])

    def test_sigla_fuerza_normalizacion_aunque_el_score_sea_alto(self):
        retrieve = retrieve_controlado({"EPOC": 0.95, "enfermedad pulmonar obstructiva crónica": 0.9})
        r = resolver_query("EPOC", retrieve_fn=retrieve,
                           normalizar_fn=normalizador_controlado(True, "enfermedad pulmonar obstructiva crónica"),
                           umbral=0.5, k=5, forzar_por_sigla=True)
        self.assertTrue(es_sigla("EPOC"))
        self.assertTrue(r["tool_invoked"])
        self.assertEqual(r["query_final"], "enfermedad pulmonar obstructiva crónica")

    def test_score_rrf_no_es_comparable_con_umbral(self):
        rrf = lambda consulta, k: [_frag("c", 0.5, score_tipo="rrf")]
        with self.assertRaises(ValueError):
            resolver_query("diabetes", retrieve_fn=rrf, normalizar_fn=normalizador_controlado(True),
                           umbral=0.5, k=5)

    def test_fallo_de_retrieval_no_rompe_el_pipeline(self):
        def retrieve_que_falla(consulta, k):
            raise ConnectionError("indice caido")

        r = resolver_query("neumonia", retrieve_fn=retrieve_que_falla,
                           normalizar_fn=normalizador_controlado(True), umbral=0.5, k=5)
        self.assertTrue(r["tool_invoked"])
        self.assertIn("sin resultados", r["tool_reason"])


class TestGeneracion(unittest.TestCase):

    def test_sin_fragmentos_no_llama_al_llm(self):
        llamado = {"n": 0}

        def generar_fn(system, user):
            llamado["n"] += 1
            return "no deberia llamarse"

        r = generar_respuesta("amiloidosis", [], generar_fn=generar_fn)
        self.assertEqual(r["answer"], MENSAJE_SIN_EVIDENCIA)
        self.assertTrue(r["fallback_used"])
        self.assertEqual(r["sources_used"], [])
        self.assertEqual(llamado["n"], 0, "no debe llamar al LLM sin evidencia")

    def test_con_fragmentos_devuelve_fuentes_y_respuesta(self):
        def generar_fn(system, user):
            self.assertIn("Contexto:", user)
            self.assertIn("c_top", user)
            return "La neumonía se trata con amoxicilina."

        r = generar_respuesta("neumonia", [_frag("c_top", 0.9)], generar_fn=generar_fn)
        self.assertFalse(r["fallback_used"])
        self.assertEqual(r["sources_used"], ["c_top"])
        self.assertIn("amoxicilina", r["answer"])

    def test_detecta_la_valvula_de_escape_como_fallback(self):
        def generar_fn(system, user):
            return MENSAJE_SIN_EVIDENCIA

        r = generar_respuesta("neumonia", [_frag("c_top", 0.9)], generar_fn=generar_fn)
        self.assertTrue(r["fallback_used"])
        self.assertEqual(r["sources_used"], [], "si es fallback, no se apoya en ninguna fuente")

    def test_contextos_para_ragas_son_los_textos(self):
        frags = [_frag("a", 0.9, texto="texto A"), _frag("b", 0.8, texto="texto B")]
        self.assertEqual(contextos_para_ragas(frags), ["texto A", "texto B"])


class TestCargaEntidades(unittest.TestCase):

    def test_json_lista_de_strings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.json"
            p.write_text(json.dumps(["diabetes", "asma"]), encoding="utf-8")
            r = _cargar_entidades(p)
            self.assertEqual([x["entidad"] for x in r], ["diabetes", "asma"])
            self.assertIsNone(r[0]["ground_truth"])

    def test_jsonl_con_campo_consulta(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            p.write_text('{"consulta": "hipertension", "respuesta_esperada": "x"}\n', encoding="utf-8")
            r = _cargar_entidades(p)
            self.assertEqual(r[0]["entidad"], "hipertension")
            self.assertEqual(r[0]["ground_truth"], "x")


class TestIntegracionPiezasReales(unittest.TestCase):
    """Verifica el cableado con las piezas del equipo (imports + YAML) con modulos
    simulados que respetan las firmas reales de Pau y Luis."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "M3/retrieval").mkdir(parents=True)
        (root / "M3/tools").mkdir(parents=True)
        (root / "M3/retrieval/retrieval.py").write_text(
            "def retrieve_advanced(consulta, k):\n    return []\n"
            "def pregunta_intencion(entidad):\n"
            "    return f'¿Cuál es el tratamiento de {entidad}?'\n", encoding="utf-8")
        (root / "M3/retrieval/config_retrieval.py").write_text(
            "def cargar_config(ruta=None):\n    return {'ok': True}\n"
            "def configurar_pipeline(cfg):\n    return None\n"
            "def parametros_orquestacion(cfg):\n"
            "    return {'umbral': 0.7, 'umbral_evidencia': 0.3, 'forzar_por_sigla': True, 'k': 5}\n",
            encoding="utf-8")
        (root / "M3/tools/tool_normalizacion.py").write_text(
            "def normalizar_entidad(entidad, ontologia='SNOMEDCT', api_key=None):\n"
            "    return {'entidad_original': entidad, 'entidad_normalizada': entidad,\n"
            "            'source_terminology': None, 'normalization_failed': True}\n",
            encoding="utf-8")
        (root / "M3/tools/config.yaml").write_text(
            "tool_normalizacion:\n  ontologia: SNOMEDCT\n", encoding="utf-8")
        (root / "M3/retrieval/config_retrieval.yaml").write_text("corpus: mock\n", encoding="utf-8")
        self.root = root

    def _limpiar_modulos(self):
        import sys
        for mod in ("retrieval", "config_retrieval", "tool_normalizacion"):
            sys.modules.pop(mod, None)

    def test_piezas_reales_se_importan_y_entregan_parametros(self):
        self._limpiar_modulos()
        cfg = {"modulos": {
            "retrieval_dir": "M3/retrieval", "tool_dir": "M3/tools",
            "retrieval_config": "M3/retrieval/config_retrieval.yaml",
            "tool_config": "M3/tools/config.yaml"}}
        try:
            retrieve_fn, normalizar_fn, params, pregunta_fn = _piezas_reales(cfg, self.root)
            self.assertEqual(params["umbral"], 0.7)
            self.assertEqual(params["umbral_evidencia"], 0.3)
            self.assertTrue(params["forzar_por_sigla"])
            self.assertIn("tratamiento", pregunta_fn("diabetes"))
            # el adaptador de la tool fija la ontologia del YAML
            self.assertTrue(normalizar_fn("x")["normalization_failed"])
        finally:
            self._limpiar_modulos()

    def tearDown(self):
        self._tmp.cleanup()


class TestConfigClaves(unittest.TestCase):

    def test_recolectar_api_keys_deduplica(self):
        import os
        previo = dict(os.environ)
        try:
            os.environ["GROQ_API_KEY"] = "k1"
            os.environ["GROQ_API_KEY_2"] = "k2"
            os.environ["GROQ_API_KEY_3"] = "k1"
            self.assertEqual(recolectar_api_keys(), ["k1", "k2"])
        finally:
            os.environ.clear()
            os.environ.update(previo)


if __name__ == "__main__":
    unittest.main(verbosity=2)
