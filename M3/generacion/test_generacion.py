"""
Pruebas de la pieza de generacion/orquestacion (Agustin / M3).

NO llaman a la API: usan mocks inyectados, que es justamente lo que permite el
desacople por contratos. Correr con:

    python test_generacion.py
    # o:  python -m unittest test_generacion -v
"""

from __future__ import annotations

import unittest

from generacion import MENSAJE_SIN_EVIDENCIA, generar_respuesta, recolectar_api_keys
from orquestacion import resolver_query


def _frag(chunk_id: str, score: float, texto: str = "texto") -> dict:
    return {"chunk_id": chunk_id, "doc_id": "doc1", "texto": texto,
            "score": score, "score_tipo": "test", "fuente": "Fuente de prueba"}


def retrieve_controlado(mapa_scores: dict[str, float]):
    """Retriever falso: score del top-1 lo decide `mapa_scores[consulta]`."""
    def retrieve(consulta: str, k: int):
        score = mapa_scores.get(consulta, 0.0)
        if score <= 0:
            return []
        return [_frag("c_top", score)]
    return retrieve


def normalizador_controlado(exito: bool):
    def normalizar(entidad: str):
        if exito:
            return {"entidad_original": entidad, "entidad_normalizada": "ENFERMEDAD CONTROLADA",
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

    def test_fallo_de_retrieval_no_rompe_el_pipeline(self):
        def retrieve_que_falla(consulta, k):
            raise ConnectionError("indice caido")

        r = resolver_query("neumonia", retrieve_fn=retrieve_que_falla,
                           normalizar_fn=normalizador_controlado(True), umbral=0.5, k=5)
        # No hay excepcion: se degrada de forma explicita.
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
            self.assertIn("c_top", user)   # el chunk_id va en el contexto
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


class TestConfigClaves(unittest.TestCase):

    def test_recolectar_api_keys_deduplica(self):
        import os
        previo = dict(os.environ)
        try:
            os.environ["GROQ_API_KEY"] = "k1"
            os.environ["GROQ_API_KEY_2"] = "k2"
            os.environ["GROQ_API_KEY_3"] = "k1"   # duplicada
            claves = recolectar_api_keys()
            self.assertEqual(claves, ["k1", "k2"])
        finally:
            os.environ.clear()
            os.environ.update(previo)


if __name__ == "__main__":
    unittest.main(verbosity=2)
