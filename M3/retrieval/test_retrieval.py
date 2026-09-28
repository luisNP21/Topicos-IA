"""
Pruebas del retrieval, la orquestación y el experimento. Usan un embedder y un reranker
falsos con la misma interfaz que los reales, así que no descargan modelos.

    python -m unittest test_retrieval -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "generacion"))

from experimento_s08 import (_medir, asignar_pliegues, calibrar, correr_experimento,  # noqa: E402
                             deltas_contrato, guardar, umbral_por_tolerancia, youden)
from orquestacion import es_sigla, resolver_query  # noqa: E402
from retrieval import (IndiceRAG, chunks_desde_chroma, clasificar_secciones, densa_chroma,  # noqa: E402
                       densa_en_memoria, info_indice, pregunta_intencion, tokenizar)


class EmbedderFalso:
    """Bolsa de palabras: dos textos se parecen si comparten palabras."""
    def encode(self, textos, **kw):
        m = np.zeros((len(textos), 256))
        for i, t in enumerate(textos):
            for w in tokenizar(t.removeprefix("query: ").removeprefix("passage: ")):
                m[i, zlib.crc32(w.encode()) % 256] += 1
        return m + 1e-3


class RerankerLogits:
    """Devuelve logits (fuera de [0, 1]) según las palabras compartidas."""
    def predict(self, pares):
        return np.array([4.0 * len(set(tokenizar(q)) & set(tokenizar(d))) / max(1, len(set(tokenizar(q)))) - 2.0
                         for q, d in pares])


class ColeccionFalsa:
    """Las operaciones de Chroma que usa el retrieval: count, get y query."""
    def __init__(self, chunks):
        self.chunks = chunks

    def count(self):
        return len(self.chunks)

    def get(self, include, ids=None):
        return {"ids": [c["chunk_id"] for c in self.chunks], "documents": [c["texto"] for c in self.chunks],
                "metadatas": [{"doc_id": c["doc_id"], "seccion": c.get("seccion") or ""} for c in self.chunks]}

    def query(self, query_embeddings, n_results, include):
        return {"ids": [["sepsis_c0", "epoc_c0"][:n_results]], "distances": [[0.1, 0.4][:n_results]]}


class EmbedderQueGraba(EmbedderFalso):
    def encode(self, textos, **kw):
        self.visto, self.kw = textos, kw
        return super().encode(textos)


CHUNKS = [
    {"chunk_id": "epoc_c0", "doc_id": "gpc_epoc", "seccion": "Diagnóstico",
     "texto": "La enfermedad pulmonar obstructiva crónica se diagnostica con espirometría."},
    {"chunk_id": "sifilis_c0", "doc_id": "gpc_sifilis", "seccion": "Tratamiento",
     "texto": "La sífilis se trata con penicilina benzatínica."},
    {"chunk_id": "sepsis_c0", "doc_id": "gpc_sepsis", "seccion": None,
     "texto": "La sepsis requiere antibiótico en la primera hora."},
]
MANIFEST = {"gpc_sifilis": {"doc_id": "gpc_sifilis", "titulo": "Guía de sífilis gestacional y congénita"}}

GUIA = [   # todos los chunks nombran la enfermedad: solo la intención distingue el tratamiento
    {"chunk_id": "dm_0", "doc_id": "gpc_dm", "seccion": "Epidemiología",
     "texto": "La diabetes mellitus tipo 2 es frecuente en adultos; la diabetes afecta a millones."},
    {"chunk_id": "dm_1", "doc_id": "gpc_dm", "seccion": "Diagnóstico",
     "texto": "El diagnóstico de diabetes mellitus tipo 2 usa la glucemia; diabetes confirmada con HbA1c."},
    {"chunk_id": "dm_2", "doc_id": "gpc_dm", "seccion": "Tratamiento farmacológico",
     "texto": "El tratamiento de la diabetes mellitus tipo 2 inicia con metformina."},
    {"chunk_id": "dm_3", "doc_id": "gpc_dm", "seccion": "5.2",
     "texto": "Se recomienda metformina y ajustar dosis; manejo con insulina si falla."},
]


def indice(chunks=CHUNKS, **kw):
    return IndiceRAG(chunks, densa_en_memoria(chunks, EmbedderFalso()), RerankerLogits(), n_candidatos=3, **kw)


def frag(score, doc="d", tipo="reranker_prob"):
    return {"chunk_id": f"{doc}_c0", "doc_id": doc, "texto": "t", "score": score, "score_tipo": tipo}


def normalizador(mapa):
    def f(e):
        if e.lower() in mapa:
            return {"entidad_original": e, "entidad_normalizada": mapa[e.lower()],
                    "source_terminology": "SNOMED CT", "normalization_failed": False}
        return {"entidad_original": e, "entidad_normalizada": e, "source_terminology": None,
                "normalization_failed": True}
    return f


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.ix = indice()

    def test_tokenizar_sin_tildes_ni_puntuacion(self):
        self.assertEqual(tokenizar("Neumonía, adquirida (comunidad)."), ["neumonia", "adquirida", "comunidad"])

    def test_bm25_no_devuelve_chunks_sin_coincidencias(self):
        self.assertEqual(self.ix.bm25("amiloidosis", 3), [])
        self.assertEqual(self.ix.chunks[self.ix.bm25("sifilis", 3)[0][0]]["doc_id"], "gpc_sifilis")

    def test_rrf_premia_el_consenso(self):
        self.ix.densa = lambda q, k: [(2, .91), (1, .83)]
        self.ix.bm25 = lambda q, k: [(0, 12.7), (1, 4.2)]
        self.assertEqual(self.ix.hibrida("x", 3)[0][0], 1)

    def test_reranker_entrega_probabilidades(self):
        self.assertTrue(self.ix.aplicar_sigmoide)
        scores = [f["score"] for f in self.ix.buscar("sepsis antibiótico", 3, "rerank")]
        self.assertTrue(all(0 <= s <= 1 for s in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fragmentos_y_tipos_de_score(self):
        esperado = {"densa": "cosine", "hibrida": "rrf", "rerank": "reranker_prob",
                    "rerank_pregunta": "reranker_prob", "intencion_sin_rerank": "rrf",
                    "intencion": "reranker_prob", "intencion_seccion": "reranker_prob"}
        for modo, tipo in esperado.items():
            f = self.ix.buscar("sífilis", 2, modo)[0]
            self.assertTrue({"chunk_id", "doc_id", "texto", "score", "fuente", "seccion_tipo"} <= f.keys())
            self.assertEqual(f["score_tipo"], tipo, modo)

    def test_encabezado_contextual(self):
        con = indice(manifest=MANIFEST)
        sin = indice(manifest=MANIFEST, enriquecer=False)
        self.assertEqual(con.chunks[con.bm25("sifilis gestacional", 3)[0][0]]["doc_id"], "gpc_sifilis")
        self.assertEqual(sin.bm25("gestacional", 3), [])
        f = con.buscar("sífilis", 1, "rerank")[0]
        self.assertEqual(f["fuente"], "Guía de sífilis gestacional y congénita")
        self.assertEqual(f["texto"], CHUNKS[1]["texto"])

    def test_lectura_desde_chroma(self):
        col = ColeccionFalsa(CHUNKS)
        self.assertIsNone(chunks_desde_chroma(col)[2]["seccion"])
        emb = EmbedderQueGraba()
        resultado = densa_chroma(col, emb)("sepsis", 2)
        self.assertEqual(emb.visto, ["query: sepsis"])
        self.assertTrue(emb.kw.get("normalize_embeddings"))
        self.assertEqual([(c, round(s, 3)) for c, s in resultado], [("sepsis_c0", 0.9), ("epoc_c0", 0.6)])

    def test_huella_del_indice(self):
        a = info_indice(indice())
        otro = [dict(c) for c in CHUNKS]
        otro[0]["texto"] += " (reingestado)"
        self.assertEqual((a["n_chunks"], a["n_documentos"]), (3, 3))
        self.assertNotEqual(a["huella_sha256"], info_indice(indice(otro))["huella_sha256"])


class TestIntencionYSecciones(unittest.TestCase):
    def test_clasificacion_por_titulo_y_por_similitud(self):
        tipos, origen = clasificar_secciones(GUIA, EmbedderFalso().encode([c["texto"] for c in GUIA]),
                                             EmbedderFalso())
        self.assertEqual(tipos, ["otra", "otra", "tratamiento", "tratamiento"])
        self.assertEqual(origen, ["titulo", "titulo", "titulo", "similitud"])

    def test_sin_embedder_la_seccion_ambigua_queda_desconocida(self):
        self.assertEqual(clasificar_secciones(GUIA)[0][3], "desconocida")

    def test_intencion_prefiere_la_seccion_de_tratamiento(self):
        top = indice(GUIA).buscar("diabetes mellitus tipo 2", 1, "intencion")[0]
        self.assertEqual((top["chunk_id"], top["seccion_tipo"]), ("dm_2", "tratamiento"))

    def test_reranker_recibe_la_pregunta_de_tratamiento(self):
        ix, vistos = indice(GUIA), []
        ix.reranker = type("R", (), {"predict": lambda self, p: vistos.extend(p) or np.zeros(len(p))})()
        ix.buscar("asma", 1, "rerank_pregunta")
        self.assertEqual(vistos[0][0], pregunta_intencion("asma"))

    def test_filtro_de_seccion_con_respaldo(self):
        f = indice(GUIA).buscar("diabetes mellitus tipo 2", 4, "intencion_seccion")
        self.assertTrue(all(x["seccion_tipo"] == "tratamiento" for x in f))
        sin_tratamiento = [dict(c, seccion="Diagnóstico") for c in GUIA]
        self.assertEqual(len(indice(sin_tratamiento).buscar("diabetes mellitus tipo 2", 2, "intencion_seccion")), 2)

    def test_metricas_por_seccion(self):
        m = _medir([frag(.9, "gpc_dm")], ["gpc_dm"], 3, ["otro_c0"])
        self.assertEqual((m["hit1"], m["hit1_sec"]), (1, 0))


class TestOrquestacion(unittest.TestCase):
    def test_zona_confiable_no_invoca_la_normalizacion(self):
        r = resolver_query("sepsis", retrieve_fn=lambda q, k: [frag(.9)], normalizar_fn=normalizador({}),
                           umbral=.7, umbral_evidencia=.3, k=5)
        self.assertFalse(r["tool_invoked"])

    def test_zona_dudosa_normaliza_y_gana_el_mejor_intento(self):
        scores = {"epoc": .4, "enfermedad pulmonar obstructiva crónica": .95}
        r = resolver_query("EPOC", retrieve_fn=lambda q, k: [frag(scores[q.lower()])],
                           normalizar_fn=normalizador({"epoc": "enfermedad pulmonar obstructiva crónica"}),
                           umbral=.7, umbral_evidencia=.3, k=5)
        self.assertEqual(r["query_final"], "enfermedad pulmonar obstructiva crónica")

    def test_si_normalizar_empeora_se_conserva_la_entidad(self):
        scores = {"lúes": .5, "sífilis": .2}
        r = resolver_query("lúes", retrieve_fn=lambda q, k: [frag(scores[q])],
                           normalizar_fn=normalizador({"lúes": "sífilis"}), umbral=.7, umbral_evidencia=.3, k=5)
        self.assertEqual(r["query_final"], "lúes")
        self.assertIn("sin mejora", r["tool_reason"])

    def test_sin_evidencia_no_entrega_fragmentos(self):
        r = resolver_query("amiloidosis", retrieve_fn=lambda q, k: [frag(.1), frag(.05)],
                           normalizar_fn=normalizador({}), umbral=.7, umbral_evidencia=.3, k=5)
        self.assertEqual(r["fragments"], [])
        self.assertIn("sin evidencia", r["tool_reason"])

    def test_compuerta_descarta_el_ruido(self):
        r = resolver_query("sepsis", retrieve_fn=lambda q, k: [frag(.9, "a"), frag(.2, "b")],
                           normalizar_fn=normalizador({}), umbral=.7, umbral_evidencia=.3, k=5)
        self.assertEqual([f["doc_id"] for f in r["fragments"]], ["a"])

    def test_regla_de_siglas(self):
        self.assertTrue(es_sigla("SIADH") and es_sigla("EPOC") and not es_sigla("sepsis"))
        scores = {"hta": (.826, "gpc_dm_gestacional"), "hipertensión arterial": (.7, "gpc_hta")}
        r = resolver_query("HTA", retrieve_fn=lambda q, k: [frag(*scores[q.lower()])],
                           normalizar_fn=normalizador({"hta": "hipertensión arterial"}),
                           umbral=.5, umbral_evidencia=.3, k=5, forzar_por_sigla=True)
        self.assertEqual(r["fragments"][0]["doc_id"], "gpc_hta")

    def test_sigla_fuera_del_corpus_se_rechaza(self):
        scores = {"epoc": .603, "enfermedad pulmonar obstructiva crónica": .1}
        r = resolver_query("EPOC", retrieve_fn=lambda q, k: [frag(scores[q.lower()])],
                           normalizar_fn=normalizador({"epoc": "enfermedad pulmonar obstructiva crónica"}),
                           umbral=.5, umbral_evidencia=.3, k=5, forzar_por_sigla=True)
        self.assertEqual(r["fragments"], [])

    def test_rechaza_scores_rrf(self):
        with self.assertRaises(ValueError):
            resolver_query("x", retrieve_fn=lambda q, k: [frag(.016, tipo="rrf")],
                           normalizar_fn=normalizador({}), umbral=.5, k=5)


class TestCalibracionYExperimento(unittest.TestCase):
    def test_youden(self):
        self.assertEqual(youden([.8, .9, .7], [.1, .2, .75]), .7)

    def test_umbral_de_evidencia_queda_sobre_el_peor_caso_fuera_del_corpus(self):
        t = umbral_por_tolerancia([.9, .5, .15], [.1, .3])
        self.assertTrue(.3 < t < .5)

    def test_calibracion_informa_si_hay_zona_dudosa(self):
        filas = [{"consulta": c, "hit1": h, "hit1_sec": None, "score_top": s, "score_relevante": s,
                  "score_relevante_sec": None, "fuera_de_corpus": f}
                 for c, h, s, f in [("a", 1, .9, False), ("b", 1, .8, False), ("c", 0, .4, False),
                                    ("d", 0, .1, True), ("e", 0, .2, True)]]
        cal = calibrar(filas)
        self.assertTrue(cal["hay_zona_dudosa"])
        self.assertLess(cal["umbral_evidencia"], cal["umbral"])

    def test_pliegues_balanceados_por_tipo(self):
        consultas = [{"tipo": t} for t in ["a"] * 6 + ["b"] * 3]
        pliegues = asignar_pliegues(consultas, 3)
        self.assertEqual(sorted(p for p, c in zip(pliegues, consultas) if c["tipo"] == "a"), [0, 0, 1, 1, 2, 2])

    def test_corrida_completa_con_validacion_cruzada(self):
        consultas = [
            {"consulta": "sífilis", "tipo": "exacta", "relevantes": ["gpc_sifilis"]},
            {"consulta": "sepsis", "tipo": "exacta", "relevantes": ["gpc_sepsis"]},
            {"consulta": "enfermedad pulmonar", "tipo": "exacta", "relevantes": ["gpc_epoc"]},
            {"consulta": "EPOC", "tipo": "sigla", "relevantes": ["gpc_epoc"]},
            {"consulta": "amiloidosis", "tipo": "fuera_de_corpus", "relevantes": [], "fuera_de_corpus": True},
            {"consulta": "lupus", "tipo": "fuera_de_corpus", "relevantes": [], "fuera_de_corpus": True},
        ]
        norm = normalizador({"epoc": "enfermedad pulmonar obstructiva crónica"})
        detalle, resumen, cal = correr_experimento(indice(), consultas, k=3, normalizar_fn=norm,
                                                   sistema_base="intencion", n_pliegues=2)
        claves = [r["Clave"] for r in resumen]
        self.assertEqual(claves[0], "ingenuo")
        self.assertEqual(claves[-2:], ["completo", "completo_siglas"])
        self.assertIn("intencion_sin_reranker", claves)
        self.assertEqual(len(cal["pliegues"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            guardar(Path(tmp), detalle, resumen, cal, 3, indice())
            for nombre in ("deltas_s08.csv", "detalle_por_consulta.csv", "umbrales.json",
                           "delta_naive_vs_advanced.json", "indice_info.json"):
                self.assertTrue((Path(tmp) / nombre).exists(), nombre)
        d = deltas_contrato(resumen, 3)
        self.assertTrue(d and all(x.keys() == {"metric_name", "naive", "advanced", "delta"} for x in d))

    def test_el_sistema_base_debe_tener_reranker(self):
        with self.assertRaises(ValueError):
            correr_experimento(indice(), [], sistema_base="intencion_sin_reranker")


class TestConfiguracion(unittest.TestCase):
    def setUp(self):
        import retrieval
        self.original = (retrieval.PLANTILLAS_INTENCION, retrieval.PREGUNTA_INTENCION)

    def tearDown(self):
        import retrieval
        retrieval.configurar_intencion(*self.original)

    def test_yaml_por_defecto(self):
        from config_retrieval import cargar_config, rutas
        cfg = cargar_config()
        self.assertEqual(cfg["corpus"], "mock")
        self.assertIn("chroma_dir", rutas(cfg))
        self.assertEqual(pregunta_intencion("asma"), cfg["intencion"]["pregunta"].format(e="asma"))

    def test_intencion_configurable(self):
        import yaml
        from config_retrieval import cargar_config
        cfg = yaml.safe_load((AQUI / "config_retrieval.yaml").read_text(encoding="utf-8"))
        cfg["intencion"] = {"plantillas": ["manejo de {e}"], "pregunta": "¿Cómo se trata {e}?"}
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "c.yaml"
            ruta.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
            cargar_config(ruta)
        self.assertEqual(pregunta_intencion("asma"), "¿Cómo se trata asma?")

    def test_parametros_de_orquestacion(self):
        import json
        from config_retrieval import cargar_config, parametros_orquestacion
        cfg = cargar_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["rutas"]["mock"]["salida"] = tmp
            provisional = cfg["orquestacion"]["umbrales_provisionales"]["umbral"]
            self.assertEqual(parametros_orquestacion(cfg)["umbral"], provisional)
            cfg["orquestacion"]["umbrales_provisionales"] = None
            with self.assertRaises(FileNotFoundError):
                parametros_orquestacion(cfg)
            (Path(tmp) / "umbrales.json").write_text(json.dumps({"final": {"umbral": .8, "umbral_evidencia": .3}}))
            self.assertEqual(parametros_orquestacion(cfg)["umbral"], .8)
        cfg["orquestacion"].update(umbral=.6, umbral_evidencia=.2)
        self.assertEqual(parametros_orquestacion(cfg)["umbral_evidencia"], .2)

    def test_rutas_relativas_a_la_raiz_del_repositorio(self):
        from config_retrieval import RAIZ_REPO, cargar_config, resolver, rutas
        cfg = cargar_config()
        self.assertEqual(Path(rutas(cfg)["consultas"]), RAIZ_REPO / "M3/data/mock/consultas_mock.jsonl")
        self.assertEqual(resolver("/content/x"), str(Path("/content/x")))
        self.assertIsNone(resolver(None))

    def test_copia_el_indice_desde_el_origen(self):
        from config_retrieval import cargar_config, preparar_indice
        cfg = cargar_config()
        with tempfile.TemporaryDirectory() as tmp:
            origen = Path(tmp) / "drive_chroma"
            origen.mkdir()
            (origen / "chroma.sqlite3").write_text("x")
            cfg["rutas"]["mock"].update(chroma_origen=str(origen), chroma_dir=str(Path(tmp) / "local"))
            self.assertTrue((Path(preparar_indice(cfg)) / "chroma.sqlite3").exists())
            cfg["rutas"]["mock"].update(chroma_origen=str(Path(tmp) / "no_existe"), chroma_dir=str(Path(tmp) / "otro"))
            with self.assertRaises(FileNotFoundError):
                preparar_indice(cfg)

    def test_normalizador_simulado_y_real(self):
        from config_retrieval import cargar_config, normalizador_desde_config
        cfg = cargar_config()
        self.assertFalse(normalizador_desde_config(cfg)("HTA")["normalization_failed"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tool_normalizacion.py").write_text(
                "def normalizar_entidad(entidad, ontologia='X'):\n"
                "    return {'entidad_original': entidad, 'entidad_normalizada': ontologia,\n"
                "            'source_terminology': ontologia, 'normalization_failed': False}\n")
            (Path(tmp) / "config.yaml").write_text("tool_normalizacion:\n  ontologia: SNOMEDCT\n")
            cfg["rutas"]["mock"]["mapa_normalizacion"] = None
            cfg["normalizacion"] = {"directorio": tmp, "config": str(Path(tmp) / "config.yaml")}
            self.assertEqual(normalizador_desde_config(cfg)("DM2")["entidad_normalizada"], "SNOMEDCT")

    def test_verifica_el_modelo_del_corpus(self):
        from config_retrieval import cargar_config, verificar_modelo_del_corpus
        cfg = cargar_config()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text("embeddings:\n  model_name: otro-modelo\n")
            cfg["corpus_pipeline"] = {"config": str(Path(tmp) / "config.yaml")}
            with self.assertRaises(RuntimeError):
                verificar_modelo_del_corpus(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
