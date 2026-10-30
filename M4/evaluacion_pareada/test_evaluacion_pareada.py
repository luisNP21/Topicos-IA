"""Pruebas de M4/evaluacion_pareada (sin modelos; fixtures y salidas reales recortadas).

    python M4/evaluacion_pareada/test_evaluacion_pareada.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))

import auditar_entidades as au  # noqa: E402
import delta_pareado as dp  # noqa: E402
import evaluar_adversariales as ea  # noqa: E402
import verificar_gold as vg  # noqa: E402

GOLD = {"c1": ["diabetes", "hta"], "c2": ["epoc"]}

# Final del texto original de ex_30 (M2/examples/gold_examples.jsonl, línea 31).
ORIGINAL_EX30 = (
    "En la exploración física, cabe destacar una marcada pigmentación cutánea generalizada y "
    "alopecia, ambos relacionados con la toxicidad del 5-FU. Pese a las medidas de soporte "
    "establecidas, se produjo un empeoramiento del estado general, con abundantes secreciones "
    "respiratorias, mal control del dolor, sin recuperación hematológica (última analítica: "
    "plaquetas = 51 χ 103 µl; neutrófilos = 370/µl). El paciente falleció por insuficiencia "
    "respiratoria aguda en el seno de una sepsis de origen respiratorio."
)
# Salida real de Qwen2-VL-2B (MPS fp16) en el reto de recorte ex_30_p2 (recortada).
VLM_EX30_RECORTE = (
    "realizada y alopecia, ambos relacionados con la toxicidad a las medidas de soporte "
    "establecidas, se produjo un deterioro respiratorio, mal control del dolor, sin recuperación "
    "hematológica y una elevada concentración de plaquetas (51 x 103 /μl); neutrófilos = 370 /μl). "
    "El paciente falleció por insuficiencia respiratoria aguda en el seno de una enfermedad "
    "pulmonar obstructiva crónica."
)


def _dim1(d: Path, nombre: str, true_by_doc: dict, pred_by_doc: dict) -> str:
    p = d / f"{nombre}.json"
    p.write_text(json.dumps({"true_by_doc": true_by_doc, "pred_by_doc": pred_by_doc}), encoding="utf-8")
    return str(p)


class TestDeltaPareado(unittest.TestCase):
    def test_pareado_escribe_scorecard(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            tx = _dim1(d, "t", GOLD, {"c1": ["diabetes", "hta"], "c2": ["epoc"]})
            vi = _dim1(d, "v", GOLD, {"c1": ["diabetes"], "c2": ["epoc", "asma"]})
            meta = d / "m.json"
            meta.write_text(json.dumps({"latencia_media_s": 4.5}))
            out = d / "scorecard_m4.csv"
            with redirect_stdout(io.StringIO()):
                rc = dp.main(["--texto", tx, "--vision", f"limpio={vi}", "--meta", f"limpio={meta}",
                              "--salida", str(out)])
            self.assertEqual(rc, 0)
            filas = list(csv.DictReader(out.open(encoding="utf-8")))
        self.assertEqual([f["condicion"] for f in filas], ["texto", "limpio"])
        self.assertEqual((filas[1]["tp"], filas[1]["fp"], filas[1]["fn"]), ("2", "1", "1"))
        self.assertAlmostEqual(float(filas[1]["delta_f1"]), round(2 / 3 - 1, 4))
        self.assertEqual(filas[1]["latencia_s_caso"], "4.5")

    def test_gold_distinto_no_escribe_delta(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            tx = _dim1(d, "t", {"ex_0": ["a"], "ex_1": ["b"]}, {"ex_0": ["a"], "ex_1": []})
            vi = _dim1(d, "v", GOLD, {"c1": ["diabetes"], "c2": []})
            out = d / "scorecard_m4.csv"
            with redirect_stderr(io.StringIO()) as err:
                rc = dp.main(["--texto", tx, "--vision", f"limpio={vi}", "--salida", str(out)])
            self.assertEqual(rc, 2)
            self.assertFalse(out.exists())
        self.assertIn("NO comparable", err.getvalue())

    def test_resultados_de_revision_son_pareados(self):
        r = DIR / "resultados_revision"
        tx = json.loads((r / "resultado_dimension1_texto.json").read_text(encoding="utf-8"))
        li = json.loads((r / "resultado_dimension1_limpio.json").read_text(encoding="utf-8"))
        self.assertEqual(dp.diferencias_gold(tx["true_by_doc"], li["true_by_doc"]), [])
        self.assertEqual(dp.prf(tx["true_by_doc"], tx["pred_by_doc"])["tp"], 29)


class TestVerificarGold(unittest.TestCase):
    def test_detecta_entidad_no_literal(self):
        gold = {"c1": {"text": "Paciente con Diabetes Mellitus tipo 2.", "entities": ["diabetes mellitus tipo 2", "depresión"]}}
        self.assertEqual(vg.no_literales(gold), [("c1", "depresión")])

    def test_comparar_golds(self):
        a = {"c1": {"text": "", "entities": ["epoc", "hta"]}}
        b = {"c1": {"text": "", "entities": ["epoc"]}}
        self.assertEqual(vg.comparar(a, a), [])
        self.assertEqual(len(vg.comparar(a, b)), 1)


class TestAuditoria(unittest.TestCase):
    def test_clasifica_con_el_gold_del_harness(self):
        gold = {"c1": {"text": "Varón con diabetes e hipertensión arterial y obesidad.",
                       "entities": ["diabetes", "hipertensión arterial"]}}
        r = au.auditar(gold, {"c1": {"diabetes", "hipertensión arterial", "obesidad", "epoc"}})
        self.assertEqual((r["acierto"], r["fp_ner"], r["inventada"]), (2, 1, 1))
        self.assertEqual(r["inventadas"][0]["entidad"], "epoc")

    def test_une_paginas(self):
        self.assertEqual(au.por_documento({"c1__p1": ["A"], "c1__p2": ["b"]}), {"c1": {"a", "b"}})


class TestAdversariales(unittest.TestCase):
    def test_frase_inventada_por_el_vlm_no_pasa(self):
        caso = {"tipo_criterio": "sin_inventado"}
        r = ea.evaluar(caso, {"texto": VLM_EX30_RECORTE, "valvula": False}, ORIGINAL_EX30)
        self.assertEqual(r["resultado"], "no_pasa")
        self.assertIn("enfermedad pulmonar obstructiva crónica", r["detalle"])

    def test_lectura_literal_con_una_palabra_mal_leida_pasa(self):
        caso = {"tipo_criterio": "sin_inventado"}
        texto = ORIGINAL_EX30.replace("pigmentación", "pigmentacion")
        self.assertEqual(ea.evaluar(caso, {"texto": texto, "valvula": False}, ORIGINAL_EX30)["resultado"], "pasa")

    def test_eco_sin_contenido_cuenta_como_valvula(self):
        caso = {"tipo_criterio": "valvula_o_sin_inventado"}
        t = {"texto": "Se aprecia texto legible en la imagen.", "valvula": False}
        self.assertEqual(ea.evaluar(caso, t, ORIGINAL_EX30), {"resultado": "pasa", "detalle": "válvula"})

    def test_sin_transcripcion(self):
        self.assertEqual(ea.evaluar({"tipo_criterio": "valvula"}, None, "")["resultado"], "sin_transcripcion")

    def test_csv_tiene_criterio_por_caso(self):
        with open(DIR / "adversariales.csv", encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        self.assertGreaterEqual(len(filas), 5)
        self.assertTrue(all(x["criterio"].strip() and x["tipo_criterio"] for x in filas))


if __name__ == "__main__":
    unittest.main(verbosity=1)
