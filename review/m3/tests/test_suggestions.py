"""Validación local de la propuesta. Los servicios y modelos son dobles: no hay APIs.

Desde el repositorio: python -m unittest discover -s review/m3/tests -p test_suggestions.py -v
Para probar contra un checkout con el patch aplicado: definir M3_REVIEW_SOURCE_ROOT.
No son pruebas end-to-end ni las pruebas de los estudiantes.
"""
from __future__ import annotations
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager, redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('M3_REVIEW_SOURCE_ROOT', Path(__file__).resolve().parents[3]))
KEYS = ('faithfulness', 'context_precision', 'context_recall', 'answer_relevancy')

def module(name, **attrs):
    m = ModuleType(name)
    for k,v in attrs.items(): setattr(m,k,v)
    return m

def load(name, filename, extra=None):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'M3/ragas' / filename)
    m = importlib.util.module_from_spec(spec)
    modules = {'dotenv': module('dotenv', load_dotenv=lambda: None)}
    modules.update(extra or {})
    with patch.dict(sys.modules, modules): spec.loader.exec_module(m)
    return m

EVAL = load('eval_propuesta', 'evaluacion_ragas.py')
CROSS = load('cruce_propuesta', 'cruce_harness.py')
CONFIG = module('config_utils', cargar_config=Mock())
CLI = load('cli_propuesta', 'run_ragas.py', {'config_utils': CONFIG, 'evaluacion_ragas': EVAL, 'cruce_harness': CROSS})

def sample(**kw):
    c = {'id':'prueba-1', 'question':'Qué color tiene el objeto?',
         'contexts':['El objeto es azul.'], 'answer':'El objeto es azul.',
         'ground_truth':'Azul.', 'fallback_used':False}
    c.update(kw); return c

class EmbModel:
    def __init__(self,*args,**kwargs): pass
    def encode(self,texts,**kwargs): return np.ones((len(texts),3))

@contextmanager
def evaluator(rows=None, *, error=None, scores=False):
    result = SimpleNamespace(scores=rows) if scores else SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))
    run = Mock(side_effect=error) if error is not None else Mock(return_value=result)
    metrics = {k:SimpleNamespace(llm=None, embeddings=None) for k in KEYS}
    mods = {
        'ragas':module('ragas', evaluate=run), 'ragas.metrics':module('ragas.metrics',**metrics),
        'ragas.llms':module('ragas.llms',LangchainLLMWrapper=lambda x:x),
        'ragas.embeddings':module('ragas.embeddings',LangchainEmbeddingsWrapper=lambda x:x),
        'ragas.run_config':module('ragas.run_config',RunConfig=lambda **kw:kw),
        'datasets':module('datasets',Dataset=SimpleNamespace(from_dict=lambda x:x)),
        'langchain_groq':module('langchain_groq',ChatGroq=lambda **kw:object()),
        'langchain_core':module('langchain_core'),
        'langchain_core.embeddings':module('langchain_core.embeddings',Embeddings=object),
        'sentence_transformers':module('sentence_transformers',SentenceTransformer=EmbModel),
    }
    with patch.dict(sys.modules,mods), patch.dict(os.environ, {'GROQ_API_KEY':'TEST-NO-NETWORK'},clear=True):
        yield run

class Tests(unittest.TestCase):
    def setUp(self):
        self.out=io.StringIO(); self.redirect=redirect_stdout(self.out); self.redirect.__enter__()
        self.model=EVAL._EMB_MODEL; EVAL._EMB_MODEL='fallback_tokens'
    def tearDown(self):
        EVAL._EMB_MODEL=self.model; self.redirect.__exit__(None,None,None)

    def test_01_failure_not_replaced_by_proxy(self):
        with evaluator(error=RuntimeError('controlado')), patch.object(EVAL,'_calcular_mock') as proxy:
            with self.assertRaises(RuntimeError): EVAL.run({'ragas':{'modo':'real'},'eval_set_externo':[sample()]})
            proxy.assert_not_called()

    def test_02_valid_zeroes_are_preserved(self):
        row=dict.fromkeys(KEYS,0.0)
        with evaluator([row]), patch.object(EVAL,'_calcular_mock') as proxy:
            r=EVAL.run({'ragas':{'modo':'real'},'eval_set_externo':[sample()]})
        self.assertEqual(r['metricas'],row); self.assertEqual(r['backend'],'ragas')
        self.assertEqual(r['estado'],'completo'); proxy.assert_not_called()

    def test_03_nan_is_null_not_zero(self):
        row=dict.fromkeys(KEYS,0.5); row['context_recall']=np.nan
        with evaluator([row]): r=EVAL.run({'ragas':{'modo':'real'},'eval_set_externo':[sample()]})
        self.assertIsNone(r['metricas']['context_recall']); self.assertEqual(r['estado'],'parcial')
        self.assertEqual(r['cobertura']['context_recall']['n_validos'],0)
        json.dumps(r,allow_nan=False)

    def test_04_partial_mean_and_denominator(self):
        rows=[dict.fromkeys(KEYS,x) for x in [0.2,np.nan,0.6]]
        r=EVAL._resumir_filas(rows,[sample() for _ in rows])
        self.assertAlmostEqual(r['metricas']['faithfulness'],0.4)
        self.assertEqual(r['cobertura']['faithfulness'],{'n_validos':2,'n_total':3,'n_no_calculados':1})

    def test_05_all_nan_is_not_measured_zero(self):
        r=EVAL._resumir_filas([dict.fromkeys(KEYS,np.nan)],[sample()])
        self.assertEqual(r['estado'],'sin_metricas_validas'); self.assertTrue(all(v is None for v in r['metricas'].values()))

    def test_06_no_overwrite_for_legacy_abstention(self):
        row={**sample(answer='No hay evidencia suficiente.',contexts=[],fallback_used=True),**dict.fromkeys(KEYS,0.2)}
        with evaluator([row]): r=EVAL._calcular_real([sample()], 'fake')
        self.assertEqual(r,dict.fromkeys(KEYS,0.2))

    def test_07_v2_response_fields_preserve_scores(self):
        row={'response':'No hay evidencia suficiente.','retrieved_contexts':[],**dict.fromkeys(KEYS,0.2)}
        with evaluator([row]): r=EVAL._calcular_real([sample()], 'fake')
        self.assertEqual(r,dict.fromkeys(KEYS,0.2))

    def test_08_scores_interface_of_pinned_version(self):
        with evaluator([dict.fromkeys(KEYS,0.3)],scores=True):
            r=EVAL._calcular_real([sample()], 'fake')
        self.assertEqual(r,dict.fromkeys(KEYS,0.3))

    def test_09_proxy_does_not_reward_mixed_abstention(self):
        c=sample(contexts=[],answer='No hay evidencia suficiente. El objeto es azul.')
        self.assertIsNone(EVAL._faithfulness_mock(c,'fallback_tokens'))
        self.assertIsNone(EVAL._context_precision_mock(c,'fallback_tokens'))

    def test_10_proxy_without_claims_is_not_perfect(self):
        self.assertIsNone(EVAL._faithfulness_mock(sample(answer=''),'fallback_tokens'))

    def test_11_loader_preserves_id_reference_and_flag(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'; p.write_text(json.dumps({'resultados':[
                {'id':'caso_x','pregunta':'Qué color?','esperado':'azul','answer':'azul','contexts':['azul'],'fallback_used':False}]}))
            r=EVAL.cargar_eval_set_de_generacion(p)[0]
        self.assertEqual(r['id'],'caso_x'); self.assertEqual(r['question'],'Qué color?')
        self.assertEqual(r['ground_truth'],'azul'); self.assertFalse(r['fallback_used'])

    def test_12_reference_not_invented_by_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'; p.write_text(json.dumps({'resultados':[{'entidad':'objeto','answer':'azul'}]}))
            r=EVAL.cargar_eval_set_de_generacion(p)
        self.assertIsNone(r[0]['ground_truth'])

    def test_13_real_mode_rejects_missing_reference_before_api(self):
        with patch.object(EVAL,'_calcular_real') as real:
            with self.assertRaises(ValueError): EVAL.run({'ragas':{'modo':'real'},'eval_set_externo':[sample(ground_truth=None)]})
            real.assert_not_called()

    def test_14_real_mode_does_not_use_embedded_set_implicitly(self):
        with patch.object(EVAL,'cargar_eval_set_de_generacion',return_value=None),patch.object(EVAL,'_calcular_real') as real:
            with self.assertRaises(FileNotFoundError): EVAL.run({'ragas':{'modo':'real'}})
            real.assert_not_called()

    def test_15_mock_remains_available_and_labeled(self):
        r=EVAL.run({'ragas':{'modo':'mock'},'eval_set_externo':[sample()]})
        self.assertEqual(r['modo'],'mock'); self.assertTrue(r['backend'].startswith('proxy_local'))

    def test_16_abstention_flag_is_not_a_safety_metric(self):
        r=EVAL._resumir_filas([dict.fromkeys(KEYS,0.4)],[sample(fallback_used=True)])
        self.assertTrue(r['por_caso'][0]['abstencion_declarada'])
        self.assertEqual(r['metricas']['faithfulness'],0.4)
        self.assertNotIn('abstencion_segura',r['por_caso'][0])

    def test_17_global_metrics_do_not_create_individual_diagnoses(self):
        history={'metrics':{'f1':1.0},'true_by_doc':{'a':['x'],'b':['y']},'pred_by_doc':{'a':['x'],'b':['y']}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m2.json'; p.write_text(json.dumps(history))
            r=CROSS.cruzar(str(p),dict.fromkeys(KEYS,0.0),{})
        self.assertEqual(r['diagnosticos'],{}); self.assertEqual(r['resumen'],{})
        self.assertEqual(r['f1_global_m2'],1.0); self.assertEqual(r['n_documentos_m2'],2)
        self.assertEqual(r['alcance'],'agregados_no_pareados')

    def test_18_m2_f1_formula_unchanged(self):
        r=CROSS._f1_por_doc({'a':['x','y']},{'a':['x']})['a']
        self.assertAlmostEqual(r['f1'],2/3)

    def test_19_table_supports_nulls(self):
        table=CROSS._tabla_markdown({'a':{'f1':0.5}},dict.fromkeys(KEYS,None))
        self.assertIn('N/D',table); self.assertNotIn('problema_corpus_retrieval',table)

    def test_20_missing_m2_does_not_make_up_documents(self):
        with patch.object(CROSS.Path,'exists',return_value=False): r=CROSS.cruzar('missing',dict.fromkeys(KEYS,None),{})
        self.assertEqual(r['alcance'],'sin_resultados_m2'); self.assertEqual(r['f1_por_doc'],{})

    def test_21_strict_json_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.json'; CLI.guardar_reporte({'value':None},p)
            self.assertEqual(json.loads(p.read_text()),{'value':None})
            with self.assertRaises(FileExistsError): CLI.guardar_reporte({'value':1},p)
            q=Path(d)/'bad.json'
            with self.assertRaises(ValueError): CLI.guardar_reporte({'value':float('nan')},q)
            self.assertFalse(q.exists())

    def test_22_cli_records_per_case_scores_and_null(self):
        with tempfile.TemporaryDirectory() as d:
            inp=Path(d)/'input.json'; out=Path(d)/'report.json'
            inp.write_text(json.dumps({'resultados':[sample()]}))
            cfg={'ragas':{'modo':'real','archivo_generacion':str(inp)}}
            row=dict.fromkeys(KEYS,0.3); row['context_recall']=np.nan
            with evaluator([row]),patch.object(CLI,'cargar_config',return_value=cfg),patch.object(sys,'argv',['run','--salida',str(out)]):
                CLI.main()
            r=json.loads(out.read_text()); self.assertEqual(r['ragas']['backend'],'ragas')
            self.assertIsNone(r['ragas']['metricas']['context_recall'])
            self.assertIn('por_caso',r['ragas']); self.assertIn('versiones',r['meta'])
            self.assertNotIn('TEST-NO-NETWORK',out.read_text())

    def test_23_cli_records_failure_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'error.json'
            with patch.object(CLI,'cargar_config',return_value={'ragas':{'modo':'real'}}),patch.object(CLI,'run_ragas',side_effect=RuntimeError('fallo controlado')),patch.object(sys,'argv',['run','--salida',str(out)]):
                with self.assertRaises(RuntimeError): CLI.main()
            r=json.loads(out.read_text()); self.assertEqual(r['estado'],'error'); self.assertNotIn('ragas',r)

    def test_24_length_mismatch_rejected(self):
        with self.assertRaises(ValueError): EVAL._resumir_filas([dict.fromkeys(KEYS,0.4)],[sample(),sample()])

    def test_25_nonfinite_and_wrong_types_not_zero(self):
        row=dict(zip(KEYS,[float('inf'),None,'0.5',True]))
        r=EVAL._resumir_filas([row],[sample()]); self.assertEqual(r['metricas'],dict.fromkeys(KEYS,None))

    def test_26_inputs_not_mutated(self):
        row={**sample(contexts=[],answer='No hay evidencia.'),**dict.fromkeys(KEYS,0.2)}
        copy=dict(row)
        r=EVAL._resumir_filas([row],[sample()]); self.assertEqual(row,copy)

if __name__=='__main__': unittest.main(verbosity=2)
