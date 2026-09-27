# M3/tools — Tool de normalización terminológica

Tool de normalización de entidades clínicas contra **SNOMED CT** vía la API de
**BioPortal**. Parte del pipeline de M3 (Luis): recibe el nombre de una enfermedad
en texto libre (salida del encoder Clinical BERT de M1/M2) y devuelve el término
canónico de SNOMED CT, para mejorar la calidad de las queries de retrieval.

---

## Qué hace

1. **`tool_normalizacion.py`** — consulta BioPortal con el nombre crudo de la entidad
   y devuelve el `prefLabel` de SNOMED CT. Si no hay API key, corre en **modo mock**
   automáticamente (sin errores, sin llamadas externas).

2. **`orquestacion.py`** — implementa la regla de negocio de cuándo invocar la tool:
   si el retrieval con la entidad cruda ya da un score ≥ `umbral_score_retrieval`, no
   se invoca (evitar llamadas innecesarias a la API). Solo se normaliza cuando el
   retrieval falla.

---

## Cómo obtener una API key de BioPortal

1. Ir a [bioportal.bioontology.org](https://bioportal.bioontology.org) y crear una cuenta (gratuita).
2. En la sección **Account → API Key**, copiar la clave generada.
3. Crear un archivo `.env` en la raíz del proyecto (o en `M3/tools/`) con:
   ```env
   PROJECT_ROOT=/ruta/a/TopicosIA/Proyecto-Salud
   BIOPORTAL_API_KEY=tu-clave-aqui
   ```
   En **Google Colab**, agregar `BIOPORTAL_API_KEY` en la ventana de Secretos/Keys
   (icono  en la barra lateral), con acceso de notebook activado.

> **Sin API key**, la tool corre en modo mock: devuelve respuestas fijas para
> `"diabetes tipo 2"` e `"hipertension"`, y marca como fallida cualquier otra entidad.
> Útil para smoke-tests y desarrollo sin depender de la API.

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

```bash
python run_tools.py --config config.yaml
```

Salida de ejemplo (modo mock):
```
[tool_normalizacion] BIOPORTAL_API_KEY no encontrada -- corriendo en MOCK_MODE.
{'modo': 'mock', 'ontologia': 'SNOMEDCT', 'n_evaluadas': 3,
 'n_normalizacion_fallida': 1,
 'resultados': [
   {'entidad_original': 'diabetes tipo 2',
    'entidad_normalizada': 'Diabetes mellitus, tipo 2',
    'source_terminology': 'SNOMED CT (mock)',
    'normalization_failed': False},
   {'entidad_original': 'hipertension',
    'entidad_normalizada': 'Hipertension esencial',
    'source_terminology': 'SNOMED CT (mock)',
    'normalization_failed': False},
   {'entidad_original': 'una enfermedad rarisima inventada',
    'entidad_normalizada': 'una enfermedad rarisima inventada',
    'source_terminology': None,
    'normalization_failed': True}
 ]}
```

---

## Estructura

```
M3/tools/
├── config.yaml           # ontologia, umbral, entidades de prueba
├── config_utils.py       # cargar_config() — mismo patrón que M3/corpus/
├── requirements.txt      # requests, python-dotenv, pyyaml
├── run_tools.py          # entrypoint CLI
├── tool_normalizacion.py # lógica de normalización (real + mock)
└── orquestacion.py       # regla de cuándo invocar la tool
```
