# Ejecutar el notebook en Google Colab

## Requisitos

- Tener en Google Drive el adaptador LoRA en:
  `MyDrive/TopicosIA/Proyecto-Salud/M1/saved_models/clinical_bert-distemist-lora/`
- Tener el gold set correspondiente al notebook elegido en:
  `MyDrive/TopicosIA/Proyecto-Salud/M2/eval_harness/`
- Crear una API key de Groq (en console.groq.com/keys) y agregarla en la ventana de **Secretos / Keys** de Colab (icono de llave  en la barra lateral izquierda):
  - **Nombre:** `GROQ_API_KEY`
  - **Valor:** tu clave de Groq (`gsk_...`)
  - Asegurarse de activar el interruptor de **Acceso de Notebook** (*Notebook access*).

## Configuracion

En `M2/harness/config.yaml` se deben establecer estas variables:

```yaml
rutas:
  gold_set: "M2/eval_harness/gold_examples_adversariales.jsonl"
  outputs_dir: "M2/outputs"
  model_dir: "M1/saved_models/clinical_bert-distemist-lora"
```

Durante la ejecucion del notebook tambien se utilizan las variables
`PROJECT_ROOT` para la ruta del proyecto en Drive y `GROQ_API_KEY` para la API.

## Ejecucion

1. Cargar en Google Colab uno de estos notebooks:
  - `start_gold.ipynb` para evaluar el gold set normal.
  - `start_adversarial.ipynb` para evaluar el gold set adversarial.
2. Asegurarse de tener la key `GROQ_API_KEY` configurada y activa en la ventana de Secretos de Colab.
3. Ejecutar las celdas en orden y autorizar el acceso a Google Drive cuando Colab lo solicite.
4. Verificar o ajustar en `M2/harness/config.yaml` estas variables:

  ```yaml
  rutas:
    gold_set: "M2/eval_harness/gold_examples_adversariales.jsonl"
    outputs_dir: "M2/outputs"
    model_dir: "M1/saved_models/clinical_bert-distemist-lora"
  ```

5. El notebook instalará los requerimientos y ejecutará `run_harness.py` con `config.yaml`.

Al finalizar, los resultados se encontraran en `M2/outputs/` dentro de Google Drive.