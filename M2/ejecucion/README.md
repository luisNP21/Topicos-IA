# Ejecutar el notebook en Google Colab

## Requisitos

- Tener en Google Drive el adaptador LoRA en:
  `MyDrive/TopicosIA/Proyecto-Salud/M1/saved_models/clinical_bert-distemist-lora/`
- Tener el gold set en:
  `MyDrive/TopicosIA/Proyecto-Salud/M2/eval_harness/gold_examples_adversariales.jsonl`
- Crear una API key de Groq y guardarla en los secretos de Colab con el nombre
  `GROQ_API_KEY`.

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

1. Abrir `start.ipynb` en Google Colab.
2. Ejecutar las celdas en orden.
3. Autorizar el acceso a Google Drive cuando Colab lo solicite.

Al finalizar, los resultados se encontraran en la carpeta `outputs` de Google Drive.