# Módulo 1 - Fine Tuning

En esta carpeta se encuentran 3 notebooks base:

- 01_dataset_preparation.ipynb
- 02_dataset_chunking_and_splits_base.ipynb
- 03_finetuning_base.ipynb

Estos fueron el punto de partida para desarrollar el mismo pipeline con los modelos propuestos en este módulo. Fueron utilizados como base para la comparación y la implementación del flujo completo de preparación, chunking y fine-tuning.

Además, dentro de la carpeta del proyecto se encuentran las ejecuciones propias de cada modelo:

- BERT/
- clinical_BERT/
- mT5/

En cada una de estas carpetas se guarda la ejecución específica del modelo junto con sus outputs y resultados asociados.

A partir del análisis de los resultados de los modelos implementados, seleccionamos el **Roberta Base Biomedical** como el modelo oficial para este módulo. Por esta razón, para ejecutar el modelo elegido, se pueden descargar directamente los notebooks 2 y 3 de la carpeta `clinical_BERT` y correrlos siguiendo la misma lógica del pipeline base, con la configuración adaptada a este modelo.

# Ejecución de los notebooks de M1 en Google Colab

La ejecución debe hacerse en el siguiente orden: Notebook 1, Notebook 2 y Notebook 3. Esto es porque el Notebook 1 genera los splits base, el Notebook 2 los procesa y crea los splits chunked, y el Notebook 3 usa esos resultados para entrenar los modelos.

## 1. Subir la carpeta original de DisTEMIST al Drive

La carpeta original descargada desde Zenodo debe quedar dentro de Google Drive en la siguiente estructura base:

/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/

Dentro de esa carpeta, como mínimo, debes tener:

```
project/
├── distemist_raw/
│   └── training/
│       ├── text_files/
│       └── subtrack1_entities/
├── distemist_final/
├── outputs/
└── saved_models/
```

**Hay dos formas de subir DisTEMIST:**

1. Opción manual: descarga la carpeta original de DisTEMIST desde Zenodo y súbela manualmente a Google Drive en la ruta anterior.
2. Opción con carpeta compartida del grupo: si recibes acceso a la carpeta compartida de Google Drive del grupo, crea un acceso directo a esa carpeta dentro de:

/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/

y asegúrate de que el nombre y la estructura interna coincidan con los nombres esperados por los notebooks, especialmente `distemist_raw` y `distemist_final`.

Si prefieres otra ruta, no importa, siempre que la uses de forma consistente en todos los notebooks. Lo importante es ajustar las rutas del código para que apunten a tu estructura real.

## 2. Abrir los notebooks en Colab

Desde Google Drive:

- Abre cada notebook con Google Colab
- Cuando aparezca el aviso de montaje de Drive, acepta la conexión
- Si los notebooks importan `google.colab`, Colab ya lo manejará

En cada notebook encontrarás esto al inicio:

```python
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    RAW_DATA_DIR = "."
```

Eso ya está preparado para Colab. Si corres el notebook localmente, cambia la ruta manualmente.

- Sigue las instrucciones en los notebooks para ajustar las variables necesarias dependiendo del caso.

## 3. Orden de ejecución

### Notebook 1: 01_dataset_preparation.ipynb

**Objetivo:**
- Cargar los datos crudos de DisTEMIST
- Preparar el dataset base
- Generar los splits train/dev/test
- Guardar el resultado en la carpeta distemist_final

**Instrucciones:**

1. Abre el notebook en Colab.
2. Ejecuta la celda de montaje de Drive:
3. Revisa y cambia la ruta base del dataset antes de correr la preparación:

```python
RAW_DATA_DIR = "/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/distemist_raw/training"
FINAL_DATA_DIR = Path("/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/distemist_final")
```

4. Si tu carpeta en Drive se llama diferente, reemplaza las rutas con la carpeta real que usaste.
5. Ejecuta el notebook completo.
6. Verifica que quede creada la carpeta de salida con los splits iniciales:

```python
/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/distemist_final
```

### Notebook 2: 02_dataset_chunking_and_splits_base.ipynb

**Objetivo:**
- Cargar los splits generados por el Notebook 1
- Definir la ventana de chunking
- Transformar el texto en formatos adecuados para BERT/Roberta y mT5
- Dejar los datos listos para entrenamiento

Instrucciones explícitas:

1. Abre este notebook en Colab después de completar el Notebook 1.
2. Ejecuta la celda de montaje de Drive.
3. Ajusta la ruta de entrada para que apunte a la carpeta generada por el Notebook 1:

```python
SPLITS_DIR = "/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/distemist_final"
```

4. Si cambiaste la ruta base de tu proyecto, usa el mismo nombre en todos los notebooks.
5. Ejecuta el notebook completo para generar los datasets listos para fine-tuning.
6. Verifica que quede creada la carpeta de salida:

```python
/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/distemist_final/FORMATO-SALIDA
```

FORMATO-SALIDA puede ser `distemist_mt5_format` o `distemist_bert_format` dependiendo del modelo elegido.

### Notebook 3: 03_finetuning_base.ipynb

**Objetivo:**
- Cargar los datos procesados del notebook anterior
- Seleccionar el modelo (`bert`, `clinical_bert` o `mt5`)
- Aplicar LoRA
- Antrenar y guardar resultados en Drive

Instrucciones explícitas:

1. Abre este notebook en Colab después de ejecutar el Notebook 2.
2. Ejecuta la celda de montaje de Drive.
3. Ajusta las rutas de entrada y salida a tu estructura real:

```python
SPLITS_DIR = Path("/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/distemist_final/")
OUTPUT_DIR = Path("/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/outputs")
```
5. Al final del entrenamiento, los resultados y modelos se guardarán en:

```python
/content/drive/MyDrive/TopicosIA/Proyecto-Salud/M1/saved_models
```

6. Ejecuta todo el notebook completo.


## 6. Resultados

En el PDF `M1.pdf` se puede encontrar el PDF con la explicación de principio a fin del desarrollo de este Módulo 1 junto a los resultados obtenidos.
