# An\u00e1lisis del repositorio OrpheusX

## Requisitos del sistema
OrpheusX exige Python \u2265 3.10 y CUDA Toolkit versi\u00f3n 12.4. Usar una versi\u00f3n m\u00e1s reciente puede provocar fallos de instalaci\u00f3n.

## Instalaci\u00f3n b\u00e1sica
```bash
git clone https://github.com/EQXai/OrpheusX.git
cd OrpheusX
bash scripts/install.sh
source venv/bin/activate
```
El script `install.sh` crea un entorno virtual, instala PyTorch 2.6 para CUDA 12.4 y dependencias como `bitsandbytes`, `accelerate`, `xformers`, `peft`, `trl`, `unsloth`, `snac`, `torchaudio`, `whisperx` y `gradio`. Adem\u00e1s instala `ffmpeg` y `libcudnn8` si el sistema lo permite.

## Estructura general
- `scripts/` contiene utilidades de preparaci\u00f3n de datasets, entrenamiento, inferencia y un CLI interactivo.
- `orpheusx/utils/` incluye funciones de segmentaci\u00f3n de texto.
- `tools/Whisper/` incorpora un wrapper de WhisperX para transcribir y segmentar audio.
- `prompt_list/` almacena listas de prompts en JSON.
- `source_audio/` es el lugar donde se colocan los archivos de audio originales.
- `audio_output/` se crea autom\u00e1ticamente para guardar los resultados de inferencia.
- `lora_models/` almacenar\u00e1 los adaptadores LoRA entrenados.
- `logs/` contiene `orpheus.log` con registros de todas las operaciones.

## Preparaci\u00f3n del dataset
El script `prepare_dataset.py` usa WhisperX para transcribir el audio y segmentarlo en clips de 15 a 25 segundos. Cada clip se alinea exactamente con su transcripci\u00f3n. Los datasets se guardan bajo `datasets/<nombre>` junto con `dataset.parquet` para facilitar su distribuci\u00f3n.
Tambien existe `prepare_dataset_interactive.py` que permite seleccionar archivos de `source_audio/` de forma interactiva.

## Entrenamiento de LoRA
`train_interactive.py` carga cada dataset (local o desde Hugging Face), genera tokens de audio con el modelo SNAC y entrena un adaptador LoRA para `unsloth/orpheus-3b-0.1-ft`. Los adaptadores se guardan en `lora_models/<dataset>/lora_model`.

Se utilizan tokens especiales:
- `TOKENISER_LENGTH = 128256`
- `start_of_text = 128000`, `end_of_text = 128009`
- `start_of_speech = TOKENISER_LENGTH + 1`, `end_of_speech = TOKENISER_LENGTH + 2`
- `start_of_human = TOKENISER_LENGTH + 3`, `end_of_human = TOKENISER_LENGTH + 4`
- `start_of_ai = TOKENISER_LENGTH + 5`, `end_of_ai = TOKENISER_LENGTH + 6`
- `pad_token = TOKENISER_LENGTH + 7`

El entrenamiento usa `Trainer` de HuggingFace con `adamw_8bit`, 60 pasos m\u00e1ximos y gradiente acumulado.

## Inferencia
Los scripts `infer_interactive.py` e `infer.py` cargan el modelo base y, opcionalmente, una LoRA. El texto puede dividirse en segmentos para evitar el l\u00edmite de 2048 tokens. Cada segmento se convierte en audio usando SNAC; los segmentos se unen con `concat_with_fade`, que aplica un crossfade de 60 ms por defecto.
El audio generado se escribe en `audio_output/<lora_name>/` con nombres incrementales.

## Segmentaci\u00f3n de prompts
Las funciones `split_prompt_by_tokens` y `split_prompt_by_sentences` (en `orpheusx/utils/segment_utils.py`) permiten dividir un texto seg\u00fan n\u00famero de tokens o l\u00edmites de oraciones. Para textos largos tambi\u00e9n existe el modo `full_segment` que separa por caracteres (, . ? !). El CLI y Gradio pueden mostrar un registro con los \u00edndices de cada segmento.

## Interfaz de Gradio
`gradio_app.py` ofrece una interfaz web para todo el flujo:
1. Preparar datasets subiendo audio o usando archivos de `source_audio/`.
2. Entrenar LoRAs seleccionando datasets locales o v\u00ednculos de Hugging Face.
3. Ejecutar inferencia manualmente o mediante listas de prompts. Incluye opciones avanzadas para temperatura, top-p, repetici\u00f3n, longitud m\u00e1xima, segmentaci\u00f3n y duraci\u00f3n del crossfade.
4. Una pesta\u00f1a **Full Segment Test** permite probar la segmentaci\u00f3n basada exclusivamente en caracteres.
El puerto de Gradio se solicita al iniciar el script y por defecto es 7860.

## Uso del CLI
`python scripts/orpheus_cli.py` muestra un men\u00fa interactivo que permite instalar dependencias, crear datasets, entrenar modelos y realizar inferencia desde la terminal.

## Verificaci\u00f3n del entorno
El script `check_env.py` comprueba que el entorno virtual y los paquetes esenciales est\u00e9n instalados y que CUDA est\u00e9 disponible.

## Tokens predefinidos para expresiones
Durante la inferencia se pueden incluir tokens como `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>` y `<gasp>` para insertar expresiones pregrabadas en el audio.

