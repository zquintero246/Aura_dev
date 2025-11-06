# Aura LoRA (Español)

Este directorio contiene todos los recursos necesarios para entrenar y desplegar **Aura**, la inteligencia artificial corporativa desarrollada por la empresa Aura, basada en el modelo [`IIC/RigoChat-7b-v2`](https://huggingface.co/IIC/RigoChat-7b-v2) y adaptada mediante LoRA exclusivamente para español.

## Contenido

- `datasets/aura_personality_es.jsonl`: dataset de instrucciones/respuestas que define la personalidad y tono de Aura.
- `train_aura_lora_es.py`: script de entrenamiento LoRA/QLoRA.
- `merge_lora_to_base.py`: fusiona el adaptador LoRA con el modelo base.
- `convert_to_gguf.sh`: conversión del modelo fusionado a formato GGUF listo para llama.cpp.
- `download_base_gguf.py`: descarga y guarda el modelo base en formato GGUF como `models/aura.gguf`.

## Requisitos

Python 3.10 o superior y GPU con al menos 16 GB VRAM recomendados.

### Dependencias

```bash
pip install "transformers>=4.36" "peft>=0.7.0" "datasets>=2.16" "bitsandbytes>=0.41" huggingface_hub
```

> **Nota:** `bitsandbytes` requiere CUDA. En CPU se puede omitir, aunque el entrenamiento será significativamente más lento.

## Entrenamiento del adaptador LoRA

1. Asegura acceso a Hugging Face (token si es necesario) y suficiente espacio en disco.
2. Descarga el modelo base en formato GGUF desde Hugging Face y guárdalo como `models/aura.gguf`:

   ```bash
   cd AuraLLM/AuraLoRA
   python download_base_gguf.py
   ```

   Puedes indicar otra variante con `--filename` o `--repo-id`.

3. (Opcional) Si deseas que la conversión final utilice directamente el script `convert-lora-to-gguf.py`, clona `llama.cpp` y define `LLAMA_CPP_DIR` apuntando a ese repositorio.
4. Ejecuta el entrenamiento:

   ```bash
   cd AuraLLM/AuraLoRA
   python train_aura_lora_es.py
   ```

   El script detecta automáticamente la memoria GPU: usa QLoRA (4 bits) cuando la GPU tiene menos de 24 GB de VRAM, o entrena en FP16 si hay más recursos. Los pesos del adaptador se guardan en `models/lora_aura_es/`.

## Fusión del LoRA con el modelo base

Una vez entrenado el adaptador:

```bash
cd AuraLLM/AuraLoRA
python merge_lora_to_base.py
```

Esto creará el modelo fusionado en `models/aura_es_merged/` en formato `.safetensors` y copiará el tokenizador correspondiente.

## Conversión a GGUF

Para preparar el modelo para `llama.cpp` (o ejecutarlo en Ollama mediante importación de GGUF), utiliza el script de conversión. El script verifica `models/aura.gguf` y, si definiste `LLAMA_CPP_DIR`, intenta fusionar directamente con `convert-lora-to-gguf.py`; en caso contrario utiliza `convert.py` sobre los pesos en formato Hugging Face (asegúrate de que el script esté disponible, por ejemplo en el repositorio `llama.cpp`).

```bash
cd AuraLLM/AuraLoRA
./convert_to_gguf.sh
```

El archivo resultante se guardará como `models/aura_final.gguf`.

## Uso en llama.cpp

Con el archivo GGUF puedes ejecutar inferencias locales:

```bash
./main -m AuraLLM/AuraLoRA/models/aura_final.gguf -p "Hola, ¿quién eres?"
```

El modelo responderá:

```
Soy Aura, la inteligencia artificial desarrollada por la empresa Aura...
```

## Uso en Ollama

1. Copia `models/aura_final.gguf` al directorio de modelos de Ollama (por ejemplo `~/.ollama/models/`).
2. Define un archivo `Modelfile` similar a:

   ```
   FROM ./aura_final.gguf
   PARAMETER temperature 0.8
   TEMPLATE "{{ .Prompt }}"
   ```
3. Construye el modelo:

   ```bash
   ollama create aura-es -f Modelfile
   ```
4. Ejecuta inferencias:

   ```bash
   ollama run aura-es "Hola Aura, necesito ayuda con un despliegue"
   ```

Aura mantendrá comunicación exclusivamente en español, con tono empático, técnico y profesional, sin mencionar otros modelos.

## Notas adicionales

- El dataset incluye ejemplos de presentación, soporte técnico, rechazos a idiomas no españoles y estilo corporativo.
- Ajusta hiperparámetros en `train_aura_lora_es.py` si necesitas más épocas o diferentes tamaños de lote según tus recursos.
- Para reanudar entrenamiento con checkpoints, modifica `TrainingArguments` para habilitar `save_total_limit` y otras opciones si lo deseas.

¡Disfruta entrenando y desplegando a Aura!
