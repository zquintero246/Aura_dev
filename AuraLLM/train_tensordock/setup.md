# Preparación de entrenamiento en TensorDock

Este documento describe los pasos sugeridos para preparar una instancia con GPU NVIDIA A100 de 80 GB en TensorDock y ejecutar el entrenamiento completo de Aura en español.

## 1. Preparar la instancia

1. Reserva una instancia con GPU A100 80 GB y al menos 16 CPU + 100 GB de almacenamiento NVMe.
2. Conéctate por SSH y actualiza el sistema:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. Instala las dependencias del sistema:
   ```bash
   sudo apt install -y build-essential git python3 python3-venv python3-dev
   ```
4. Crea un entorno virtual dedicado y actívalo:
   ```bash
   python3 -m venv ~/aura-env
   source ~/aura-env/bin/activate
   ```
5. Instala las librerías necesarias (PyTorch con CUDA 11.8, Transformers y utilidades):
   ```bash
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets tqdm sentencepiece bitsandbytes
   ```

## 2. Colocar el dataset en español

1. Copia el corpus al directorio `/datasets/spanish_corpus/` dentro de la instancia. Puedes usar `rsync`, `scp` o montar un bucket.
2. El script acepta ficheros `.txt` (un texto grande o un documento por línea) y `.jsonl` con una clave `text` por entrada. Ejemplos:
   ```bash
   /datasets/spanish_corpus/wikipedia_es.txt
   /datasets/spanish_corpus/redes_sociales.jsonl
   ```
3. Si tienes varios archivos, concaténalos o indica el principal mediante el argumento `--dataset_path` al lanzar el entrenamiento.

## 3. Configurar y lanzar el entrenamiento

1. Sitúate en la carpeta del repositorio y activa el entorno virtual.
2. Crea un directorio para los artefactos de entrenamiento (checkpoints, logs, tokenizer entrenado, etc.):
   ```bash
   mkdir -p /workspace/aura_runs/aura_spanish
   ```
3. Ejecuta el script de entrenamiento:
   ```bash
   python AuraLLM/train_tensordock/train_tensordock.py \
       --dataset_path /datasets/spanish_corpus/wikipedia_es.txt \
       --output_dir /workspace/aura_runs/aura_spanish \
       --tokenizer_name_or_path spanish-tokenizer \
       --epochs 5 \
       --batch_size 8 \
       --gradient_accumulation_steps 8 \
       --learning_rate 3e-4 \
       --save_steps 1000
   ```
   Ajusta `--tokenizer_name_or_path` al nombre o ruta local de tu tokenizador en español. Si aún no cuentas con uno, crea uno con [tokenizers](https://huggingface.co/docs/tokenizers/python/latest/quicktour) antes de iniciar.

4. El script detectará la GPU automáticamente, usará `torch.cuda.amp` para mezclado de precisión y guardará checkpoints periódicamente en el directorio indicado. Internamente reutiliza la arquitectura `GPT2` y la configuración `Config` definidas en `AuraLLM/train/train_aura.py`, por lo que es coherente con el pipeline de entrenamiento original del repositorio.

## 4. Reanudar un entrenamiento

1. Localiza el checkpoint más reciente, por ejemplo `/workspace/aura_runs/aura_spanish/checkpoint-step-4000.pt`.
2. Relanza el entrenamiento indicando la ruta en `--resume_from`:
   ```bash
   python AuraLLM/train_tensordock/train_tensordock.py \
       --dataset_path /datasets/spanish_corpus/wikipedia_es.txt \
       --output_dir /workspace/aura_runs/aura_spanish \
       --tokenizer_name_or_path spanish-tokenizer \
       --resume_from /workspace/aura_runs/aura_spanish/checkpoint-step-4000.pt
   ```
3. El script restaurará modelo, optimizador, `GradScaler` y continuará desde el paso guardado.

## 5. Recomendaciones operativas

- **Interrupciones controladas**: si necesitas detener el entrenamiento, presiona `Ctrl+C`. El script captura la interrupción y guarda un checkpoint inmediatamente antes de salir.
- **Frecuencia de checkpoints**: usa `--save_steps` para controlar cada cuántos pasos se guarda el estado. Para una ejecución de 72 horas, se recomienda guardar al menos cada 30-45 minutos.
- **Almacenamiento**: asegúrate de contar con espacio suficiente para checkpoints (cada uno puede ocupar varios GB). Elimina checkpoints antiguos o muévelos a almacenamiento externo si es necesario.
- **Monitoreo**: revisa el archivo `training_log.csv` dentro de `output_dir` para seguir la pérdida por época y el progreso total.
- **Snapshots finales**: al finalizar, el script guarda `final_model.pt`, `model_config.json` y el tokenizador (`tokenizer.json`, `vocab.json`, etc.) en el directorio de salida.

Con estos pasos podrás preparar TensorDock, cargar el corpus y entrenar Aura en español de forma reproducible y segura.
