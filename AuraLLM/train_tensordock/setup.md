# Entrenamiento en TensorDock con GPU A100 80 GB

Este documento describe cómo preparar TensorDock y ejecutar un entrenamiento de Aura completamente en español, optimizando la arquitectura para aprovechar una A100 de 80 GB durante ~72 horas.

## 1. Preparar la instancia

1. Reserva una instancia con al menos:
   - 1× NVIDIA A100 80 GB.
   - 16 CPU físicos (32 hilos recomendados).
   - 100 GB de SSD NVMe para checkpoints y datasets.
2. Conéctate por SSH y actualiza el sistema:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. Instala dependencias de compilación y Python:
   ```bash
   sudo apt install -y build-essential git python3 python3-venv python3-dev
   ```
4. Crea un entorno virtual aislado e instala los paquetes principales (PyTorch CUDA 11.8, Transformers, bitsandbytes, etc.):
   ```bash
   python3 -m venv ~/aura-env
   source ~/aura-env/bin/activate
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets tqdm sentencepiece bitsandbytes
   ```

## 2. Colocar el dataset en español

1. Copia el corpus a `/datasets/spanish_corpus/`. Puedes usar `scp`, `rsync`, montar un bucket o descargar directamente.
2. El script acepta:
   - Archivos `.txt` (texto continuo o un documento por línea).
   - Archivos `.jsonl` con una clave `text` por entrada.
3. Ejemplos de rutas válidas:
   ```bash
   /datasets/spanish_corpus/wikipedia_es.txt
   /datasets/spanish_corpus/red_social.jsonl
   ```
4. Si tienes varios ficheros, concaténalos previamente o apunta al principal mediante `--dataset_path`.

## 3. Lanzar un entrenamiento de 72 h en la A100 (preset automático)

El script `train_tensordock.py` detecta la memoria disponible. Con una A100 de 80 GB seleccionará automáticamente el preset **`aura-72h-max`** (≈2.5B parámetros, contexto 2048) y ajustará los hiperparámetros clave:

- `batch_size = 2`
- `gradient_accumulation_steps = 16` (lote efectivo de 32 secuencias)
- `learning_rate = 2e-4`
- `save_steps = 500`
- `warmup_steps ≈ 3%` del total de pasos de optimización

Ejemplo de ejecución recomendada:

```bash
python AuraLLM/train_tensordock/train_tensordock.py \
    --dataset_path /datasets/spanish_corpus/corpus.txt \
    --output_dir /workspace/aura_runs/aura_spanish \
    --tokenizer_name_or_path spanish-tokenizer \
    --epochs 3 \
    --model_preset auto \
    --gradient_checkpointing \
    --log_interval 100
```

### Qué verás en consola

- Detección de GPU y preset elegido.
- Parámetros totales (≈2.5B) y tokens por paso efectivo (~65k con contexto 2048).
- Throughput medio en tokens/s y ETA estimado.
- Guardado de checkpoints cada 500 pasos y registro `training_log.csv` por época.

> **Sugerencia**: Asegúrate de que el tokenizer en español está disponible (por ejemplo un modelo SentencePiece propio o uno alojado en Hugging Face). El script añadirá automáticamente un token de padding si falta.

### Ajustes opcionales

- `--model_preset aura-72h-extended`: ~1.3B parámetros (2048 contexto) si deseas una opción más ligera.
- `--model_preset manual` junto a `--embed_size`, `--num_layers`, `--num_heads`, `--seq_length` para personalizar totalmente.
- Desactiva `--gradient_checkpointing` si prefieres priorizar velocidad sobre memoria (no recomendado para el preset máximo).
- Cambia `--save_steps` o `--log_interval` para acomodar tus políticas de backup.

## 4. Reanudar un entrenamiento

1. Identifica el checkpoint más reciente, por ejemplo:
   ```bash
   /workspace/aura_runs/aura_spanish/checkpoint-step-4500.pt
   ```
2. Reanuda especificando `--resume_from`:
   ```bash
   python AuraLLM/train_tensordock/train_tensordock.py \
       --dataset_path /datasets/spanish_corpus/corpus.txt \
       --output_dir /workspace/aura_runs/aura_spanish \
       --tokenizer_name_or_path spanish-tokenizer \
       --resume_from /workspace/aura_runs/aura_spanish/checkpoint-step-4500.pt
   ```
3. Se restaurarán modelo, optimizador, `GradScaler`, scheduler y metadatos del preset automáticamente.

## 5. Buenas prácticas operativas

- **Interrupciones controladas**: Usa `Ctrl+C`. El manejador de señales guardará un checkpoint `interrupt-step-XXXX.pt` antes de salir.
- **Monitorización**: Observa `training_log.csv` y la consola para verificar la pérdida y el throughput. Si el ETA supera las 72 h, reduce épocas o ajusta `gradient_accumulation_steps`.
- **Almacenamiento**: Cada checkpoint puede ocupar varios GB. Planea rotación (por ejemplo, conserva 1 de cada 3) o mueve los más antiguos a almacenamiento externo.
- **Warmup y scheduler**: Con el preset automático se activa un warmup lineal (~3 % de pasos). Si necesitas un arranque más largo, especifica manualmente `--warmup_steps`.
- **Validación**: Ajusta `--validation_split` (por defecto 1 %) para medir pérdida de validación periódicamente.
- **Tokenizador**: El directorio de salida siempre incluye el tokenizer actualizado (`tokenizer.json`, `vocab.json`, etc.), `final_model.pt` y `model_config.json` con todos los hiperparámetros usados.

Con estos pasos podrás escalar Aura al mayor tamaño posible en una sola A100 de 80 GB dentro de TensorDock, manteniendo estabilidad y capacidad de recuperación ante interrupciones.
