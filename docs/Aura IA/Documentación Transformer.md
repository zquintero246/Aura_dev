# Documentación técnica y conceptual del modelo autoregresivo tipo GPT-2

## 1. Introducción
El archivo `train_gpt2_spanish.py` implementa un modelo autoregresivo de lenguaje inspirado en GPT-2. "Autoregresivo" significa que el modelo predice el siguiente token observando únicamente el pasado inmediato; para lograrlo emplea una **máscara causal** que impide mirar hacia adelante. A diferencia de la arquitectura encoder-decoder del paper original *Attention Is All You Need*, aquí solo se conserva el **decoder**: una pila de bloques Transformer que, dado un historial de tokens, produce logits para la siguiente palabra. Este enfoque es el núcleo de GPT-2 y otros modelos generativos como GPT-3 o GPT-Neo, ideales para generación continua de texto en español.

## 2. Contexto teórico
- **Atención:** la operación base es \(\text{Attention}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k})V\). Las matrices \(Q\), \(K\) y \(V\) representan "qué busca" cada token, "qué ofrece" y "qué contenido transmite" respectivamente. La normalización con \(\sqrt{d_k}\) estabiliza los gradientes.
- **Self-attention multi-cabeza:** en self-attention los mismos embeddings generan \(Q\), \(K\) y \(V\), permitiendo que cada token se relacione con el resto de la secuencia. Dividimos el espacio de representación en `num_heads` subespacios paralelos; cada cabeza actúa como un grupo de neuronas que se enfoca en patrones distintos (sintaxis, concordancia, semántica). Tras procesar en paralelo, concatenamos y mezclamos la información con una proyección final.
- **Máscara causal:** GPT-2 debe respetar el orden temporal. La matriz triangular inferior registrada en `MultiHeadSelfAttention` establece puntuaciones de \(-\infty\) para conexiones al futuro, obligando a cada token a usar solo el contexto previo. Esta máscara reproduce la causalidad que describe el paper para el decoder.
- **Encoder-decoder vs. solo decoder:** el Transformer original usa un encoder que lee toda la entrada y un decoder que la genera paso a paso. GPT-2 elimina el encoder y reutiliza únicamente el decoder auto-regresivo. Esta simplificación reduce parámetros cuando solo necesitamos modelado de lenguaje sin condicionamiento explícito.
- **Residual, layer norm y FFN:** cada bloque sigue el patrón: normalización → atención → residual, luego normalización → red feed-forward → residual. Las conexiones residuales facilitan el flujo de gradientes; la `LayerNorm` estabiliza las estadísticas de activación; la red FFN introduce no linealidad y transforma cada token independientemente, como un "microperceptrón" que refina la información contextualizada.

## 3. Arquitectura general del modelo
Configuración → Atención → Feed Forward → Bloque Transformer → GPT-2 → Entrenamiento. En forma de diagrama textual:

```
[Input tokens]
     ↓
[Token + Positional Embedding]
     ↓
[12 Transformer Blocks]
     ↓
[LayerNorm + Linear Projection]
     ↓
  [Output logits]
```

Cada bloque replica la estructura descrita en Vaswani et al.: multi-head self-attention seguida de una red feed-forward posicional. Finalmente, un `LayerNorm` y la proyección a vocabulario (mediante peso compartido con el embedding) producen los logits para cada token.

## 4. Explicación del código
- **`Config`** (`train_gpt2_spanish.py`, líneas 16-31): encapsula hiperparámetros globales. Esta clase facilita instanciar modelos con distintos tamaños, como propone el paper (base, medium, large). Ajustar `embed_size`, `num_layers` o `num_heads` cambia proporcionalmente la capacidad de atención.
- **`MultiHeadSelfAttention`** (líneas 34-80): crea las proyecciones lineales `W_q`, `W_k`, `W_v` y la salida. El método `forward` reorganiza tensores a forma `[batch, heads, seq, head_dim]`, calcula la matriz \(QK^\top\), aplica la máscara causal (`register_buffer` con triángulo inferior) y normaliza con softmax antes de combinar con \(V\). El dropout se utiliza tanto en la matriz de atención como en la salida para mitigar sobreajuste.
- **`FFN`** (líneas 83-96): implementa la feed-forward network posicional descrita como "posición-sabia" en el paper: dos lineales con activación GELU, multiplicando el ancho interno por 4 para aumentar la capacidad expresiva, y dropout para regularización.
- **`Transformer`** (líneas 99-110): representa un bloque completo del decoder. Aplica `LayerNorm` antes de cada subcapa (pre-norm), suma residual y repite con la FFN. Esto facilita estabilidad cuando profundizamos la red, siguiendo prácticas modernas posteriores al paper pero consistentes con el objetivo de mantener gradientes bien condicionados.
- **`GPT2`** (líneas 113-132): define la pila de bloques. Primero, embeddings de tokens y posiciones (`pos_embed` aprende desplazamientos en lugar de usar sinusoidales fijas). Luego, 12 bloques secuenciales y una normalización final antes de proyectar de vuelta al vocabulario usando el mismo peso del embedding (weight tying implícito al multiplicar por `token_embed.weight.t()`).
- **`SpanishCorpus`** (líneas 135-149): dataset que produce pares `(entrada, objetivo)` desplazados en un token, emulando el aprendizaje autoregresivo con cross-entropy. Cada muestra es una ventana deslizante de longitud `seq_length`.
- **`sample`** (líneas 152-174): genera texto autoregresivo. Recorta el contexto al máximo permitido, obtiene logits del modelo, aplica temperatura (controla aleatoriedad; <1 produce texto más conservador) y realiza muestreo multinomial. Esto imita el proceso de inferencia token a token.
- **Funciones auxiliares**: `reset_parameters` reinicia pesos tras NaNs; `evaluate` (líneas 180-205) calcula pérdida de validación con AMP opcional; `save_checkpoint` (líneas 208-222) persiste estado del entrenamiento para recuperación.
- **`train`** (líneas 225-317): orquesta el entrenamiento con AdamW, warmup lineal y AMP (GradScaler) para aprovechar GPU RTX 3050. Incluye clipping de gradiente, reinicio automático si surge NaN, validación por epoch, generación de muestra y guardado de checkpoints periódicos.
- **Preparación de datos**: `tokenize_text_stream` (líneas 337-372) recorre los textos del dataset de Hugging Face, codifica con GPT-2 tokenizer y añade el token EOS; `prepare_hf_dataset` (líneas 375-436) descarga `oscar` o `wikimedia/wikipedia`, tokeniza, divide en entrenamiento/validación y guarda tensores.
- **`main`** (líneas 515-607): parsea argumentos CLI (batch size, LR, dataset, etc.), inicializa tokenizer, asegura la preparación de datos, construye `Config`, crea `DataLoader`, selecciona dispositivo, entrena el modelo y guarda el checkpoint final `gpt2_spanish.pth`.

## 5. Flujo de entrenamiento
`SpanishCorpus` convierte el vector continuo de tokens en ejemplos con entradas y objetivos desplazados. Los `DataLoader` agrupan estos ejemplos en lotes. En `train`, cada paso ejecuta: forward con AMP, cálculo de pérdida de entropía cruzada (comparando logits vs. tokens objetivo), retropropagación escalada, clipping de gradientes y actualización del optimizador. Tras cada epoch, `evaluate` mide la pérdida en validación (sin gradientes) y `sample` muestra una generación autoregresiva para evaluar cualitativamente el aprendizaje. El flujo entero se controla con `tqdm`, checkpoints y reinicio si aparece NaN.

## 6. Resumen de hiperparámetros importantes
- `vocab_size`: tamaño del vocabulario del tokenizer GPT-2 (50257). Define la dimensión de la proyección de salida; mayor vocabulario implica logits más amplios.
- `embed_size`: dimensión de los embeddings y representaciones internas. Controla la capacidad del modelo; valores mayores permiten capturar patrones complejos, pero incrementan memoria y tiempo.
- `num_heads`: número de cabezas de atención. Cada cabeza procesa subespacios de tamaño `embed_size / num_heads`. Más cabezas permiten diversidad de focos atencionales.
- `num_layers`: profundidad de la pila de bloques Transformer. Más capas agregan múltiples pasos de refinamiento contextual.
- `max_seq_length`: longitud máxima de contexto. Determina el tamaño de la máscara causal y la tabla de embeddings posicionales; fija cuántos tokens previos puede recordar el modelo.
- `dropout`: probabilidad de apagar neuronas durante entrenamiento. Regulariza tanto atención como FFN para evitar overfitting.
- `learning rate` y `batch size`: LR controla el tamaño de los pasos de optimización; se combina con warmup para estabilizar el inicio. Batch size define cuántos ejemplos se procesan en paralelo, afectando la estimación del gradiente y el aprovechamiento de la GPU.

## 7. Conclusión y mejoras posibles
El script implementa un Transformer solo-decoder coherente con GPT-2: embeddings aprendidos, atención multi-cabeza con máscara causal, residuales con normalización previa, FFN posicional y entrenamiento completo con AMP, warmup y checkpoints. Algunas extensiones recomendadas:
- Implementar weight tying explícito (actualmente implícito al reutilizar el embedding; podría refactorizarse para claridad).
- Añadir una capa de logits independiente para experimentar con bias o escalados adicionales.
- Probar embeddings sinusoidales o rotary (RoPE) para una mejor generalización a secuencias largas.
- Aplicar clipping dinámico o programaciones de learning rate más avanzadas (cosine, OneCycle).
- Automatizar guardado de checkpoints según métrica de validación y explorar fine-tuning con corpora más grandes o específicos del dominio.
