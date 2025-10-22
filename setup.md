# Guía de configuración para entrenamiento distribuido con AuraLM

Esta guía describe cómo ejecutar `train_gpt2_spanish.py` en dos laptops con Windows 11 (denominadas **node0** y **node1**) conectadas a la misma red Wi-Fi. El procedimiento cubre desde la verificación de la red hasta la ejecución del entrenamiento distribuido y la validación de la sincronización de gradientes.

## Requisitos previos

1. **Hardware**
   - Dos laptops con Windows 11 conectadas a la misma red Wi-Fi.
   - Cada laptop debe tener al menos una GPU NVIDIA compatible con CUDA 12.1 (si solo hay CPU el script seguirá funcionando, pero el rendimiento será menor).
   - Al menos 20 GB de espacio libre en disco para datasets y checkpoints.

2. **Software**
   - Python 3.11 instalado en ambas máquinas (`py --version` debe mostrar 3.11.x).
   - Git instalado para clonar este repositorio.
   - Drivers NVIDIA actualizados (versión 531.xx o superior compatible con CUDA 12.1).
   - Conectividad entre ambas máquinas (sin reglas de firewall que bloqueen el puerto que se usará para el entrenamiento).

3. **Código fuente**
   - Clona el repositorio en ambas máquinas y coloca los archivos dentro de la misma ruta, por ejemplo `C:\proyectos\Aura_dev`.

## Configuración de red

1. **Conectar ambas máquinas a la misma red Wi-Fi.**
2. **Obtener la dirección IP de node0 (máquina que actuará como maestro):**
   - Abre PowerShell y ejecuta `ipconfig`.
   - Localiza la interfaz Wi-Fi y anota la dirección IPv4, por ejemplo `192.168.0.10`.
3. **Verificar conectividad desde node1:**
   - En node1 ejecuta `ping 192.168.0.10`.
   - Si no hay respuesta, revisa la red o el firewall (habilita temporalmente el permiso de entrada para `python.exe` o el puerto que se usará).
4. **Elegir un puerto libre para la comunicación distribuida (ej. `12355`).**
   - Asegúrate de que el puerto esté abierto en el firewall de node0 (puedes crear una regla de entrada para TCP 12355 si es necesario).

## Configuración de entorno

Realiza los siguientes pasos tanto en **node0** como en **node1**.

1. **Crear y activar un entorno virtual:**
   ```powershell
   cd C:\proyectos\Aura_dev
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
2. **Instalar PyTorch 2.5.1 con CUDA 12.1:**
   ```powershell
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Instalar dependencias del proyecto:**
   ```powershell
   pip install transformers datasets tqdm
   ```
4. **(Opcional) Preparar el dataset previamente:**
   - Puedes ejecutar en node0: `python AuraLM\Gpt-2PaperTry\train_gpt2_spanish.py --prepare --dataset-name wikipedia --dataset-max-tokens 500000`
   - Cuando termine la tokenización, copia la carpeta `AuraLM\Gpt-2PaperTry\Data` a node1 para evitar descargar dos veces.

## Ejecución del entrenamiento distribuido

A continuación se asume que cada laptop usa una sola GPU. Si dispones de más GPUs por nodo, ajusta el valor de `--nproc_per_node` con la cantidad de GPUs de ese nodo.

1. **Definir variables comunes:**
   - Dirección IP de node0: `192.168.0.10` (ejemplo).
   - Puerto maestro: `12355`.
   - Número total de procesos: `WORLD_SIZE = nnodes * nproc_per_node`. Con 2 nodos y 1 GPU por nodo, `WORLD_SIZE = 2`.

2. **Comandos a ejecutar (PowerShell) en cada nodo:**

   **node0 (maestro, rank 0):**
   ```powershell
   cd C:\proyectos\Aura_dev
   .\.venv\Scripts\activate
   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 `
       --master_addr=192.168.0.10 --master_port=12355 `
       AuraLM\Gpt-2PaperTry\train_gpt2_spanish.py --distributed --batch-size 8 --seq-len 128
   ```

   **node1 (rank 1):**
   ```powershell
   cd C:\proyectos\Aura_dev
   .\.venv\Scripts\activate
   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 `
       --master_addr=192.168.0.10 --master_port=12355 `
       AuraLM\Gpt-2PaperTry\train_gpt2_spanish.py --distributed --batch-size 8 --seq-len 128
   ```

   > Nota: el script ajusta automáticamente el backend distribuido a **gloo** en Windows y permite sobreescribir `MASTER_ADDR`, `MASTER_PORT`, `RANK` y `WORLD_SIZE` mediante variables de entorno o argumentos CLI.

3. **Logs esperados:**
   - Cada proceso imprime una línea `"[Rank X] cudnn.benchmark=..."` indicando su rank y dispositivo.
   - Solo el proceso con `rank 0` muestra las pérdidas de entrenamiento y validación, así como ejemplos de texto generado.

## Verificación de la sincronización de gradientes

1. Ejecuta un par de épocas y revisa que no haya divergencias de pérdida (por ejemplo, valores `NaN`).
2. Observa que los pasos de entrenamiento avanzan sin errores y que el `Loss entrenamiento` disminuye progresivamente en la consola de node0.
3. Para un chequeo más detallado puedes activar el modo de depuración de PyTorch antes de lanzar `torchrun`:
   ```powershell
   $env:TORCH_DISTRIBUTED_DEBUG = "DETAIL"
   ```
   Esto generará registros adicionales que confirman la sincronización de los gradientes y los all-reduce entre procesos.
4. Al finalizar, verifica que el archivo `gpt2_spanish.pth` se genere en `AuraLM\Gpt-2PaperTry\` únicamente en node0 (el proceso maestro es el encargado de guardarlo).

## Troubleshooting común

- **Ping fallido o tiempo de espera agotado:**
  - Asegúrate de que ambas laptops estén en la misma subred Wi-Fi.
  - Desactiva temporalmente el firewall o agrega una excepción para `python.exe` y el puerto `MASTER_PORT` seleccionado.

- **Error "Address already in use" al iniciar torchrun:**
  - Cambia el valor de `--master_port` por uno libre (ej. 12360) y vuelve a lanzar ambos comandos.

- **`RuntimeError: Distributed package doesn't have NCCL built in`:**
  - En Windows, NCCL no está disponible. El script selecciona automáticamente el backend `gloo`, pero asegúrate de no forzar `--master-port` o `--backend nccl` manualmente.

- **Errores relacionados con CUDA o drivers:**
  - Actualiza los drivers NVIDIA, verifica que `nvidia-smi` muestre la GPU correctamente y que la versión de CUDA 12.1 esté instalada.
  - Comprueba que el entorno virtual esté usando la versión correcta de PyTorch (`python -c "import torch; print(torch.__version__)"`).

- **Diferencias de versión entre nodos:**
  - Usa el mismo commit del repositorio y las mismas versiones de dependencias en ambas máquinas. Puedes ejecutar `pip freeze > requirements.txt` en node0 y replicarlo en node1.

Siguiendo estos pasos deberías poder entrenar el modelo en paralelo aprovechando las GPUs disponibles en tus dos laptops.
