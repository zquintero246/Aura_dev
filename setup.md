# Guía de configuración para entrenamiento distribuido con AuraLM

Esta guía describe cómo ejecutar `train_gpt2_spanish.py` en dos máquinas (node0 y node1) conectadas a la misma red local vía Wi-Fi. Las instrucciones asumen que se utiliza **Windows 11** con entornos de entrenamiento dentro de **WSL2 (Ubuntu 22.04)**, pero también se indican notas específicas para PowerShell cuando corresponda.

## Requisitos previos

1. **Hardware y sistema operativo**
   - Dos laptops con Windows 11 y soporte de virtualización por hardware habilitado (BIOS/UEFI).
   - Adaptadores Wi-Fi conectados a la misma red inalámbrica y con buena intensidad de señal.
   - GPUs NVIDIA RTX compatibles con CUDA 12.1. Ambas máquinas deben compartir una versión similar de GPU/driver.
   - Al menos 16 GB de RAM y 30 GB de espacio libre en disco por nodo para datasets y checkpoints.

2. **Controladores y WSL**
   - Instalar los controladores NVIDIA Studio/Game Ready más recientes y el paquete **NVIDIA CUDA para WSL**.
   - Habilitar WSL con `wsl --install -d Ubuntu-22.04` y reiniciar el equipo cuando sea requerido.
   - Actualizar WSL: `wsl --update` y reiniciar los entornos (`wsl --shutdown`).

3. **Herramientas de software**
   - Git para clonar el repositorio.
   - Python 3.11 dentro de WSL (instalado mediante `sudo apt install python3.11 python3.11-venv python3.11-dev`).
   - utilidades de red: `ping`, `ip`, `nc` (instalar con `sudo apt install iputils-ping netcat-openbsd`).

4. **Código fuente**
   - Clonar este repositorio en ambas máquinas dentro de WSL, por ejemplo en `/home/usuario/Aura_dev`.

## Configuración de red

1. **Obtener la IP de cada nodo (dentro de WSL):**
   ```bash
   ip addr show wlan0
   ```
   - Toma nota de la dirección IPv4 (`inet 192.168.1.X/24`). node0 será el nodo maestro.
   - Si usas WSL con Hyper-V, asegúrate de que el adaptador virtual esté en modo bridge con tu Wi-Fi. Desde Windows, crea un interruptor virtual en Hyper-V Manager y asígnalo a la interfaz inalámbrica.

2. **Verificar conectividad entre nodos:**
   - En node1 (WSL) ejecuta `ping 192.168.1.10` (sustituye por la IP de node0). Debes recibir respuestas.
   - Comprueba que el firewall de Windows permite tráfico entrante en el puerto que se usará para torchrun (por ejemplo 12355). Agrega una regla en “Firewall de Windows con seguridad avanzada” si es necesario.

3. **Probar apertura de puertos con netcat:**
   - En node0:
     ```bash
     nc -l 12355
     ```
   - En node1:
     ```bash
     echo "hola" | nc 192.168.1.10 12355
     ```
   - node0 debe mostrar el mensaje recibido. Si falla, revisa firewall, puenteo WSL o la IP utilizada.

4. **Sincronizar relojes del sistema:**
   - Verifica que ambos nodos tengan fecha/hora similares (`date`). Grandes diferencias pueden causar expiraciones en el rendezvous TCPStore.

## Configuración del entorno

Realiza los pasos siguientes en **node0** y **node1** dentro de WSL.

1. **Actualizar paquetes y crear entorno virtual:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   cd ~/Aura_dev
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. **Instalar PyTorch 2.5.1 con CUDA 12.1:**
   ```bash
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
       --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Instalar dependencias del proyecto:**
   ```bash
   pip install transformers datasets tqdm
   ```

4. **Verificar acceso a la GPU dentro de WSL:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
   ```

5. **(Opcional) Preparar el dataset en node0 y compartirlo:**
   ```bash
   python AuraLM/Gpt-2PaperTry/train_gpt2_spanish.py --prepare --dataset-name wikipedia \
       --dataset-max-tokens 500000
   ```
   - Copia la carpeta `AuraLM/Gpt-2PaperTry/Data` a node1 para evitar descargas duplicadas (via `rsync` o `scp`).

6. **Variables de entorno recomendadas:**
   - En Windows nativo (no WSL) establece `TORCH_DISTRIBUTED_USE_LIBUV=0` antes de `torchrun`. En WSL no es necesario, pero puedes exportarlo para mantener consistencia:
     ```bash
     export TORCH_DISTRIBUTED_USE_LIBUV=0
     ```
   - Activa logs detallados de PyTorch cuando depures: `export TORCH_DISTRIBUTED_DEBUG=DETAIL`.

## Ejecución del entrenamiento distribuido

Supongamos 2 nodos con 1 GPU cada uno. Ajusta `--nproc_per_node` al número de GPUs disponibles en cada máquina. El puerto maestro será `12355` y la IP del maestro (node0) `192.168.1.10`.

### node0 (rank 0)
```bash
cd ~/Aura_dev
source .venv/bin/activate
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=12355
export WORLD_SIZE=2  # nnodes * nproc_per_node
export RANK=0
export LOCAL_RANK=0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    AuraLM/Gpt-2PaperTry/train_gpt2_spanish.py --distributed --batch-size 8 --seq-len 128
```

### node1 (rank 1)
```bash
cd ~/Aura_dev
source .venv/bin/activate
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=1
export LOCAL_RANK=0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    AuraLM/Gpt-2PaperTry/train_gpt2_spanish.py --distributed --batch-size 8 --seq-len 128
```

Notas:
- Para múltiples GPUs por nodo, aumenta `--nproc_per_node` y deja que `torchrun` gestione `LOCAL_RANK`. El script detectará el índice y sincronizará gradientes automáticamente mediante DDP.
- El backend `nccl` se usa de forma predeterminada cuando hay GPU. Si NCCL falla (por ejemplo, builds sin soporte completo en Windows), el código cae automáticamente a `gloo` y lo indicará en los logs.
- El script imprime por proceso: rank global, local rank, world size, backend efectivo y el nombre de la GPU. También sincroniza explícitamente antes y después del entrenamiento.

## Verificación de sincronización

1. **Prueba rápida sin entrenar modelo:**
   ```bash
   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.1.10 \
       --master_port=12355 AuraLM/Gpt-2PaperTry/test_ddp.py
   ```
   Ejecuta el mismo comando cambiando `--node_rank=1` en node1. Todos los procesos deben reportar el mismo resultado de `all_reduce`.

2. **Durante el entrenamiento:**
   - Observa que los logs muestran "Sincronización inicial" y "Todos los procesos completaron el entrenamiento" para cada rank.
   - Solo rank 0 reporta pérdidas y genera muestras de texto, mientras que los demás procesos trabajan en silencio tras imprimir sus estados iniciales.

3. **Validar checkpoints compartidos:**
   - El archivo `AuraLM/Gpt-2PaperTry/gpt2_spanish.pth` debe generarse únicamente en node0 al finalizar.

## Troubleshooting

- **Timeout en TCPStore o torchrun se queda esperando:**
  - Comprueba conectividad (`ping` y `nc`).
  - Verifica que `MASTER_ADDR` sea accesible desde ambos WSL. Si usas WSL con NAT, habilita el modo bridge en Hyper-V.
  - Asegúrate de que el firewall de Windows permita el puerto `MASTER_PORT` para conexiones entrantes.

- **`RuntimeError: use_libuv was requested but PyTorch was built without libuv support`:**
  - Establece `TORCH_DISTRIBUTED_USE_LIBUV=0` antes de ejecutar `torchrun` (tanto en Windows como en WSL si se reutilizan shells heredadas).

- **`RuntimeError: Distributed package doesn't have NCCL built in` o errores NCCL en Windows:**
  - Los builds oficiales de PyTorch para Windows no incluyen NCCL. El script detectará el fallo y migrará a `gloo`. Comprueba que los logs indiquen `fallback=gloo`.
  - Si usas WSL y aún falla NCCL, revisa que `nvidia-smi` funcione y que el kernel WSL esté actualizado (`wsl --update`).

- **Desincronización de pesos (loss divergente entre nodos):**
  - Verifica que `DistributedSampler` esté activo (los logs lo confirman) y que no haya reinicios parciales de nodos.
  - Si se detecta un NaN, el script reinicia los pesos y los retransmite a todos los procesos.

- **Rutas o versiones distintas del repositorio:**
  - Asegúrate de que ambos nodos ejecuten el mismo commit (`git rev-parse HEAD`) y tengan las mismas dependencias (`pip freeze`).

- **Debug avanzado:**
  - Establece `export NCCL_DEBUG=INFO` y `export TORCH_DISTRIBUTED_DEBUG=DETAIL` para obtener trazas detalladas.
  - Usa `sudo iptables -L -n` dentro de WSL para confirmar que no existan reglas bloqueando tráfico.

Siguiendo estos pasos deberías poder entrenar en paralelo aprovechando las GPUs de ambas laptops mediante PyTorch DistributedDataParallel.
