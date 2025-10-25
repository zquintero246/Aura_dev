# Guía completa para entrenamiento distribuido (5 nodos) con AuraLM

Esta guía describe cómo preparar cinco PCs con **Windows 11 + WSL (Ubuntu 22.04)** conectadas mediante un switch gigabit **LS1005G** y cableado Cat 6 para entrenar `train_gpt2_spanish.py` usando **DistributedDataParallel (DDP)**.

## ✅ Requisitos previos

1. **Hardware**
   - Cinco PCs denominadas `node0` … `node4` con CPU de al menos 8 núcleos, 32 GB de RAM y almacenamiento libre ≥ 100 GB.
   - GPU NVIDIA con soporte CUDA 12.1 (idealmente al menos 1 GPU por nodo). Si un nodo tiene varias GPUs, `torchrun` las gestionará vía `--nproc_per_node`.
   - Switch gigabit TP-Link LS1005G, cables Cat 6 directos PC ↔ switch.

2. **Software y versiones**
   - Windows 11 Pro/Enterprise con **Hyper-V** y **WSL2** habilitados.
   - Distribución Ubuntu 22.04 en WSL con kernel actualizado (`wsl --update`).
   - Controlador NVIDIA actualizado (versión compatible con CUDA 12.1) e instalación de **NVIDIA CUDA para WSL**.
   - Python 3.11 dentro de WSL.
   - PyTorch `2.5.1+cu121`, torchvision `0.20.1+cu121`, torchaudio `2.5.1+cu121`.
   - CUDA Toolkit 12.1.0 y cuDNN 8.9.7 ya incluidos en la build de PyTorch.

3. **Repositorios y dependencias**
   - Clonar este repositorio en `/home/<usuario>/Aura_dev` dentro de cada WSL.
   - Dependencias Python adicionales: `transformers`, `datasets`, `tensorboard`, `psutil`, `pynvml` (opcional para métricas avanzadas), `tqdm`.

## ⚙️ Configuración de red local y WSL

1. **Asignar IPs estáticas en Windows (adaptador vEthernet Hyper-V)**
   - Abrir *Centro de redes y recursos compartidos → Cambiar configuración del adaptador*.
   - Para cada PC, configurar una IP estática siguiendo la tabla (máscara `/24` y gateway opcional si solo se usa LAN interna):

     | Nodo  | IP Windows/Hyper-V |
     |-------|--------------------|
     | node0 | 192.168.50.10      |
     | node1 | 192.168.50.11      |
     | node2 | 192.168.50.12      |
     | node3 | 192.168.50.13      |
     | node4 | 192.168.50.14      |

2. **Exponer IP en WSL**
   - En Windows PowerShell (admin) crear un conmutador virtual externo asociado a la NIC física:
     ```powershell
     New-VMSwitch -Name "LanDDP" -NetAdapterName "Ethernet" -AllowManagementOS $true
     ```
   - Asignar este switch al adaptador WSL: `Set-VMSwitch -Name "WSL" -NetAdapterName "LanDDP"`.
   - Reiniciar WSL: `wsl --shutdown` y volver a abrir Ubuntu.
   - En WSL, crear un archivo `/etc/netplan/01-static.yaml` con el bloque apropiado para la interfaz (`eth0` o `ens33`). Ejemplo para `node0`:
     ```yaml
     network:
       version: 2
       ethernets:
         eth0:
           dhcp4: false
           addresses: [192.168.50.20/24]
           gateway4: 192.168.50.1   # opcional si se comparte internet
           nameservers:
             addresses: [8.8.8.8, 1.1.1.1]
     ```
   - Aplicar cambios: `sudo netplan apply`.
   - Repetir con IPs `.21`, `.22`, `.23`, `.24` para `node1` … `node4`.

3. **Verificar conectividad**
   - Desde cada WSL: `ip addr show eth0` para confirmar IP.
   - Probar ping hacia el maestro (`node0`): `ping -c 4 192.168.50.20`.
   - Verificar apertura de puertos (usar `nc`):
     ```bash
     # En node0 (WSL)
     nc -l 12355
     # En node1 (WSL)
     echo "hola" | nc 192.168.50.20 12355
     ```
   - Si no hay respuesta, revisar firewall de Windows y reglas NAT en Hyper-V.

4. **Sincronizar hora del sistema**
   - Ejecutar en cada WSL: `sudo timedatectl set-ntp true`.

## 🧩 Instalación del entorno y dependencias

Ejecutar los pasos siguientes en **cada nodo** dentro de WSL.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev     build-essential git iputils-ping netcat-openbsd
cd ~/Aura_dev
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121     --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tensorboard psutil tqdm pynvml
```

Validar GPU en WSL:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Clonar datasets (solo una vez en `node0`, luego replicar mediante `rsync` o `scp` si se desea evitar descargas dobles):
```bash
python AuraLM/Gpt-2PaperTry/train_gpt2_spanish.py --prepare     --dataset-name wikipedia --dataset-max-tokens 800000
```

## 🔥 Ejecución del entrenamiento distribuido

1. **Variables comunes en cada nodo (WSL)**
   ```bash
   cd ~/Aura_dev
   source .venv/bin/activate
   export MASTER_ADDR=192.168.50.20      # IP de node0 en WSL
   export MASTER_PORT=12355
   export WORLD_SIZE=5                   # nnodes * nproc_per_node (1 GPU por nodo)
   export TORCH_DISTRIBUTED_USE_LIBUV=0  # redundante pero recomendado en Windows/WSL
   ```

2. **Lanzar `torchrun` en cada nodo**

   - **node0** (`NODE_RANK=0`):
     ```bash
     export NODE_RANK=0
     torchrun --nnodes=5 --nproc_per_node=1 --node_rank=$NODE_RANK        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT        AuraLM/Gpt-2PaperTry/train_gpt2_spanish.py        --backend nccl --batch-size 8 --seq-len 128 --log-dir runs/node0
     ```

   - **node1** (`NODE_RANK=1`):
     ```bash
     export NODE_RANK=1
     torchrun --nnodes=5 --nproc_per_node=1 --node_rank=$NODE_RANK        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT        AuraLM/Gpt-2PaperTry/train_gpt2_spanish.py        --backend nccl --batch-size 8 --seq-len 128 --log-dir runs/node1
     ```

   - Repetir análogamente en `node2`, `node3`, `node4` ajustando `NODE_RANK` y `--log-dir`.
   - Para habilitar media precisión en GPUs compatibles añade `--precision fp16` (o `--precision bf16` en hardware con soporte). Por defecto el entrenamiento usa `fp32` para maximizar la estabilidad y evitar errores de gradientes.

3. **Ejemplo para nodos con múltiples GPUs**
   - Si `node0` y `node1` tienen 2 GPUs, usar `--nproc_per_node=2` y NO establecer `LOCAL_RANK` manualmente (lo hace `torchrun`).
   - `WORLD_SIZE` debe ser `nnodes * nproc_per_node`. Con 2 GPUs en dos nodos y 1 GPU en los restantes: `WORLD_SIZE=8`.

4. **Logs esperados en consola**
   ```
[Rank 0/5 | Node node0 | Device cuda:0] Modo Distribuido. Backend=nccl. Sincronizando...
[Rank 0/5 | Node node0 | Device cuda:0] Precisión activa: fp32 | Autocast=OFF | GradScaler=OFF
[Rank 3/5 | Node node3 | Device cuda:0] Epoch 1 | Step 200 | Loss: 2.1345 | Sync=OK
[Rank 0/5 | Node node0 | Device cuda:0] Epoch 1/5 | Loss entrenamiento: 2.0813 | Loss validación: 2.0456 | Duración epoch: 318.4s
[Rank 0/5 | Node node0 | Device cuda:0] Entrenamiento completado. Gradientes sincronizados.
   ```
   - Observe cómo cada mensaje incluye `Rank`, `Node`, `Device` y el estado de sincronización (`Sync=OK`).

5. **Validar sincronización de gradientes**
   - Ejecutar el script de prueba antes del entrenamiento:
     ```bash
     torchrun --nnodes=5 --nproc_per_node=1 --node_rank=$NODE_RANK        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT        AuraLM/Gpt-2PaperTry/test_ddp.py --backend nccl
     ```
   - Todos los nodos deben imprimir el mismo resultado de `all_reduce`. Durante el entrenamiento, revise que `Gradientes sincronizados` aparezca al final en cada proceso.

## 🧠 Monitoreo remoto (TensorBoard y CLI)

1. **TensorBoard**
   - En `node0` (o en cualquier nodo con acceso a los logs), lanzar:
     ```bash
     source ~/Aura_dev/.venv/bin/activate
     tensorboard --logdir runs --host=0.0.0.0 --port=6006
     ```
   - Desde otra máquina abrir `http://192.168.50.20:6006`.
   - Métricas disponibles:
     - `Loss/train_step`, `Loss/train_epoch`, `Loss/val_epoch`
     - `System/CPU_percent`, `System/RAM_percent`, `System/GPU_VRAM_MB`
     - `Time/epoch_seconds`
   - Ejemplo de gráfico esperado: disminución progresiva de `Loss/train_epoch` y convergencia de `Loss/val_epoch`.

2. **Monitor CLI en tiempo real**
   - El script `AuraLM/Gpt-2PaperTry/monitor.py` imprime uso de CPU/GPU/RAM cada pocos segundos.
     ```bash
     python AuraLM/Gpt-2PaperTry/monitor.py --interval 4
     ```
   - Salida típica:
     ```
[2024-05-21 22:13:04] CPU: 68.5% | RAM: 72.1% (47.9 GB / 66.4 GB) | RAM proceso: 2.3 GB
  - GPU 0 (NVIDIA RTX 4090) | Utilización: 89% | Memoria: 12.1 GB / 24.0 GB
     ```
   - Para una única lectura: `python monitor.py --once`.

## 🧯 Troubleshooting común

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| `RuntimeError: Connection timed out during rendezvous` | IP/puerto incorrectos, firewall bloqueando | Verificar `MASTER_ADDR`, `MASTER_PORT`, probar `nc`, abrir puerto en firewall de Windows. |
| `RuntimeError: NCCL error` en WSL | Build sin NCCL o driver desactualizado | El script cambia automáticamente a `gloo`. Asegurarse de que `nvidia-smi` funcione y el kernel WSL esté actualizado. |
| Gradientes no sincronizados (loss divergente) | `WORLD_SIZE` incorrecto, `NODE_RANK` repetido o dataset no compartido | Confirmar valores exportados en cada nodo, limpiar cachés y repetir `torchrun`. Revisar que los logs indiquen `Sync=OK`. |
| `Address already in use` al arrancar | Puerto 12355 ocupado | Cambiar `MASTER_PORT` en todas las máquinas (`export MASTER_PORT=22345`). |
| TensorBoard no accesible desde otra PC | Puerto bloqueado o host incorrecto | Ejecutar `tensorboard --host=0.0.0.0`, validar que el navegador apunte a la IP del maestro (`http://192.168.50.20:6006`). |
| Monitor CLI sin métricas de GPU | Falta NVML o GPU no visible en WSL | Instalar `pynvml` (`pip install nvidia-ml-py`) y confirmar `nvidia-smi`. |
| `ValueError: Attempting to unscale FP16 gradients` | Uso de FP16 en hardware/driver no estable | Relanzar con `--precision fp32` (valor por defecto) o actualizar controladores CUDA. |

---

Con esta configuración los cinco nodos colaboran mediante PyTorch DDP, utilizando `DistributedSampler`, registro de métricas en TensorBoard y monitoreo remoto vía CLI. Ajusta `WORLD_SIZE`, `NODE_RANK` y `--nproc_per_node` según la cantidad real de GPUs disponibles por nodo.
