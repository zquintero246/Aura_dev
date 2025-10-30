# Guía completa para entrenamiento distribuido con AuraLM

Esta guía describe cómo preparar PCs conectadas mediante un switch gigabit **LS1005G** y cableado Cat 6 para entrenar `train_aura.py` usando **DistributedDataParallel (DDP)**. Puedes operar en **Windows 11 + WSL2**

## Requisitos previos

1. **Hardware**
   - GPU NVIDIA con soporte CUDA 12.1 
   - Switch gigabit TP-Link LS1005G, cables Cat 6 directos PC ↔ switch.

2. **Software y versiones**
   - Windows 11 Pro/Enterprise con **Hyper-V** y **WSL2** habilitados
   - Distribución Ubuntu 22.04 (en WSL) con kernel actualizado.
   - Controlador NVIDIA actualizado (versión compatible con CUDA 12.1) e instalación de **NVIDIA CUDA para WSL** o los drivers oficiales de Ubuntu.
   - Python 3.11 dentro del entorno Linux (WSL).
   - PyTorch `2.5.1+cu121`.
   - CUDA Toolkit 12.1.0 y cuDNN 8.9.7 ya incluidos en la build de PyTorch.

3. **Repositorios y dependencias**
   - Clonar este repositorio en `/home/<usuario>/Aura_dev` dentro de cada entorno Linux.
   - Dependencias Python adicionales: `transformers`, `datasets`, `tensorboard`, `psutil`, `pynvml`, `tqdm`.

##  Configuración de red local y sistemas operativos

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
           addresses: [192.168.50.11/24]
           gateway4: 192.168.50.1   # opcional si se comparte internet
           nameservers:
             addresses: [8.8.8.8, 1.1.1.1]
     ```
   - Aplicar cambios: `sudo netplan apply`.
   - Repetir con IPs `.11`, `.12`, `.13`, `.14` para `node1` … `node4`.

3. **Verificar conectividad**
   - Desde cada WSL: `ip addr show eth0` para confirmar IP.
   - Probar ping hacia el maestro (`node0`): `ping -c 4 192.168.50.10`.
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

## Instalación del entorno y dependencias

Ejecutar los pasos siguientes en **cada nodo** dentro del entorno Linux.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev     build-essential git iputils-ping netcat-openbsd
cd ~/Aura_dev
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1+cu121  --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tensorboard psutil tqdm pynvml
```

Validar GPU en Linux:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Clonar datasets (solo una vez en `node0`, luego replicar mediante `rsync` o `scp` si se desea evitar descargas dobles:
```bash
python AuraLM/Gpt-2PaperTry/train_aura.py --prepare  --dataset-name wikipedia --dataset-max-tokens 280000000 --epochs 0
```

## Ejecución del entrenamiento distribuido

1. **Variables comunes en cada nodo (Linux)**
   ```bash
   cd ~/Aura_dev
   source .venv/bin/activate
   export MASTER_ADDR=192.168.50.10      # IP de node0 en Linux
   export MASTER_PORT=12355
   export WORLD_SIZE=5                   # nnodes * nproc_per_node (1 GPU por nodo)
   export TORCH_DISTRIBUTED_USE_LIBUV=0  # redundante pero recomendado en Windows/WSL
   ```

2. **Entrenar un GPT-2 mediano (~350 M) desde cero**

   Este preset replica las dimensiones oficiales del modelo GPT-2 Medium (24 capas, 16 cabezas, 1 024 dimensiones de embedding, secuencia 1 024) y requiere al menos ~10 GB de VRAM libres por GPU cuando se usa `fp16` y acumulación de gradientes. Ajusta la acumulación según la memoria disponible.

   1. **Preparar dataset ampliado (solo `node0`)**
      ```bash
      python train_aura.py --prepare \
        --dataset-name wikipedia --dataset-max-tokens 280000000 --seq-len 1024
      ```
      Replica luego `Data/wikipedia` hacia el resto de nodos con `rsync` o `scp` para evitar descargas repetidas.

   2. **Comando base por nodo (1 GPU por máquina)**

      - **node0** (`NODE_RANK=0`):
        ```bash
        export NODE_RANK=0
        torchrun --nnodes=5 --nproc_per_node=1 --node_rank=$NODE_RANK \
          --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          train_aura.py \
          --backend nccl --use-custom-model --custom-preset gpt2-medium \
          --seq-len 1024 --batch-size 1 --gradient-accumulation-steps 64 \
          --precision fp32 --lr 2e-4 --weight-decay 0.01 --log-dir runs/gpt2m_node0
        ```

      - **node1** (`NODE_RANK=1`):
        ```bash
        export NODE_RANK=1
        torchrun --nnodes=5 --nproc_per_node=1 --node_rank=$NODE_RANK \
          --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          train_aura.py \
          --backend nccl --use-custom-model --custom-preset gpt2-medium \
          --seq-len 1024 --batch-size 1 --gradient-accumulation-steps 64 \
          --precision fp32 --lr 2e-4 --weight-decay 0.01 --log-dir runs/gpt2m_node1
        ```

      - Repite en `node2` … `node4` cambiando únicamente `NODE_RANK` y `--log-dir`.

      - Con estas opciones el batch efectivo por actualización es `1 * 32 * 5 = 160` secuencias de 1 024 tokens. Reduce `--gradient-accumulation-steps` si alguna GPU agota memoria, o incrementa `--batch-size` si usas GPUs con ≥16 GB.

   3. **Verificaciones adicionales**
      - Confirma en los logs que aparece `Config personalizada 'gpt2-medium': d_model=1024, layers=24, heads=16, seq_len=1024`.

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

## Mucha suerte, y que Dios te bendiga.