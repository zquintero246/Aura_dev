# Guía de entrenamiento distribuido con Docker y GPU

Esta guía describe cómo preparar y ejecutar el proyecto **GPT-2 en español** dentro de contenedores Docker con GPU NVIDIA. Todas las máquinas deben comunicarse por LAN directa (switch o router) y tener instalados los drivers oficiales de NVIDIA.

## A. Preparar cada máquina

1. **Actualizar drivers NVIDIA**
   - En Windows instala los controladores *Game Ready* o *Studio* más recientes.
   - En Linux usa los repositorios oficiales de NVIDIA (`sudo apt install nvidia-driver-535` o superior) y reinicia.
2. **Instalar Docker**
   - Windows: instala **Docker Desktop** y habilita el backend Hyper-V o WSL2 (solo para Docker). Activa la opción *Use the WSL 2 based engine* y reinicia.
   - Linux: instala Docker Engine siguiendo la guía de [docs.docker.com](https://docs.docker.com/engine/install/).
3. **Habilitar soporte GPU en Docker**
   - Windows: en Docker Desktop ve a *Settings → Resources → GPU* y habilita **Enable GPU support**. Docker Desktop utiliza WSL2 internamente, pero no es necesario trabajar dentro de WSL; todo el entrenamiento se hará en contenedores.
   - Linux: instala **NVIDIA Container Toolkit** (`sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
4. **Clonar este repositorio** en **la misma ruta** de todas las máquinas, por ejemplo `~/proyectos/gpt2-spanish`.
5. **Construir la imagen Docker localmente** en cada nodo:

   ```bash
   docker compose build
   ```

   Esto descarga la imagen base `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`, instala las dependencias y prepara el entorno `/workspace`.

## B. Verificar la GPU dentro del contenedor

Comprueba que Docker ve la GPU antes de entrenar:

```bash
docker compose run --rm trainer nvidia-smi
```

Deberías ver la GPU (ej. `NVIDIA GeForce RTX 3050 Laptop GPU`) y la memoria total disponible.

## C. Entrenamiento en una sola máquina (world_size = 1)

Para pruebas locales sin distribución entre nodos:

```bash
docker compose run --rm trainer bash -lc "python train_gpt2_spanish.py --epochs 1 --batch-size 2 --seq-len 128"
```

Este comando ejecuta el entrenamiento en modo de proceso único (`world_size=1`). El script detecta automáticamente si hay GPU disponible (`cuda:0`) y mantiene la misma lógica de logging, checkpoints y TensorBoard.

## D. Entrenamiento distribuido en varias máquinas

1. **Elegir la máquina maestra** (rank 0) y anotar su IP LAN, por ejemplo `192.168.1.50`. Asegúrate de que el puerto 12355 esté abierto en el firewall del host.
2. **Exportar las variables de entorno** en cada nodo antes de arrancar el contenedor. Ajusta `NODE_RANK` para cada máquina:

   ```bash
   MASTER_ADDR=192.168.1.50 \
   MASTER_PORT=12355 \
   NNODES=5 \
   NODE_RANK=0 \
   GPUS_PER_NODE=1 \
   docker compose run --rm trainer bash -lc "./ddp_launch.sh"
   ```

   - En la máquina maestra usa `NODE_RANK=0`.
   - En las máquinas siguientes usa `NODE_RANK=1`, `NODE_RANK=2`, etc.
   - Mantén `MASTER_ADDR`, `MASTER_PORT`, `NNODES` y `GPUS_PER_NODE` iguales en todos los nodos.
3. **Cómo funciona**
   - `ddp_launch.sh` invoca `torchrun` con los parámetros anteriores.
   - `train_gpt2_spanish.py` inicializa `torch.distributed`, sincroniza gradientes mediante `all_reduce` y guarda checkpoints exclusivamente desde el proceso de rank 0.
4. **Probar la conectividad con test_ddp.py**

   ```bash
   MASTER_ADDR=192.168.1.50 MASTER_PORT=12355 \
   WORLD_SIZE=5 RANK=0 LOCAL_RANK=0 \
   docker compose run --rm trainer bash -lc "python test_ddp.py --backend auto"
   ```

   Ajusta `RANK` en cada nodo y revisa que el script imprima el backend (`nccl` o `gloo`) y el resultado correcto del `all_reduce`.

## E. Monitorización de recursos

Utiliza el monitor ligero para inspeccionar el uso de CPU, RAM y GPU desde el contenedor:

```bash
docker compose run --rm trainer bash -lc "python monitor.py --interval 5"
```

Esto muestra estadísticas cada 5 segundos. Es ideal para detectar desequilibrios entre nodos o saturación de VRAM.

## F. Checkpoints y directorios de trabajo

- Todos los checkpoints se almacenan en `./checkpoints` dentro del contenedor, que está montado en la carpeta local gracias al volumen `.:/workspace`.
- Solo el proceso de rank 0 guarda checkpoints para evitar sobrescrituras concurrentes.
- Los datasets preparados se guardan en `./Data`. Puedes preprocesarlos en un nodo y compartir la carpeta con el resto para ahorrar tiempo de descarga.

Con esto el entorno queda listo para entrenar un GPT-2 en español desde cero o continuar entrenamientos existentes en un clúster Docker multinodo.
