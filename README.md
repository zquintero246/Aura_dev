# GPT-2 Español

Proyecto para entrenar y monitorizar un modelo tipo GPT-2 en español con PyTorch.

- El flujo de trabajo oficial se ejecuta dentro de contenedores Docker con GPU NVIDIA.
- Consulta `docs/DOCKER_TRAINING_SETUP.md` para instrucciones de ejecución en una o varias máquinas.
- Consulta `docs/ARQUITECTURA.md` para detalles técnicos de la arquitectura y del pipeline de entrenamiento.

Scripts principales en la raíz del repositorio:

- `train_gpt2_spanish.py`: entrenamiento distribuido, checkpoints, generación y TensorBoard.
- `test_ddp.py`: verificación rápida de `torch.distributed` entre nodos.
- `monitor.py`: monitorización ligera de CPU/RAM/GPU.
- `ddp_launch.sh`: ayudante para lanzar `torchrun` con variables de entorno.
