"""Prueba mínima de torch.distributed para verificar la comunicación entre nodos."""

import argparse
import os

import torch
import torch.distributed as dist

if os.name == "nt" and "TORCH_DISTRIBUTED_USE_LIBUV" not in os.environ:
    # Igual que en el script de entrenamiento, evitamos el backend basado en libuv en
    # builds de PyTorch para Windows que no lo incluyen.
    os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba de sincronización DDP")
    parser.add_argument(
        "--backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Backend de comunicación a utilizar (auto selecciona nccl si hay GPU).",
    )
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--master-port", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    return parser.parse_args()


def initialize_process_group(backend: str, rank: int, world_size: int) -> str:
    preferred_backend = backend
    if backend == "auto":
        preferred_backend = "nccl" if torch.cuda.is_available() else "gloo"

    try:
        dist.init_process_group(preferred_backend, init_method="env://", rank=rank, world_size=world_size)
        return preferred_backend
    except RuntimeError as exc:
        if preferred_backend == "nccl":
            print(
                f"[Rank {rank}] Error inicializando NCCL ({exc}). Reintentando con 'gloo'.",
                flush=True,
            )
            dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world_size)
            return "gloo"
        raise


def main() -> None:
    args = parse_args()

    if args.master_addr is not None:
        os.environ["MASTER_ADDR"] = str(args.master_addr)
    if args.master_port is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)
    if args.world_size is not None:
        os.environ["WORLD_SIZE"] = str(args.world_size)
    if args.rank is not None:
        os.environ["RANK"] = str(args.rank)
    if args.local_rank is not None:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = initialize_process_group(args.backend, rank, world_size)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = local_rank if 0 <= local_rank < device_count else 0
        if local_rank >= device_count:
            print(
                f"[Rank {rank}] LOCAL_RANK {local_rank} fuera de rango. Se usará la GPU 0.",
                flush=True,
            )
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")

    ones = torch.ones(1, device=device)
    dist.all_reduce(ones, op=dist.ReduceOp.SUM)
    print(
        f"[Rank {rank}] Backend activo: {backend}. Resultado del all_reduce: {ones.item()} (WORLD_SIZE={world_size}).",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
