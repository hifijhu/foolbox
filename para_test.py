import torch
import torch.multiprocessing as mp
import time

# Long, heavy GPU workload per process
def worker(local_rank, duration_sec):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[GPU {local_rank}] started", flush=True)

    # Very large tensors to force 100% utilization
    a = torch.randn(16384, 16384, device=device)
    b = torch.randn(16384, 16384, device=device)

    torch.cuda.synchronize()
    end_time = time.time() + duration_sec

    # Run continuously for a fixed duration
    while time.time() < end_time:
        c = a @ b

    torch.cuda.synchronize()
    print(f"[GPU {local_rank}] finished", flush=True)


def main():
    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found")

    duration_sec = 60  # run long enough for monitoring

    mp.spawn(
        worker,
        args=(duration_sec,),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    main()