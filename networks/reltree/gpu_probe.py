"""Run inside a Slurm allocation to verify each assigned GPU, including autograd."""
import json
import os
import socket

import torch


def main():
    print(json.dumps({"host": socket.gethostname(), "job_id": os.getenv("SLURM_JOB_ID"),
                      "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                      "torch": torch.__version__, "cuda_build": torch.version.cuda}, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Slurm allocation did not expose a usable CUDA device")
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        x = torch.randn(256, 256, device=device, requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        print(json.dumps({"gpu": i, "name": torch.cuda.get_device_name(i),
                          "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                          "forward_backward": "passed", "loss": loss.item()}), flush=True)


if __name__ == "__main__":
    main()
