from __future__ import annotations

import torch


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"device_count={torch.cuda.device_count()}")
        print(f"device_name={torch.cuda.get_device_name(0)}")
        print(f"capability={torch.cuda.get_device_capability(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        print(f"smoke_matmul_sum={float(y.sum().detach().cpu()):.4f}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
