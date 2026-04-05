"""
check_cuda.py - Print basic PyTorch/CUDA environment information.

Usage:
    python check_cuda.py
"""

import torch


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda or 'No'}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("Selected device: cpu")
        return

    device_count = torch.cuda.device_count()
    current_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_idx)

    print(f"CUDA device count: {device_count}")
    print(f"Current CUDA device index: {current_idx}")
    print(f"Current CUDA device name: {device_name}")

    if torch.backends.cudnn.is_available():
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")


if __name__ == "__main__":
    main()
