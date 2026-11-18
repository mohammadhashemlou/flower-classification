"""Configure CUDA DLL search paths on Windows before TensorFlow is imported.

TensorFlow on Windows loads CUDA/cuDNN DLLs at import time via the standard
DLL search order. When CUDA lives in a non-default location, or when cuDNN
is installed as a pip wheel (``nvidia-cudnn-cu12``), those directories are
not on ``PATH`` and TF silently falls back to CPU.

Import this module **before** ``import tensorflow``:

    from gpu_env import setup_cuda_dlls
    setup_cuda_dlls()

    import tensorflow as tf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_dirs() -> list[Path]:
    """Return candidate directories that may contain CUDA/cuDNN DLLs."""
    dirs: list[Path] = []

    if os.name != "nt":
        return dirs

    # 1. System CUDA toolkit (CUDA_PATH, or the default install root)
    cuda_roots = [
        Path(os.environ.get("CUDA_PATH", "")),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
    ]
    for root in cuda_roots:
        if not root or not root.exists():
            continue
        # If the root itself is a versioned dir (e.g. ...\CUDA\v12.6), use it.
        if (root / "bin").exists():
            dirs.append(root / "bin")
        # Otherwise look at versioned children, newest first.
        for child in sorted(root.iterdir(), reverse=True):
            if child.is_dir() and (child / "bin").exists():
                dirs.append(child / "bin")

    # 2. pip-installed NVIDIA wheels (nvidia-*-cu11 / nvidia-*-cu12)
    # Wheel layouts differ: some ship DLLs in ``bin/``, others in ``lib/x64/``.
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in (
        "cudnn",
        "cublas",
        "cuda_runtime",
        "cuda_nvrtc",
        "cufft",
        "cusparse",
        "cusolver",
        "curand",
        "nvjitlink",
        "nccl",
    ):
        package = site_packages / sub
        if not package.exists():
            continue
        dirs.append(package / "bin")
        dirs.append(package / "lib" / "x64")
        dirs.append(package / "lib")

    return dirs


def setup_cuda_dlls(verbose: bool = True) -> list[str]:
    """Register CUDA DLL directories for the current process.

    Args:
        verbose: Print each directory that was registered.

    Returns:
        The list of directories that were added (as strings).
    """
    added: list[str] = []

    for directory in _candidate_dirs():
        if not directory.exists():
            continue
        # Verify it actually holds CUDA DLLs before registering
        has_dlls = any(
            f.name.startswith(
                ("cudart", "cublas", "cudnn", "cufft", "cusparse", "cusolver", "curand")
            )
            for f in directory.glob("*.dll")
        )
        if not has_dlls:
            continue

        try:
            os.add_dll_directory(str(directory))
        except (AttributeError, OSError):
            pass

        # Also prepend to PATH — some loaders still rely on it
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        added.append(str(directory))
        if verbose:
            print(f"  + CUDA DLL dir: {directory}")

    if verbose and not added:
        print("  (no CUDA DLL directories found)")

    return added


if __name__ == "__main__":
    setup_cuda_dlls()
    import tensorflow as tf  # noqa: E402

    print(f"\nTensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs found: {len(gpus)}")
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"  - {details.get('device_name', gpu.name)}")
    if not gpus:
        print("\nNo GPU detected — check the CUDA/cuDNN versions above.")
