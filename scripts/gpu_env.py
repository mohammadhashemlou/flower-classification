"""Register CUDA DLL search paths on Windows before TensorFlow is imported.

TensorFlow loads the CUDA and cuDNN DLLs at import time using the normal DLL
search order. If CUDA sits in a non-default location, or cuDNN came from a pip
wheel, those folders are not on PATH and TF quietly falls back to CPU.

Import this before tensorflow:

    from gpu_env import setup_cuda_dlls
    setup_cuda_dlls()

    import tensorflow as tf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_dirs() -> list[Path]:
    """Directories that may hold CUDA/cuDNN DLLs."""
    dirs: list[Path] = []

    if os.name != "nt":
        return dirs

    # System CUDA toolkit, either from CUDA_PATH or the default install root
    cuda_roots = [
        Path(os.environ.get("CUDA_PATH", "")),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
    ]
    for root in cuda_roots:
        if not root or not root.exists():
            continue
        # The root may already be a versioned folder such as ...\CUDA\v12.6
        if (root / "bin").exists():
            dirs.append(root / "bin")
        for child in sorted(root.iterdir(), reverse=True):
            if child.is_dir() and (child / "bin").exists():
                dirs.append(child / "bin")

    # pip-installed NVIDIA wheels. Layouts differ: some put DLLs in bin/,
    # others in lib/x64/.
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
    """Register CUDA DLL directories for this process and return what was added."""
    added: list[str] = []

    for directory in _candidate_dirs():
        if not directory.exists():
            continue
        # Only register folders that actually contain CUDA DLLs
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

        # Also prepend to PATH, some loaders still look there
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        added.append(str(directory))
        if verbose:
            print(f"  CUDA DLL dir: {directory}")

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
        print("\nNo GPU detected, check the CUDA/cuDNN versions above.")
