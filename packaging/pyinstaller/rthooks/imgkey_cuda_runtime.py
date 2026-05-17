from __future__ import annotations

import os
from pathlib import Path
import sys


_DLL_DIRECTORY_HANDLES = []


def _runtime_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))


def _add_dll_directory(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    text = str(path)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(text))
        except OSError:
            pass
    current = os.environ.get("PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if text not in parts:
        os.environ["PATH"] = text + (os.pathsep + current if current else "")


root = _runtime_root()

for relative in (
    Path("torch") / "lib",
    Path("torchvision"),
):
    _add_dll_directory(root / relative)

nvidia_root = root / "nvidia"
if nvidia_root.exists():
    for bin_dir in nvidia_root.glob("*/bin"):
        _add_dll_directory(bin_dir)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

bundled_model = root / "models" / "BiRefNet"
if bundled_model.exists() and not os.environ.get("IMGKEY_BIREFNET_MODEL"):
    os.environ["IMGKEY_BIREFNET_MODEL"] = str(bundled_model)
