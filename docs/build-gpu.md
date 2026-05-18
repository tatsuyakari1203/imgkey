# ImgKey GPU build notes

ImgKey has two Windows build flavors. Keep them separated so the default app remains lightweight and the optional GPU executable can carry CUDA tensor-runtime files.

## 1. Default `ImgKey.exe`

No torch and no CUDA runtime.

```powershell
python -m venv .venv-classical
.\.venv-classical\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python smoke_test.py
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

`ImgKey.spec` is the default release source of truth and keeps optional GPU packages out of the lightweight bundle.

## 2. GPU runtime `ImgKey-GPU.exe`

Includes PyTorch CUDA for tensor-runtime/probe support. The GPU spec uses PyInstaller's boot splash (`packaging/imgkey_splash.png`) so onefile extraction shows visible progress before the Qt UI can start.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python -m pip install -r requirements-gpu-runtime-cu128.txt
python -m gpu_runtime --probe --json
python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec
.\dist\ImgKey-GPU.exe --gpu-probe --json
```

## RTX 5060 Ti / Blackwell constraints

- Use PyTorch CUDA 12.8 or newer for RTX 50-series / Blackwell. Do not use old `cu121` or `cu124` wheels.
- Install from the official PyTorch CUDA wheel index, currently represented by `requirements-gpu-runtime-cu128.txt`.
- The target machine needs an NVIDIA driver whose `nvidia-smi` reports CUDA 12.8 or newer. A local CUDA Toolkit install is not required for the packaged EXE.
- Validate with `python -m gpu_runtime --probe --json` before packaging and with `ImgKey-GPU.exe --gpu-probe --json` after packaging.

## Clean-target testing expectations

Test generated EXEs on a clean Windows x64 target with an NVIDIA driver only: no Python, no pip packages, and no CUDA Toolkit on PATH.

1. `ImgKey.exe` opens and passes a manual import/export smoke path without torch files.
2. `ImgKey-GPU.exe --gpu-probe --json` reports CUDA availability or a clear driver/runtime error.
3. Confirm `build/`, `dist/`, wheels, caches, and `.artifact/` outputs remain ignored and are not committed.
