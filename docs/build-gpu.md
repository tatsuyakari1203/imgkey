# ImgKey GPU build notes

ImgKey has three Windows build flavors. Keep them separated so the default app remains lightweight and so BiRefNet remains the only AI model path.

## 1. Classical `ImgKey.exe`

No torch, no CUDA runtime, no model weights.

```powershell
python -m venv .venv-classical
.\.venv-classical\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python smoke_test.py
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

`ImgKey.spec` is the default release source of truth and explicitly excludes AI/GPU packages.

## 2. GPU runtime `ImgKey-GPU.exe`

Includes PyTorch CUDA for probing the GPU, but no Transformers/BiRefNet model stack and no model weights.

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

## 3. GPU BiRefNet `ImgKey-GPU-BiRefNet.exe`

Includes PyTorch CUDA plus the BiRefNet-only worker/adapter path. It never downloads model weights at runtime.

```powershell
python -m venv .venv-gpu-birefnet
.\.venv-gpu-birefnet\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python -m pip install -r requirements-gpu-runtime-cu128.txt
python -m pip install -r requirements-gpu-birefnet-cu128.txt
$env:IMGKEY_BIREFNET_MODEL="D:\models\BiRefNet-e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
python -m PyInstaller --noconfirm --clean ImgKey-GPU-BiRefNet.spec
```

To bundle a model snapshot into the EXE, first update `ai_backends/birefnet_manifest.json` with exact SHA256 hashes and reviewed license/notice metadata for the selected local snapshot. Then build with:

```powershell
$env:IMGKEY_BIREFNET_BUNDLE_MODEL="D:\models\BiRefNet-e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
$env:IMGKEY_BIREFNET_BUNDLE_LICENSE_OK="1"
python -m PyInstaller --noconfirm --clean ImgKey-GPU-BiRefNet.spec
```

The spec intentionally refuses bundled model builds when the license acknowledgment is missing or required manifest hashes are not filled. For non-bundled builds, set `IMGKEY_BIREFNET_MODEL` on the target machine to an existing local BiRefNet snapshot.

Optional local worker smoke with a real model and source image:

```powershell
New-Item -ItemType Directory -Path ".artifact" -Force | Out-Null
$sourceImage = "D:\test-images\green-screen-source.png"  # replace with an existing PNG/JPG/TIFF/BMP
@{
  backend = "birefnet"
  input_image_path = $sourceImage
  model_path = $env:IMGKEY_BIREFNET_MODEL
  device = "cuda"
  mode = "global_plus_roi"
  max_side = 1536
  precision = "fp16"
  output_dir = ".artifact\ai-worker"
  temp_dir = ".artifact\ai-worker\temp"
} | ConvertTo-Json | Set-Content -LiteralPath ".artifact\birefnet-request.json" -Encoding UTF8
python ai_worker.py --request .artifact\birefnet-request.json --json
```

## RTX 5060 Ti / Blackwell constraints

- Use PyTorch CUDA 12.8 or newer for RTX 50-series / Blackwell. Do not use old `cu121` or `cu124` wheels.
- Install from the official PyTorch CUDA wheel index, currently represented by `requirements-gpu-runtime-cu128.txt`.
- The target machine needs an NVIDIA driver whose `nvidia-smi` reports CUDA 12.8 or newer. A local CUDA Toolkit install is not required for the packaged EXE.
- Validate with `python -m gpu_runtime --probe --json` before packaging and with `ImgKey-GPU.exe --gpu-probe --json` after packaging.

## Model and offline behavior

- Only `ZhengPeng7/BiRefNet` at the pinned manifest revision is in scope.
- Runtime accepts only local directories; URLs and Hugging Face repo IDs are rejected.
- The worker sets/obeys `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `local_files_only=True`.
- No Matting Anything, SAM, U2Net, MODNet, ViTMatte, CorridorKey, or other model packages/weights may be bundled in v6.

## Clean-target testing expectations

Test generated EXEs on a clean Windows x64 target with an NVIDIA driver only: no Python, no pip packages, and no CUDA Toolkit on PATH.

1. `ImgKey.exe` opens and passes a manual import/export smoke path without torch/model files.
2. `ImgKey-GPU.exe --gpu-probe --json` reports CUDA availability or a clear driver/runtime error; it must not require a model.
3. `ImgKey-GPU-BiRefNet.exe` starts without hidden downloads; missing local/bundled model reports a clear model-not-ready state.
4. With a validated local or bundled BiRefNet snapshot, `Generate BiRefNet Hint` writes worker outputs under `.artifact/`/temp only and Hybrid BiRefNet preview/export uses the generated hint.
5. Confirm `build/`, `dist/`, wheels, model snapshots, caches, and `.artifact/` outputs remain ignored and are not committed.
