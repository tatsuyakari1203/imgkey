# ImgKey Release Guide

This document describes how ImgKey releases are produced, verified, and published.

## Current release profile

- Target platform: Windows x64
- Release artifact: `ImgKey-<version>-windows-x64.exe`
- Optional local artifact: `ImgKey-GPU.exe` legacy/dev CUDA compatibility build only
- Build type: one-file, windowed PyInstaller executable
- Runtime profile: classical CPU path plus bundled native D3D12 backend with automatic CPU fallback

## What is included

- Large-image chroma-key workflow for green, blue, auto-detected, and picked custom screen colors.
- Viewer-first PySide6 UI with pan/zoom, fit/100%, eyedropper, debug views, imported mattes, and background previews.
- High Accuracy geometric defaults tuned from the `green_cyan_safe` benchmark profile for blue, green, cyan, and uneven screen assets.
- Global connected-background matte decisions before tiled full-resolution export.
- Edge-only trimap refinement and tile-safe PNG export.
- Edge Color Reconstruction Pro: alpha/spill-gated fringe mask, alpha-aware foreground unmix, Vlahos-style key-channel clamp, nearest-inner foreground color pull, luminance protection, and zeroed transparent RGB.
- Backend-neutral GPU registry with D3D12 as the primary native Windows backend, deferred Vulkan runtime probe telemetry, legacy CUDA compatibility for local comparisons, and CPU fallback.

## Release workflow

Releases are built by GitHub Actions from `.github/workflows/release.yml`.

The workflow runs on `windows-latest` and performs:

1. Check out the repository.
2. Set up Python 3.10.
3. Install dependencies from `requirements.txt` plus PyInstaller.
4. Build `native/imgkey_gpu/build/imgkey_gpu.dll` with `native/imgkey_gpu/build.ps1 -Clean`.
5. Run verification:
   - `python smoke_test.py`
   - `python -m gpu_runtime --probe --json --no-kernel-smoke`
   - PowerShell-expanded `python -m py_compile` over `app.py`, `keyer.py`, GPU/runtime modules, `imgkey_engine/*.py`, and `ui/*.py`
   - `python -c "import app, keyer; print('import ok')"`
6. Build the executable with `python -m PyInstaller --noconfirm --clean ImgKey.spec`.
7. Probe `dist\ImgKey.exe --gpu-probe --json` with a sanitized PATH and verify CPU fallback plus any available D3D12 status.
8. Rename the release asset to `ImgKey-<version>-windows-x64.exe`.
9. Upload the EXE as a workflow artifact.
10. Create or update the GitHub Release and attach the EXE.

## How to publish a release

Use a semantic version tag prefixed with `v`:

```powershell
git status --short --branch
git tag v1.1.0
git push origin v1.1.0
```

After the tag is pushed, monitor the workflow:

```powershell
gh run list --repo tatsuyakari1203/imgkey --workflow Release --limit 5
gh run watch --repo tatsuyakari1203/imgkey <run-id>
```

## Manual release dispatch

The same workflow can be run manually from GitHub:

1. Open **Actions**.
2. Select **Release**.
3. Click **Run workflow**.
4. Enter a version such as `v1.1.0`.
5. Optionally mark it as a prerelease.

Manual dispatch is useful for rebuilding an existing release asset, but normal releases should be tag-driven.

## Local pre-release checklist

Run these commands before creating a release tag:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File native/imgkey_gpu/build.ps1 -Clean
python smoke_test.py
python smoke_test.py --write-geometric-benchmark
python smoke_test.py --tune-geometric-defaults
python smoke_test.py --gpu-parity
python smoke_test.py --gpu-benchmark
python smoke_test.py --write-perf-baseline
python -m gpu_runtime --probe --json
$files = @("app.py", "keyer.py", "smoke_test.py", "gpu_runtime.py", "screen_analysis.py", "gpu_accel.py", "gpu_backend.py", "native_toolchain.py", "packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py") + (Get-ChildItem -Path "imgkey_engine", "ui" -Filter "*.py").FullName
python -m py_compile @files
python -c "import app, keyer; print('import ok')"
python -m PyInstaller --noconfirm --clean ImgKey.spec
.\dist\ImgKey.exe --gpu-probe --json
```

Optional diagnostics:

```powershell
python smoke_test.py --write-diagnostics
python smoke_test.py --write-edge-repair-diagnostics
```

Diagnostics are generated under `.artifact/` and are intentionally not committed.

## Dependency policy

- Allowed default dependencies: `numpy`, `opencv-python`, `Pillow`, `PySide6`, and Python standard library modules.
- Keep PyTorch/CUDA, CuPy, ONNX Runtime, PyOpenCL, model runtimes, Vulkan SDK files, and shader compilers out of the default dependency file and primary PyInstaller spec.
- The primary `ImgKey.exe` may include only the compact native `imgkey_gpu.dll` plus imported MSVC runtime DLLs if the native DLL needs them.
- Keep `ImgKey-GPU.exe` as a legacy/dev CUDA compatibility artifact only; do not publish it as the primary public release asset.

## Artifact policy

Committed source of truth:

- `app.py`
- `keyer.py`
- `gpu_runtime.py`
- `screen_analysis.py`
- `smoke_test.py`
- `requirements.txt`
- `requirements-gpu-runtime-cu128.txt`
- `ImgKey.spec`
- `ImgKey-GPU.spec`
- `docs/build-gpu.md`
- `README.md`
- `RELEASE.md`
- `CHANGELOG.md`
- `AGENTS.md`
- `.github/workflows/release.yml`
- `.plan/*.md`

Ignored/generated outputs:

- `.artifact/`
- `build/`
- `dist/`
- Python caches
- downloaded package caches

## Rollback

If a release workflow fails before publishing, fix the issue on `main`, push the fix, then move or recreate the tag only if the failed tag has not produced a trusted public release.

If a bad public release is already published, mark it as superseded, publish a fixed patch version, and document the issue in the release notes.
