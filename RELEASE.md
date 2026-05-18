# ImgKey Release Guide

This document describes how ImgKey releases are produced, verified, and published.

## Current release profile

- Target platform: Windows x64
- Release artifact: `ImgKey-<version>-windows-x64.exe`
- Optional local artifact: `ImgKey-GPU.exe`
- Build type: one-file, windowed PyInstaller executable
- Runtime profile: default classical build

## What is included

- Large-image chroma-key workflow for green, blue, auto-detected, and picked custom screen colors.
- Viewer-first PySide6 UI with pan/zoom, fit/100%, eyedropper, debug views, imported mattes, and background previews.
- High Accuracy geometric defaults tuned from the `green_cyan_safe` benchmark profile for blue, green, cyan, and uneven screen assets.
- Global connected-background matte decisions before tiled full-resolution export.
- Edge-only trimap refinement and tile-safe PNG export.
- Edge Color Reconstruction Pro: alpha/spill-gated fringe mask, alpha-aware foreground unmix, Vlahos-style key-channel clamp, nearest-inner foreground color pull, luminance protection, and zeroed transparent RGB.

## Release workflow

Releases are built by GitHub Actions from `.github/workflows/release.yml`.

The workflow runs on `windows-latest` and performs:

1. Check out the repository.
2. Set up Python 3.10.
3. Install dependencies from `requirements.txt` plus PyInstaller.
4. Run verification:
   - `python smoke_test.py`
   - `python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py`
   - `python -c "import app, keyer; print('import ok')"`
5. Build the executable with `python -m PyInstaller --noconfirm --clean ImgKey.spec`.
6. Rename the release asset to `ImgKey-<version>-windows-x64.exe`.
7. Upload the EXE as a workflow artifact.
8. Create or update the GitHub Release and attach the EXE.

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
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py
python -c "import app, keyer; print('import ok')"
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

Optional diagnostics:

```powershell
python smoke_test.py --write-diagnostics
python smoke_test.py --write-edge-repair-diagnostics
```

Diagnostics are generated under `.artifact/` and are intentionally not committed.

## Dependency policy

- Allowed default dependencies: `numpy`, `opencv-python`, `Pillow`, `PySide6`, and Python standard library modules.
- Keep PyTorch/CUDA out of the default dependency file and default PyInstaller spec.
- Publish `ImgKey-GPU.exe` as a separate optional asset when CUDA tensor-runtime support is needed.

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
