# ImgKey Release Guide

This document describes how ImgKey releases are produced, verified, and published.

## Current release

- Version: `v1.0.0`
- Target platform: Windows x64
- Release artifact: `ImgKey-v1.0.0-windows-x64.exe`
- Build type: one-file, windowed PyInstaller executable
- Runtime profile: default non-AI build; no PyTorch/CUDA/model weights are bundled

## What is included

The `v1.0.0` release includes:

- Large-image chroma-key workflow for green, blue, auto-detected, and picked custom screen colors.
- Viewer-first PySide6 UI with pan/zoom, fit/100%, eyedropper, debug views, and background previews.
- High Accuracy Graphic defaults tuned for blue-screen graphic/poster images.
- Global connected-background matte decisions before tiled full-resolution export.
- Edge-only trimap refinement and tile-safe PNG export.
- v4 Edge Color Reconstruction Pro:
  - alpha/spill-gated fringe mask,
  - alpha-aware foreground unmix,
  - Vlahos-style key-channel clamp,
  - nearest-inner foreground color pull,
  - luminance protection,
  - zeroed transparent RGB.
- Optional external AI seams through `ai_assist.py`; no AI dependencies are installed or bundled by default.

## Release workflow

Releases are built by GitHub Actions from `.github/workflows/release.yml`.

The workflow runs on `windows-latest` and performs:

1. Check out the repository.
2. Set up Python 3.10.
3. Install dependencies from `requirements.txt` plus PyInstaller.
4. Run verification:
   - `python smoke_test.py`
   - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
   - `python -c "import app, keyer; print('import ok')"`
5. Build the executable with:
   - `python -m PyInstaller --noconfirm --clean ImgKey.spec`
6. Rename the release asset to:
   - `ImgKey-<version>-windows-x64.exe`
7. Upload the EXE as a workflow artifact.
8. Create or update the GitHub Release and attach the EXE.

## How to publish a release

Use a semantic version tag prefixed with `v`:

```powershell
git status --short --branch
git tag v1.0.0
git push origin v1.0.0
```

After the tag is pushed, monitor the workflow:

```powershell
gh run list --repo tatsuyakari1203/imgkey --workflow Release --limit 5
gh run watch --repo tatsuyakari1203/imgkey <run-id>
```

When the workflow completes, the public release is available at:

```text
https://github.com/tatsuyakari1203/imgkey/releases/tag/v1.0.0
```

## Manual release dispatch

The same workflow can be run manually from GitHub:

1. Open **Actions**.
2. Select **Release**.
3. Click **Run workflow**.
4. Enter a version such as `v1.0.0`.
5. Optionally mark it as a prerelease.

Manual dispatch is useful for rebuilding an existing release asset, but normal releases should be tag-driven.

## Local pre-release checklist

Run these commands before creating a release tag:

```powershell
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py
python -c "import app, keyer; print('import ok')"
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

Optional diagnostics:

```powershell
python smoke_test.py --write-diagnostics
python smoke_test.py --write-edge-repair-diagnostics
```

Diagnostics are generated under `.artifact/` and are intentionally not committed.

## Dependency and licensing policy

The default release must stay lightweight and non-AI:

- Allowed default dependencies: `numpy`, `opencv-python`, `Pillow`, `PySide6`, and Python standard library modules.
- Do not bundle PyTorch, CUDA, ONNX Runtime GPU, PyMatting, CorridorKey, BiRefNet weights, or other model assets without an explicit distribution and licensing decision.
- Optional AI integrations must remain external/plugin-style and must not import heavy runtimes at app startup.

## Artifact policy

Committed source of truth:

- `app.py`
- `keyer.py`
- `ai_assist.py`
- `smoke_test.py`
- `requirements.txt`
- `ImgKey.spec`
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
- optional model caches/weights

## Rollback

If a release workflow fails before publishing, fix the issue on `main`, push the fix, then move or recreate the tag only if the failed tag has not produced a trusted public release.

If a bad public release is already published:

1. Mark the release as a prerelease or delete the bad asset.
2. Publish a patch release, for example `v1.0.1`.
3. Document the fix in `CHANGELOG.md`.
