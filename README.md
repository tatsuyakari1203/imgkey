# ImgKey

ImgKey is a Windows/Python desktop app for removing green, blue, or custom-color screens from large still images and exporting straight-alpha PNGs. The default build is a classical NumPy/OpenCV/PySide6 path with linear-light edge color reconstruction, crop-only full-resolution preview rendering, and tile-local large-image fallbacks.

Public Windows builds are limited to `ImgKey.exe` for the default CPU path and `ImgKey-GPU.exe` for the optional CUDA tensor-runtime path.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Workflow

1. Click **Open image** or drag an image into the window.
2. Choose **Auto**, **Green**, **Blue**, or **Pick**. In **Pick** mode, click the canvas to sample the screen color accurately.
3. Inspect the matte and repair maps with **Result**, **Source**, **Alpha**, **Imported Matte**, **Background Mask**, **Edge Mask**, **Fringe Mask**, **Despill Mask**, **Foreground RGB**, and **Split Compare** views.
4. Use **Fit** for whole-image work, **100%** for crop-quality inspection, and pan/zoom in the central canvas before export.
5. Tune grouped controls in the inspector:
   - **Screen**: key mode, sample size, key chip.
   - **Matte**: clip background/foreground, matte gamma, core strength, despeckle, connected vs aggressive interior removal.
   - **Edges**: refine radius, edge softness, erode/expand.
   - **Spill Cleanup**: despill, decontaminate, luminance restore/protect, fringe remove, edge color repair, inner color pull, and fringe band.
   - **Masks & Export**: proxy/full-crop preview, imported matte, matte export, and full-resolution PNG export.
6. Optionally import grayscale masks with **Import Keep**, **Import Remove**, or **Import Matte**. Bright imported-matte pixels protect foreground/core and can raise alpha where the classical connected-background decision does not mark background. Dark matte pixels do not force removal; use **Import Remove** for forced removal.
7. Click **Export PNG** to process the full-resolution image in a worker with progress/cancel support.

## Large-image behavior

- Key/color sampling, connected-background decisions, trimap construction, manual masks, and imported mattes are resolved globally before tiled export.
- Full-resolution color unmix/despill/edge repair runs in overlap tiles and writes only each tile core, preventing tile seams while avoiding a full-image float32 RGB working copy.
- Source pixels remain `uint8`; masks are `uint8`; nearest-inner repair labels are bounded `int32` only when under the memory cap; float work is limited to tiles, crop regions, and edge-band ROIs.
- The local screen estimate builds a full-image `uint8` screen map only below its cap. Large tiled renders fall back to tile-local screen estimates from connected/background-safe pixels inside each read tile.
- Nearest-inner foreground references use the global label map when it is within the cap. If the label map is skipped, tile-local nearest-inner labels are built inside the overlapped read tile and fall back to unmix/channel-clamp repair when too few/too-far inner pixels are available.
- **Full Crop** preview renders only the requested full-resolution crop plus required read overlap; result/debug arrays are crop-shaped and aligned while source-size UI metadata is preserved.

## Classical keying notes

- The default **Aggressive Interior Removal** target removes disconnected high-confidence key-colored islands. Switch to **Connected Background** when preserving same-key foreground islands matters more.
- Edge-only alpha refinement keeps hard background at alpha 0 and foreground core at alpha 255 while preserving soft anti-aliased edges.
- Optional guided alpha refinement settings are available for API/test use: `guided_alpha_refine`, `guided_radius`, `guided_eps`, and `guided_max_pixels`. The default strength is `0.0` (off), and capped edge-band ROI filtering skips deterministically when the memory cap would be exceeded.
- **Edge Color Repair** repairs contaminated soft-edge RGB after alpha generation in linear light. It builds an alpha/spill-gated **Fringe Mask**, unmixes screen color, applies a Vlahos-style channel clamp, optionally pulls color from the nearest clean opaque foreground pixel, and preserves linear-luminance detail via **Luminance Restore**.
- **Fringe Remove** controls channel/spill removal strength, **Edge Color Repair** blends reconstructed RGB into the edge, **Inner Color Pull** controls nearest-foreground color pull, and **Fringe Band** widens the soft-edge repair area. Repair is color-only; it does not change alpha.
- Exported PNGs are straight-alpha; fully transparent RGB is zeroed.

## Verification

```powershell
python smoke_test.py
python smoke_test.py --write-edge-repair-diagnostics
python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py
python -c "import app, keyer; print('import ok')"
```

Optional diagnostics are written under `.artifact/` only:

```powershell
python smoke_test.py --write-diagnostics
```

Edge repair diagnostics include before/after PNGs and black/white/gray/checkerboard composites under `.artifact/edge-repair-verification/`.

## Build default EXE

`ImgKey.spec` is the packaging source of truth. It builds `dist\ImgKey.exe` as a one-file, windowed app from `app.py` and keeps the default bundle lightweight.

```powershell
python -m PyInstaller --noconfirm --clean ImgKey.spec
$p = Start-Process -FilePath ".\dist\ImgKey.exe" -PassThru
Start-Sleep -Seconds 6
if ($p.HasExited) { exit 1 } else { Stop-Process -Id $p.Id }
```

## Build optional GPU runtime EXE

`ImgKey-GPU.spec` builds `dist\ImgKey-GPU.exe` with PyTorch CUDA tensor-runtime/probe support and visible onefile startup splash/progress. See `docs/build-gpu.md` for exact clean-environment install/build commands and RTX 50-series / CUDA 12.8 constraints.

## Release workflow

The repository includes a GitHub Actions release workflow at `.github/workflows/release.yml`. See `RELEASE.md` for the full release process and `CHANGELOG.md` for release notes.

Create a public release by pushing an approved version tag:

```powershell
git tag v1.1.0
git push origin v1.1.0
```

The workflow runs on `windows-latest`, installs the default dependencies, runs the smoke/import checks, builds with `ImgKey.spec`, and uploads `ImgKey-<version>-windows-x64.exe` to the GitHub Release.
