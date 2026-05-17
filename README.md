# ImgKey

ImgKey is a Windows/Python desktop app for removing green, blue, or custom-color screens from large still images and exporting straight-alpha PNGs. The current v4 workflow is a classical NumPy/OpenCV/PySide6 path with edge color reconstruction: no AI runtime, no PyTorch/CUDA, and no bundled model weights.

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
3. Inspect the matte and repair maps with **Result**, **Source**, **Alpha**, **AI Hint**, **Background Mask**, **Edge Mask**, **Fringe Mask**, **Despill Mask**, **Foreground RGB**, and **Split Compare** views.
4. Use **Fit** for whole-image work, **100%** for crop-quality inspection, and pan/zoom in the central canvas before export.
5. Tune grouped controls in the inspector:
   - **Key**: key mode, sample size, key chip.
   - **Matte**: clip background/foreground, matte gamma, core strength, despeckle, connected vs aggressive interior removal.
   - **Edge**: refine radius, edge softness, erode/expand, edge restore.
   - **Color / Spill Cleanup**: despill, decontaminate, luminance restore/protect, fringe remove, edge color repair, inner color pull, and fringe band.
   - **Output**: proxy/full-crop preview and full-resolution PNG export.
6. Optionally import grayscale masks with **Import Keep**, **Import Remove**, or **Import AI Hint**. Use **Export Matte** to save the current matte as a reusable grayscale PNG.
7. Click **Export PNG** to process the full-resolution image in a worker with progress/cancel support.

## Large-image behavior

- Key/color sampling, connected-background decisions, trimap construction, manual masks, and AI alpha hints are resolved globally before tiled export.
- Full-resolution color unmix/despill/edge repair runs in overlap tiles and writes only each tile core, preventing tile seams while avoiding a full-image float32 RGB working copy.
- Source pixels remain `uint8`; masks are `uint8`; nearest-inner repair labels are `int32` only when under the memory cap; float work is limited to tiles/edge bands.
- Fringe masks and nearest-inner foreground references are computed from the global matte before export so tile repair decisions stay deterministic. If the label map would exceed the memory cap, export falls back to unmix/channel-clamp repair without allocating a full-resolution foreground RGB image.
- Default settings target still images around 8K+ on normal RAM. If memory is tight, reduce preview scale or tile size before export.

## Classical keying notes

- The default **Connected Background** policy removes only screen-colored regions connected to the image border, preserving foreground islands that happen to match the key color.
- **Aggressive Interior Removal** can remove disconnected high-confidence key-colored islands, but should be used deliberately.
- Edge-only alpha refinement keeps hard background at alpha 0 and foreground core at alpha 255 while preserving soft anti-aliased edges.
- v4 **Edge Color Reconstruction** repairs contaminated soft-edge RGB after alpha generation. It builds an alpha/spill-gated **Fringe Mask**, unmixes screen color, applies a Vlahos-style channel clamp, optionally pulls color from the nearest clean opaque foreground pixel, and preserves perceived luminance via **Luminance Restore**.
- **Fringe Remove** controls channel/spill removal strength, **Edge Color Repair** blends reconstructed RGB into the edge, **Inner Color Pull** controls nearest-foreground color pull, and **Fringe Band** widens the soft-edge repair area. Repair is color-only; it does not change alpha.
- Exported PNGs are straight-alpha; fully transparent RGB is zeroed.

## AI Alpha Hint and optional AI seams

ImgKey currently **does not bundle, download, or install** PyTorch/CUDA, `onnxruntime-gpu`, BiRefNet weights, CorridorKey, or noncommercial model/runtime assets. `requirements.txt` contains only the default non-AI runtime dependencies.

- **Import AI Hint** accepts PNG/TIFF/JPG/BMP grayscale masks as coarse alpha hints from any external tool. Bright pixels protect foreground/core and can raise alpha where the classical connected-background model does not mark background. Dark hint pixels do **not** automatically remove background; use **Import Remove** for forced removal.
- **BiRefNet Assist** in `ai_assist.py` is an external-adapter seam only. It checks optional packages, `IMGKEY_BIREFNET_MODEL`, and `IMGKEY_BIREFNET_ADAPTER=module:function`; it does not import heavy runtimes at startup and does not download a model.
- **CorridorKey Plugin** is also external/plugin-style. Contract: input `rgb_u8` HxWx3 + `alpha_hint_u8` HxW; output optional foreground RGB, alpha grayscale, and processed RGBA. Stop for a license/distribution decision before bundling or redistributing CorridorKey, runtimes, or weights.

## Verification

```powershell
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py
python -c "import app, keyer; print('import ok')"
```

Optional diagnostics are written under `.artifact/` only:

```powershell
python smoke_test.py --write-diagnostics
python smoke_test.py --write-edge-repair-diagnostics
```

Edge repair diagnostics include before/after PNGs and black/white/gray/checkerboard composites under `.artifact/edge-repair-verification/`.

## Build default non-AI EXE

`ImgKey.spec` is the packaging source of truth. It builds `dist\ImgKey.exe` as a one-file, windowed app from `app.py` and explicitly excludes optional AI/model runtimes from the default bundle.

```powershell
python -m PyInstaller --noconfirm --clean ImgKey.spec
$p = Start-Process -FilePath ".\dist\ImgKey.exe" -PassThru
Start-Sleep -Seconds 6
if ($p.HasExited) { exit 1 } else { Stop-Process -Id $p.Id }
```
