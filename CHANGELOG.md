# Changelog

All notable changes to ImgKey are documented here.

## Unreleased

### Added

- Separate GPU packaging scaffolding: `ImgKey-GPU.spec` for CUDA runtime/probe builds and `ImgKey-GPU-BiRefNet.spec` for the BiRefNet-only worker/adapter build path.
- CUDA 12.8 GPU requirement files and GPU build notes covering RTX 50-series constraints, local/offline BiRefNet model paths, manifest/license gates, and clean-target testing.
- PyInstaller and Qt startup splashes for GPU builds so large onefile extraction/startup no longer appears frozen.

### Changed

- BiRefNet `global_plus_roi` now performs bounded high-resolution ROI passes and conservative alpha post-processing to preserve foreground/detail with less erosion.
- Hybrid BiRefNet alpha merge is less aggressive about clamping weak BiRefNet detail to background while preserving confident screen background clamps.

### Notes

- Default `ImgKey.spec` remains the classical non-AI build: no torch, no CUDA runtime, no hidden model downloads, and no bundled weights.
- v6 GPU AI scope is BiRefNet-only; no other AI model packages or weights are introduced.

## v1.1.0 - 2026-05-17

### Added

- Linear-light edge color reconstruction for unmixing, Vlahos-style channel clamp, nearest-inner color pull, and luminance protection while keeping repair RGB-only and alpha-gated.
- Optional guided alpha refinement settings (`guided_alpha_refine`, `guided_radius`, `guided_eps`, `guided_max_pixels`) with the default state off and a deterministic cap/fallback path for large edge-band ROIs.
- Tile-local screen model fallback for large tiled renders when the full-image screen map is skipped by the memory cap.
- Crop-only full-resolution preview rendering that renders the selected crop plus required overlap instead of rendering full-image RGBA and cropping afterward.
- Tile-local nearest-inner fallback for large-image edge repair when global nearest-inner labels are skipped by the memory cap.

### Changed

- Large-image tiled render overlap now accounts for active local algorithms: edge/fringe repair, guided radius, tile-local screen estimation, and tile-local nearest-inner pull.
- Crop preview results now keep `KeyResult` image/debug arrays crop-shaped and aligned while preserving source-coordinate metadata for the UI.
- Smoke coverage now includes v5 linear-light repair, guided alpha, tile-local screen, crop render parity, tile-local nearest-inner fallback, seam metrics, transparent RGB zeroing, and dependency-fence checks.

### Notes

- Default release remains classical/non-AI and does not add dependencies beyond NumPy, OpenCV, Pillow, PySide6, and the Python standard library.
- No PyTorch, CUDA, ONNX Runtime, PyMatting, SciPy, numba, CorridorKey runtime, model weights, or noncommercial AI assets are bundled or imported by default.

## v1.0.0 - 2026-05-17

Initial public release.

### Added

- Windows/Python desktop app for removing green, blue, auto-detected, or picked custom-color screens from large still images.
- PySide6 flat viewer-first UI with:
  - large canvas,
  - fit/100% zoom,
  - pan and spring-loaded Space-to-pan,
  - direct eyedropper sampling,
  - grouped inspector controls,
  - Result, Source, Alpha, Background Mask, Edge Mask, Fringe Mask, Despill Mask, Foreground RGB, AI Hint, and Split Compare debug views.
- High Accuracy Graphic defaults tuned for blue-screen graphic/poster keying.
- Large-image keyer built with NumPy/OpenCV/Pillow:
  - global key-color sampling,
  - connected-background matte decisions,
  - foreground island preservation,
  - aggressive interior removal option,
  - edge trimap refinement,
  - tiled full-resolution PNG export with progress/cancel.
- v4 Edge Color Reconstruction Pro:
  - fringe/spill mask,
  - alpha-aware foreground unmix,
  - Vlahos-style key-channel clamp,
  - nearest-inner color pull,
  - luminance protection,
  - tile-safe RGB repair,
  - transparent RGB zeroing.
- Optional external AI seams for BiRefNet/CorridorKey-style workflows without bundling or importing heavy AI runtimes by default.
- Smoke tests and optional diagnostic fixture generation.
- PyInstaller packaging through `ImgKey.spec`.
- GitHub Actions release workflow that builds and publishes `ImgKey-<version>-windows-x64.exe`.

### Verification

- `python smoke_test.py`
- `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
- `python -c "import app, keyer; print('import ok')"`
- `python -m PyInstaller --noconfirm --clean ImgKey.spec`

### Notes

- Default release is non-AI and does not bundle PyTorch, CUDA, model weights, CorridorKey, or noncommercial assets.
- Build and diagnostic outputs are ignored and not part of the source release.
