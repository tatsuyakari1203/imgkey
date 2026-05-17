# Changelog

All notable changes to ImgKey are documented here.

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
