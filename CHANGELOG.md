# Changelog

All notable changes to ImgKey are documented here.

## Unreleased

### Changed

- Removed the retired assisted-matte product/runtime surface so ImgKey is classical-only before v7 transition cleanup work.
- Kept two build flavors: default `ImgKey.exe` and optional CUDA tensor-runtime `ImgKey-GPU.exe`.
- Locked the optional GPU packaging requirement to the torch CUDA runtime and tightened PyInstaller excludes for optional companion/scientific packages.
- Renamed retained manual matte import UI to **Imported Matte**.
- Updated smoke tests, docs, packaging specs, and release workflow for the classical-only surface.

## v1.1.0 - 2026-05-17

### Added

- Linear-light edge color reconstruction for unmixing, Vlahos-style channel clamp, nearest-inner color pull, and luminance protection while keeping repair RGB-only and alpha-gated.
- Optional guided alpha refinement settings (`guided_alpha_refine`, `guided_radius`, `guided_eps`, and `guided_max_pixels`) with the default state off and a deterministic cap/fallback path for large edge-band ROIs.
- Tile-local screen fallback for large tiled renders when the full-image screen map is skipped by the memory cap.
- Crop-only full-resolution preview rendering that renders the selected crop plus required overlap instead of rendering full-image RGBA and cropping afterward.
- Tile-local nearest-inner fallback for large-image edge repair when global nearest-inner labels are skipped by the memory cap.

### Changed

- Large-image tiled render overlap now accounts for active local algorithms: edge/fringe repair, guided radius, tile-local screen estimation, and tile-local nearest-inner pull.
- Crop preview results now keep `KeyResult` image/debug arrays crop-shaped and aligned while preserving source-coordinate metadata for the UI.
- Smoke coverage includes linear-light repair, guided alpha, tile-local screen, crop render parity, tile-local nearest-inner fallback, seam metrics, transparent RGB zeroing, and dependency-fence checks.

## v1.0.0 - 2026-05-17

Initial public release.

### Added

- Windows/Python desktop app for removing green, blue, auto-detected, or picked custom-color screens from large still images.
- PySide6 flat viewer-first UI with large canvas, fit/100% zoom, pan and spring-loaded Space-to-pan, direct eyedropper sampling, grouped inspector controls, debug views, and split compare.
- High Accuracy Graphic defaults tuned for blue-screen graphic/poster keying.
- Large-image keyer built with NumPy/OpenCV/Pillow: global key-color sampling, connected-background matte decisions, foreground island preservation, aggressive interior removal option, edge trimap refinement, and tiled full-resolution PNG export with progress/cancel.
- Edge Color Reconstruction Pro: fringe/spill mask, alpha-aware foreground unmix, Vlahos-style key-channel clamp, nearest-inner color pull, luminance protection, tile-safe RGB repair, and transparent RGB zeroing.
- Smoke tests and optional diagnostic fixture generation.
- PyInstaller packaging through `ImgKey.spec`.
- GitHub Actions release workflow that builds and publishes `ImgKey-<version>-windows-x64.exe`.
