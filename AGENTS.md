# Repo Working Context

## What this repo is
- ImgKey is a Windows/Python desktop chroma-key app for large still images, built with PySide6, NumPy, OpenCV, and Pillow.
- The product/runtime surface is classical-only. Default `ImgKey.exe` is CPU/lightweight; `ImgKey-GPU.exe` is an optional CUDA tensor-runtime probe/acceleration flavor.
- Current source of truth is the root source files, `ImgKey.spec`, `ImgKey-GPU.spec`, `docs/build-gpu.md`, and the active v9 geometric defaults plan. `build/`, `dist/`, caches, and `.artifact/` outputs are generated/disposable.

## Core architecture
- `app.py` — PySide6 viewer-first UI, `ImageCanvas`, inspector controls, preview/export threads, eyedropper, manual keep/remove masks, imported matte support, GPU status probe, and full-resolution PNG export wiring.
- `keyer.py` — image I/O helpers, preview resize, compatible `KeySettings`/`KeyResult`, global matte/trimap logic, linear-light fringe/edge color repair, guided alpha refinement, tile-local fallbacks, crop render support, tile export, despill, and checkerboard compositing.
- `gpu_runtime.py` — lazy torch-only-inside-probe CUDA diagnostics and `python -m gpu_runtime --probe --json`.
- `screen_analysis.py` — deterministic classical screen maps/plates used by tests and future cleanup work.
- `smoke_test.py` — synthetic smoke tests for green/blue/custom keying, connected-background preservation, edge alpha, linear-light repair, guided alpha, tile-local screen/nearest-inner fallbacks, crop render parity, despill, tile consistency, removed runtime-surface guards, and import fences.
- `ImgKey.spec` — default onefile/windowed EXE packaging source.
- `ImgKey-GPU.spec` — optional PyTorch CUDA tensor runtime/probe EXE.

## Build and verification
- `pip install -r requirements.txt` — install default runtime dependencies.
- `python app.py` — run the desktop app locally.
- `python smoke_test.py` — required smoke test.
- `python smoke_test.py --write-diagnostics` — optional, writes synthetic fixture outputs under `.artifact/smoke-fixtures/`.
- `python smoke_test.py --write-edge-repair-diagnostics` — optional, writes before/after edge repair composites and metrics under `.artifact/edge-repair-verification/`.
- `python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py` — syntax/import-surface compile check.
- `python -c "import app, keyer; print('import ok')"` — import check when PySide6 is installed.
- `python -m PyInstaller --noconfirm --clean ImgKey.spec` — default onefile/windowed EXE build that produces `dist\ImgKey.exe`.
- `python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec` — optional CUDA tensor-runtime EXE build after installing `requirements-gpu-runtime-cu128.txt`.

## Change boundaries / risky areas
- Do not rewrite the algorithm or UI while doing repo-context/baseline-safety work.
- Large-image keying must avoid full-image float32 RGB allocations; keep source as `uint8`, masks as `uint8`, nearest-inner labels as bounded `int32`, and use float work per tile/ROI only.
- Global screen sampling, connected-background decisions, trimaps, manual masks, fringe masks, and capped nearest-inner repair labels must happen before tiled export to avoid seams.
- UI defaults are the **High Accuracy** `green_cyan_safe` geometric benchmark profile: key color `(30, 80, 235)`, sample size `10`, tolerance `0.26`, softness `0.02`, clip background `0.95`, clip foreground `0.08`, matte gamma `1.60`, core strength `0.45`, edge radius `24`, erode/expand `-4`, despill `0.80`, decontaminate `0.70`, luminance restore/protect `0.85`, fringe remove `0.85`, edge color repair `0.80`, inner color pull `0.60`, fringe band radius `5`, transition alpha recover `0.90`, key-vector despill `0.85`, and foreground color pull `0.75`.
- Linear-light repair rule: edge repair is RGB-only and alpha/edge/fringe-gated; unmix, Vlahos-style channel clamp, nearest-inner color pull, and luminance protection operate in linear light with linear key/screen vectors, then convert back to sRGB. Do not turn it into a global color-grade pass, do not change alpha from the repair path, and always zero RGB where alpha is 0.
- Guided-filter rule: `guided_alpha_refine` defaults to `0.0` (off). When enabled, guided alpha refinement must use grayscale/linear-luma edge-band ROI work only, clamp exact known background/foreground/core regions afterward, obey `guided_max_pixels`, and deterministically skip/fall back to unchanged alpha when the cap would be exceeded.
- Tile-local screen rule: full-image `uint8` screen maps are allowed only under `max_local_screen_model_pixels`; otherwise tiled render must estimate screen color inside each read tile from connected/background-safe pixels. Read overlap must include the screen-estimation radius and write only tile cores.
- Tile-local nearest-inner rule: keep the global `int32` label map below cap; when labels are skipped and `inner_color_pull > 0`, build tile-local labels only inside the overlapped read tile, require enough nearby clean inner pixels, bound useful radius by overlap/margins, and fall back to unmix/clamp when local labels are absent/too far.
- Crop-render contract: when `settings.full_res_crop`/render crop is active, global matte decisions still use the full image, color rendering is crop+overlap only, and debug/result arrays must be crop-shaped and mutually aligned while UI metadata can still report original source coordinates/size.
- The default dependency fence remains `numpy`, `opencv-python`, `Pillow`, `PySide6`, and stdlib only. Do not add PyTorch/CUDA or other heavy optional packages to `requirements.txt` or `ImgKey.spec`.
- User approved `Aggressive Interior Removal` as the default/reset target; do not revert it to connected-background default without explicit direction.
- Space is a spring-loaded pan override: holding Space temporarily pans even while Pick is active, and releasing Space restores the prior tool without toggling toolbar state.

## Common workflows
- Keep generated diagnostics, screenshots, temporary exports, and backup snapshots under `.artifact/` only.
- Keep package caches and downloaded binaries out of the source tree or in ignored cache folders.
- Run `python smoke_test.py`, the required `py_compile`, `python -c "import app, keyer; print('import ok')"`, and the dependency fence before marking a phase complete. Use `docs/build-gpu.md` for GPU packaging commands and clean-target expectations.

## Notes for agents
- This folder is a git repo. Inspect status/diff before commits and never stage `build/`, `dist/`, `.artifact/`, caches, or generated exports.
- Treat `build/`, `dist/`, generated PNG exports, and package caches as disposable artifacts.
