# Repo Working Context

## What this repo is
- ImgKey is a Windows/Python desktop chroma-key app for large still images, built with PySide6, NumPy, OpenCV, and Pillow.
- Current source of truth is the root source files, `ImgKey.spec`, `.plan/imgkey-v2-large-image-keyer.md`, `.plan/imgkey-v3-ui-ux-redesign.md`, and `.plan/imgkey-v4-edge-color-reconstruction.md`; `build/` and `dist/` are generated outputs, not recoverable source.
- The current implementation is the v4 non-AI large-image keyer with the v3 viewer-first UI/UX redesign, global matte decisions, edge color reconstruction, tiled full-resolution export, and optional external AI seams.

## Core architecture
- `app.py` — PySide6 viewer-first UI, `ImageCanvas`, inspector controls, preview/export threads, eyedropper, masks, and full-resolution PNG export wiring.
- `keyer.py` — image I/O helpers, preview resize, compatible `KeySettings`/`KeyResult`, global matte/trimap logic, v4 fringe/edge color repair, tile export, despill, and checkerboard compositing.
- `ai_assist.py` — optional external AI alpha-hint/plugin seams only; no default AI dependency import/download/bundling.
- `smoke_test.py` — synthetic smoke tests for green/blue/custom keying, connected-background preservation, edge alpha, v4 fringe repair, despill, tile consistency, and optional AI no-dependency behavior.
- `ImgKey.spec` — packaging source of truth for the default non-AI onefile/windowed EXE.

## Build and verification
- `pip install -r requirements.txt` — install default non-AI runtime dependencies.
- `python app.py` — run the desktop app locally.
- `python smoke_test.py` — required smoke test for the current v4 implementation.
- `python smoke_test.py --write-diagnostics` — optional, writes synthetic fixture outputs under `.artifact/smoke-fixtures/`.
- `python smoke_test.py --write-edge-repair-diagnostics` — optional, writes before/after edge repair composites and metrics under `.artifact/edge-repair-verification/`.
- `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py` — syntax/import-surface compile check.
- `python -c "import app, keyer; print('import ok')"` — import check when PySide6 is installed.
- `python -m PyInstaller --noconfirm --clean ImgKey.spec` — default non-AI onefile/windowed EXE build that produces `dist\ImgKey.exe`.

## Change boundaries / risky areas
- Do not rewrite the algorithm or UI while doing repo-context/baseline-safety work.
- Large-image keying must avoid full-image float32 RGB allocations; keep source as `uint8`, masks as `uint8`, nearest-inner labels as bounded `int32`, and use float work per tile/ROI only.
- Global screen sampling, connected-background decisions, trimaps, semantic masks, fringe masks, and nearest-inner repair labels must happen before tiled export to avoid seams.
- v3/v4 UI defaults are the user-approved **High Accuracy Graphic** Blue preset: key color `(30, 80, 235)`, sample size `10`, tolerance `0.45`, softness `0.01`, clip background `0.97`, clip foreground `0.00`, matte gamma `2.20`, core strength `0.38`, edge radius `32`, erode/expand `-8`, despill `0.70`, decontaminate `0.50`, luminance restore/protect `0.76`, fringe remove `0.75`, edge color repair `0.65`, inner color pull `0.45`, and fringe band radius `3`.
- Edge repair is RGB-only and alpha/edge-gated: it builds `fringe_mask`, applies unmix/Vlahos-style channel clamp, optionally pulls nearest clean foreground color, preserves luminance, and zeroes RGB where alpha is 0. Do not turn it into a global color-grade pass or allocate full-resolution `foreground_rgb` during export.
- The default dependency fence remains `numpy`, `opencv-python`, `Pillow`, `PySide6`, and stdlib only. Do not add PyMatting/SciPy/numba/PyTorch/CUDA/model dependencies without explicit approval.
- User approved `Aggressive Interior Removal` as the v3 default/reset target; do not revert it to connected-background default without explicit direction.
- Space is a spring-loaded pan override: holding Space temporarily pans even while Pick is active, and releasing Space restores the prior tool without toggling toolbar state.
- Stop and ask before bundling or redistributing CorridorKey, PyTorch/CUDA stacks, noncommercial weights, or other optional AI runtimes.

## Common workflows
- Keep generated diagnostics, screenshots, temporary exports, and backup snapshots under `.artifact/` only.
- Keep model downloads/caches out of the source tree or in ignored cache folders.
- Run `python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, and `python -c "import app, keyer; print('import ok')"` before marking a phase complete.

## Notes for agents
- This folder may not be a git repo; create a non-destructive timestamped backup under `.artifact/source-backup-*` before deep rewrites if version control is still absent.
- Treat `build/`, `dist/`, generated PNG exports, and model caches as disposable artifacts.
