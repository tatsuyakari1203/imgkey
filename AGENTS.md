# Repo Working Context

## What this repo is
- ImgKey is a Windows/Python desktop chroma-key app for large still images, built with PySide6, NumPy, OpenCV, and Pillow.
- Current source of truth is the root source files, `ImgKey.spec`, `ImgKey-GPU.spec`, `ImgKey-GPU-BiRefNet.spec`, `.plan/imgkey-v2-large-image-keyer.md`, `.plan/imgkey-v3-ui-ux-redesign.md`, `.plan/imgkey-v4-edge-color-reconstruction.md`, `.plan/imgkey-v5-classical-algorithm-upgrade.md`, and `.plan/imgkey-v6-birefnet-detail-keyer.md`; `build/` and `dist/` are generated outputs, not recoverable source.
- The current implementation is the v6 BiRefNet-only hybrid detail keyer on top of the v5 classical large-image keyer. The default `ImgKey.spec` build remains classical/non-AI; GPU builds are separate flavors.

## Core architecture
- `app.py` — PySide6 viewer-first UI, `ImageCanvas`, inspector controls, preview/export threads, eyedropper, masks, and full-resolution PNG export wiring.
- `keyer.py` — image I/O helpers, preview resize, compatible `KeySettings`/`KeyResult`, global matte/trimap logic, v5 linear-light fringe/edge color repair, guided alpha refinement, tile-local fallbacks, crop render support, tile export, despill, and checkerboard compositing.
- `ai_assist.py` — optional external AI alpha-hint/plugin seams only; no default AI dependency import/download/bundling.
- `gpu_runtime.py` — lazy torch-only-inside-probe CUDA diagnostics and `python -m gpu_runtime --probe --json`.
- `ai_worker.py` and `ai_backends/birefnet_adapter.py` — isolated BiRefNet-only local/offline worker path; no torch/transformers import at default startup.
- `screen_analysis.py` and `hybrid_trimap.py` — classical screen maps and BiRefNet/classical trimap merge helpers.
- `smoke_test.py` — synthetic smoke tests for green/blue/custom keying, connected-background preservation, edge alpha, v5 linear-light repair, guided alpha, tile-local screen/nearest-inner fallbacks, crop render parity, despill, tile consistency, v6 BiRefNet hybrid behavior, diagnostics, and import fences.
- `ImgKey.spec` — packaging source of truth for the default non-AI onefile/windowed EXE.
- `ImgKey-GPU.spec` — optional PyTorch CUDA runtime/probe EXE, no model stack/weights.
- `ImgKey-GPU-BiRefNet.spec` — optional PyTorch CUDA + BiRefNet-only worker/adapter EXE; model bundling is gated by license/manifest hashes.

## Build and verification
- `pip install -r requirements.txt` — install default non-AI runtime dependencies.
- `python app.py` — run the desktop app locally.
- `python smoke_test.py` — required smoke test for the current v5 implementation.
- `python smoke_test.py --write-diagnostics` — optional, writes synthetic fixture outputs under `.artifact/smoke-fixtures/`.
- `python smoke_test.py --write-edge-repair-diagnostics` — optional, writes before/after edge repair composites and metrics under `.artifact/edge-repair-verification/`.
- `python smoke_test.py --write-birefnet-diagnostics` — optional, writes synthetic/mock BiRefNet diagnostics under `.artifact/birefnet-diagnostics/` without requiring model weights.
- `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py gpu_runtime.py ai_worker.py screen_analysis.py hybrid_trimap.py ai_backends/__init__.py ai_backends/birefnet_adapter.py` — syntax/import-surface compile check.
- `python -c "import app, keyer; print('import ok')"` — import check when PySide6 is installed.
- `python -c "import sys, app, keyer; blocked={'torch','torchvision','transformers','timm','kornia','einops','accelerate','huggingface_hub','safetensors','skimage','onnxruntime','onnxruntime_gpu','pymatting','scipy','numba'}; loaded=sorted(m for m in blocked if m in sys.modules); assert not loaded, f'blocked optional/heavy modules imported at default startup: {loaded}'; print('default dependency fence ok')"` — dependency-fence verification.
- `python -c "import sys, app, keyer, ai_assist, gpu_runtime, screen_analysis, hybrid_trimap; import ai_backends; blocked={'torch','torchvision','transformers','timm','kornia','einops','accelerate','huggingface_hub','safetensors','skimage','scipy','onnxruntime','onnxruntime_gpu'}; loaded=sorted(m for m in blocked if m in sys.modules); assert not loaded, f'AI/heavy modules imported at startup: {loaded}'; print('AI import fence ok')"` — AI import-fence verification.
- `python -m PyInstaller --noconfirm --clean ImgKey.spec` — default non-AI onefile/windowed EXE build that produces `dist\ImgKey.exe`.
- `python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec` — optional GPU runtime EXE build after installing `requirements-gpu-runtime-cu128.txt`.
- `python -m PyInstaller --noconfirm --clean ImgKey-GPU-BiRefNet.spec` — optional GPU BiRefNet EXE build after installing GPU runtime + BiRefNet requirements and satisfying local/bundled model policy.

## Change boundaries / risky areas
- Do not rewrite the algorithm or UI while doing repo-context/baseline-safety work.
- Large-image keying must avoid full-image float32 RGB allocations; keep source as `uint8`, masks as `uint8`, nearest-inner labels as bounded `int32`, and use float work per tile/ROI only.
- Global screen sampling, connected-background decisions, trimaps, semantic masks, fringe masks, and capped nearest-inner repair labels must happen before tiled export to avoid seams.
- v3/v5 UI defaults are the user-approved **High Accuracy Graphic** Blue preset: key color `(30, 80, 235)`, sample size `10`, tolerance `0.45`, softness `0.01`, clip background `0.97`, clip foreground `0.00`, matte gamma `2.20`, core strength `0.38`, edge radius `32`, erode/expand `-8`, despill `0.70`, decontaminate `0.50`, luminance restore/protect `0.76`, fringe remove `0.75`, edge color repair `0.65`, inner color pull `0.45`, and fringe band radius `3`.
- v5 linear-light repair rule: edge repair is RGB-only and alpha/edge/fringe-gated; unmix, Vlahos-style channel clamp, nearest-inner color pull, and luminance protection operate in linear light with linear key/screen vectors, then convert back to sRGB. Do not turn it into a global color-grade pass, do not change alpha from the repair path, and always zero RGB where alpha is 0.
- v5 guided-filter rule: `guided_alpha_refine` defaults to `0.0` (off). When enabled, guided alpha refinement must use grayscale/linear-luma edge-band ROI work only, clamp exact known background/foreground/core regions afterward, obey `guided_max_pixels`, and deterministically skip/fall back to unchanged alpha when the cap would be exceeded.
- v5 tile-local screen rule: the local screen model may build a full-image `uint8` screen map only under `max_local_screen_model_pixels`; otherwise tiled render must estimate screen color inside each read tile from connected/background-safe pixels. Read overlap must include the screen-estimation radius and write only tile cores.
- v5 tile-local nearest-inner rule: keep the global `int32` label map below cap; when labels are skipped and `inner_color_pull > 0`, build tile-local labels only inside the overlapped read tile, require enough nearby clean inner pixels, bound useful radius by overlap/margins, and fall back to unmix/clamp when local labels are absent/too far. Read overlap must include edge radius, fringe band, guided radius, screen radius, and local nearest-inner radius; write only tile cores.
- v5 crop-render contract: when `settings.full_res_crop`/render crop is active, global matte decisions still use the full image, color rendering is crop+overlap only, and `KeyResult.rgba`, `alpha`, `despill_mask`, `fringe_mask`, `screen_probability`, `alpha_hint`, debug RGB/display arrays must be crop-shaped and mutually aligned while UI metadata can still report original source coordinates/size.
- The default dependency fence remains `numpy`, `opencv-python`, `Pillow`, `PySide6`, and stdlib only. Do not add PyMatting/SciPy/numba/PyTorch/CUDA/model dependencies to `requirements.txt` or `ImgKey.spec`.
- v6 GPU edition scope is BiRefNet-only. Do not package Matting Anything, SAM, U2Net, MODNet, ViTMatte, CorridorKey models, or other AI model packages/weights.
- BiRefNet runtime must be local/offline only: no hidden downloads, no URL/repo-ID model paths, and bundled model builds require reviewed license/notice files plus SHA256 manifest hashes.
- Verify the dependency fence before phase completion with the blocked-module import check and source diff inspection; optional AI/plugin seams must not import heavy runtimes at startup.
- User approved `Aggressive Interior Removal` as the v3 default/reset target; do not revert it to connected-background default without explicit direction.
- Space is a spring-loaded pan override: holding Space temporarily pans even while Pick is active, and releasing Space restores the prior tool without toggling toolbar state.
- Stop and ask before bundling or redistributing model weights, CorridorKey, noncommercial assets, or any optional AI runtime outside the approved BiRefNet-only GPU path.

## Common workflows
- Keep generated diagnostics, screenshots, temporary exports, and backup snapshots under `.artifact/` only.
- Keep model downloads/caches out of the source tree or in ignored cache folders.
- Run `python smoke_test.py`, the required `py_compile`, `python -c "import app, keyer; print('import ok')"`, the default dependency fence, and the AI import fence before marking a v6 phase complete. Use `docs/build-gpu.md` for GPU packaging commands and clean-target expectations.

## Notes for agents
- This folder may not be a git repo; create a non-destructive timestamped backup under `.artifact/source-backup-*` before deep rewrites if version control is still absent.
- Treat `build/`, `dist/`, generated PNG exports, and model caches as disposable artifacts.
