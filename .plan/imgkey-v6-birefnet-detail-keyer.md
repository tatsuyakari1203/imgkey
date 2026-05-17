# 05 - ImgKey v6 BiRefNet Detail Keyer

Date: 2026-05-18
Status: In progress
Owner: ImgKey AI/GPU Detail Keyer
Scope: Integrate BiRefNet as the only AI model path for detail-preserving alpha hints, then merge it with the classical chroma keyer and keep classical RGB cleanup.

---

## 1) Goal

Implement **BiRefNet-only** hybrid detail keying:

```text
Classical keyer -> high-confidence key background + RGB cleanup authority
BiRefNet        -> foreground/detail protection alpha hint
Hybrid merge    -> final alpha from classical + BiRefNet, not BiRefNet alone
RGB cleanup     -> classical screen/clean-plate unmix/despill using final hybrid alpha
```

Hard model scope:
- Use **BiRefNet only**.
- Do not implement Matting Anything, SAM, U2Net, MODNet, ViTMatte, or any other model.
- No hidden model downloads at runtime.
- No `torch` import on normal app startup.
- Default lightweight `ImgKey.spec` build remains non-AI.

---

## 2) Context / decisions

- Current public release: `v1.1.0`, classical v5 keyer.
- User wants a heavy self-contained GPU/AI EXE and is fine with large size.
- User's GPU target: RTX 5060 Ti / RTX 50-series, so PyTorch CUDA must be recent enough for Blackwell support; avoid old `cu121`/`cu124` stacks.
- BiRefNet public repo / Hugging Face metadata appears MIT-licensed from current search results, but packaging must still verify exact model/code license files before bundling weights.
- BiRefNet integration should protect small detail; final chroma removal and edge color cleanup remain classical.
- Existing CorridorKey/external AI seams remain legacy-disabled and out of v6 scope; no CorridorKey UI expansion, packaging, or runtime work is allowed in this plan.

---

## 3) Risks / stop conditions

- Stop and ask before public bundling if exact BiRefNet code/weights license, notices, or redistribution terms are unclear.
- Before any BiRefNet inference phase, choose and record the exact local model snapshot: repo/source, commit/revision, directory layout, license file, SHA256 manifest, and expected local path. Reject repo IDs/URLs at runtime; only local paths are valid.
- BiRefNet loading must be offline/local-only. If Hugging Face/Transformers APIs are used, force offline/local-files-only behavior and pass a network-denied test.
- Stop and ask before changing distribution away from onefile EXE if PyInstaller GPU+model asset exceeds practical size limits; provide measured size/build evidence.
- GPU build can be multiple GB, but keep it separate from lightweight classical asset.
- GitHub-hosted Windows runners likely cannot validate CUDA; local RTX machine or self-hosted GPU runner is required for real GPU verification.
- AI worker must isolate torch/model failures so UI does not crash.
- All runtime caches, model downloads, wheels, `build/`, and `dist/` stay out of git.
- Preserve current manual mask priority unless explicitly changed: keep masks override remove masks in conflicts.

---

## 4) Phases

### Phase 1 - Keep app stable before AI

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `app.py`, `keyer.py`, `smoke_test.py`; no torch/model dependency yet.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 1 implemented and verified; app remains default classical/non-AI with no torch/model imports.



#### P1.1 - Cancel stale previews and reduce UI churn
- Add preview cancellation: `PreviewThread.request_cancel()` and `cancel_callback` into `process_key_image()`.
- Invalidate generation as soon as settings/sliders schedule a new preview.
- Cancel old preview on slider/settings/image changes and ignore stale results.
- Avoid QPixmap re-upload when only background mode changes.
- Avoid large temporary arrays for missing AI Hint/mask debug views where practical.

Acceptance:
- Rapid slider changes do not finish stale heavy previews.
- No zoom reset regression; Space pan unchanged.
- `python smoke_test.py`, compile/import checks pass.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Preview jobs now cancel via `PreviewThread.request_cancel()` and `process_key_image(cancel_callback=...)`; scheduling invalidates generations immediately, stale results/progress are ignored, background-only changes avoid pixmap re-upload except Split Compare, and missing mask/hint debug views use direct QImage blanks/grayscale instead of large RGB temporary arrays.



#### P1.2 - Low-memory export result mode
- Add `include_debug` or equivalent to `process_key_image()`.
- Export path avoids `foreground = rgba[:, :, :3].copy()` and avoids retaining debug masks not needed to write PNG.
- Preserve preview/debug behavior when debug arrays are requested.
- Measure before/after export peak memory on a representative large synthetic fixture.

Acceptance:
- Export memory peak reduced; decoded PNG pixels identical or within existing tolerance.
- `process_chroma_key()` remains compatible.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `include_debug` to `process_key_image()`; export and `process_chroma_key()` use low-memory result mode, skipping foreground RGB copies and retained debug masks while preserving RGBA output. Representative 2048×1536 synthetic fixture retained result field bytes dropped from 39.0 MiB to 15.0 MiB with identical RGBA; unique export result storage is the RGBA array plus an alpha view.


---

### Phase 2 - GPU probe

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `gpu_runtime.py`, docs/tests. No model inference yet. UI display of GPU Status belongs to Phase 5.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `gpu_runtime.py` with lazy torch-only-inside-probe CUDA diagnostics, `python -m gpu_runtime --probe --json`, nvidia-smi driver/GPU/VRAM reporting, CUDA device metadata, and a CUDA matmul smoke test. CPU/no-torch environments return valid JSON with an actionable unavailable status and do not crash.



#### P2.1 - Add scriptable CUDA probe
- Create `gpu_runtime.py`.
- `torch` imports only inside probe functions, never at module import time.
- Add CLI:
  - `python -m gpu_runtime --probe --json`
  - packaged target later: `ImgKey-GPU.exe --gpu-probe --json`.
- Probe fields:
  - torch import success/error,
  - torch version,
  - CUDA version,
  - `torch.cuda.is_available()`,
  - GPU name,
  - device capability / arch list where available,
  - VRAM total/free where available,
  - `nvidia-smi` driver info where available,
  - CUDA matmul smoke test.

Acceptance:
- Default app import does not import torch.
- CPU/no-torch environment reports actionable unavailable status.
- GPU environment reports RTX 5060 Ti/CUDA/matmul success.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added smoke coverage for `gpu_runtime` import fencing and JSON probe shape using fake missing/CPU-only torch loaders; verified the real probe reports RTX 5060 Ti via nvidia-smi while PyTorch is not installed in this environment.


---

### Phase 3 - BiRefNet adapter

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `ai_backends/`, no UI worker yet. Adapter must require explicit local/bundled model path and never download.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 3 implemented and verified. Added BiRefNet-only `ai_backends` adapter plus checked-in offline manifest for the pinned `ZhengPeng7/BiRefNet` Hugging Face snapshot (`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`), with local-path-only validation, optional SHA256 enforcement, no runtime downloads, and no torch/transformers import at default startup.

Verification:
- 2026-05-18: Passed `python smoke_test.py`, required `py_compile`, `import app, keyer`, extended default dependency fence, and AI import fence including `ai_backends.birefnet_adapter`.



#### P3.0 - Record BiRefNet model snapshot and offline manifest
- Before real inference, choose the exact BiRefNet variant and local snapshot.
- Record source/repo, commit or revision, expected directory layout, license/notice files, SHA256 manifest for code/config/weights, and expected local/bundled path.
- Reject repo IDs and URLs at runtime; only local paths are accepted.
- Verify offline behavior with empty cache/network denied before adapter execution is considered complete.

Acceptance:
- A checked-in machine-readable manifest exists without committing model weights; adapter/worker consumes it to validate hashes/license/layout.
- Missing/URL/repo-ID/empty-cache/network-denied paths fail cleanly without network access.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `ai_backends/birefnet_manifest.json` recording source repo, pinned revision, expected files/layout, license metadata files, offline policy, and optional hash validation rules. Validator rejects empty paths, URLs, repo IDs, missing directories, and incomplete/empty local snapshots before any AI runtime import.



#### P3.1 - Implement BiRefNet-only adapter API
- Add:

```text
ai_backends/
  __init__.py
  birefnet_adapter.py
```

- Implement:

```python
def generate_alpha_hint(
    rgb_u8,
    model_path,
    device="cuda",
    max_side=1536,
    mode="global_plus_roi",
    tile_size=1024,
    tile_overlap=192,
    precision="fp16",
    progress_callback=None,
    cancel_callback=None,
) -> dict:
    return {"alpha_hint": alpha_u8, "message": "...", "tile_info": {...}}
```

- Supported first modes:
  - `global_only`: resize to `max_side`, run BiRefNet, upscale alpha.
  - `global_plus_roi`: global pass first, then high-res crop/ROI pass. ROI selection may use BiRefNet alpha edges alone in the adapter; when worker/keyer context is available, it must also accept classical edge/conflict masks supplied by the caller. Do not invent ROI from raw RGB only when classical context is required.
- Do not implement full tiled mode yet.
- Add a model-path manifest gate before inference: local directory must exist, contain expected model/config/code/weights files, include license/notice metadata, and match recorded hashes when bundled.
- Do not implement any other model.
- No hidden download; `model_path` must exist and be local/bundled. Runtime must set/obey offline mode such as `local_files_only=True`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` where relevant.

Acceptance:
- Adapter imports without torch at app startup.
- Adapter can be called in a torch-enabled environment with a local BiRefNet model path.
- Missing/repo-ID/URL model path fails cleanly without network access.
- Output `alpha_hint` is HxW `uint8`, same shape as input.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `ai_backends/__init__.py` and `ai_backends/birefnet_adapter.py` with lazy inference imports, `generate_alpha_hint(...)`, `global_only` support, conservative `global_plus_roi` global fallback metadata, local `AutoModelForImageSegmentation.from_pretrained(..., local_files_only=True, trust_remote_code=True)`, offline env handling, shape helpers, and focused smoke coverage.


---

### Phase 4 - AI worker subprocess

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `ai_worker.py` and worker tests. UI integration waits until Phase 5.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 4 implemented and verified. Added isolated BiRefNet-only `ai_worker.py` JSON worker/CLI with safe output and temp handling under `.artifact/` or explicit dirs, cancellation-file support, structured failure responses, diagnostics JSON, and smoke coverage that does not require torch/model weights.

Verification:
- 2026-05-18: Passed `python smoke_test.py`, required `py_compile`, `import app, keyer`, default dependency fence, AI import fence including `ai_worker`, and CLI missing-input JSON failure test.


#### P4.1 - Add BiRefNet worker process
- Create `ai_worker.py`.
- Torch/model failures must not crash the main UI.
- Support request JSON:

```json
{
  "backend": "birefnet",
  "input_image_path": "source.png",
  "model_path": "models/BiRefNet",
  "device": "cuda",
  "mode": "global_plus_roi",
  "max_side": 1536,
  "tile_size": 1024,
  "tile_overlap": 192,
  "precision": "fp16"
}
```

- Response JSON:

```json
{
  "ok": true,
  "alpha_hint_path": "alpha_hint.png",
  "diagnostics_path": "diagnostics.json",
  "message": "BiRefNet completed"
}
```

- Include cancel/error/OOM reporting.
- Write temp outputs under `.artifact/` or a user temp directory, never source root.

Acceptance:
- Worker handles invalid model path, CUDA unavailable, and cancellation with clear JSON errors.
- Worker can run probe/self-test without UI.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: `ai_worker.py` supports `python ai_worker.py --request <json-or-file-or-stdin> [--json]`, validates the BiRefNet-only contract, rejects unsupported backends/local path errors/missing inputs before model runtime import, maps dependency/CUDA/OOM/cancel failures to structured JSON errors, writes alpha PNG plus diagnostics on success, and cleans staging temp directories on failure.


---

### Phase 5 - UI BiRefNet controls

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `app.py` UI wiring and debug view additions. Do not change core alpha merge yet.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 5 implemented and verified. Added UI controls/actions for Generate BiRefNet Hint, Cancel AI, GPU Status, and BiRefNet Alpha view; BiRefNet worker/probe subprocesses run asynchronously and generated alpha is stored only as separate `biref_alpha_mask` state for display, not classical preview/export.

Verification:
- 2026-05-18: Passed `python smoke_test.py`, required `py_compile`, `import app, keyer`, default dependency fence, AI import fence, and headless UI probe confirming controls/view mode exist without heavy AI imports.



#### P5.1 - Add BiRefNet UI flow
- Add controls/actions:
  - `Generate BiRefNet Hint`,
  - `Cancel AI`,
  - `GPU Status`,
  - view mode `BiRefNet Alpha`.
- Status text: model ready / running / done / failed.
- Generated BiRefNet alpha must be stored as distinct `biref_alpha` state/input, not silently mixed into the existing manual `alpha_hint` path.
- For this phase, `biref_alpha` is stored/displayed only; it is not consumed by preview/export until the later hybrid wiring milestone.
- Before Phase 7/8, UI status must clearly say generated hint is an alpha hint, not final hybrid output.
- Do not block UI while AI runs.

Acceptance:
- App can generate/cancel/load BiRefNet alpha hint via worker.
- Existing manual alpha hint import still works.
- Default app startup still does not import torch.
- Worker lifecycle tests cover invalid model path, CUDA unavailable, OOM/error reporting, cancellation during load/inference, temp cleanup, and no zombie process.
- Mode-isolation test proves generated BiRefNet hints do not change classical preview/export in this phase.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: `app.py` now launches `ai_worker.py` through `QProcess`, writes cancel flags and terminates/kills stuck AI subprocesses, loads worker output into distinct `biref_alpha_mask`, and reports model ready/running/done/failed/cancelled states with explicit alpha-hint-only wording. Manual `alpha_hint_mask` import remains separate and still controls the existing `AIHint` classical mode.

Verification:
- 2026-05-18: Smoke coverage extended with worker subprocess failure/temp-cleanup checks and a headless MainWindow UI/isolation/cancel-cleanup probe; generated BiRefNet state does not change `current_settings().mode` while manual alpha hints still do.


---

### Phase 6 - Classical screen analysis and hybrid trimap

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `screen_analysis.py`, `hybrid_trimap.py`, and tests. No RGB cleanup changes yet.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 6 implemented standalone classical screen analysis maps and the BiRefNet/classical hybrid trimap helper without wiring a new keyer mode, RGB cleanup path, torch/model runtime, or default-startup heavy imports.

Verification:
- 2026-05-18: Added smoke coverage for green, blue, cyan-ish, and uneven-lit screen analysis fixtures; hybrid trimap known-bg/known-fg/conflict/unknown/manual override/detail region behavior; import fences; and low-memory export non-retention of new Phase 6 maps.
- 2026-05-18: Passed `python smoke_test.py`, required `py_compile`, `import app, keyer`, extended default dependency fence, and AI import fence including `screen_analysis`/`hybrid_trimap`.



#### P6.0 - Add classical screen analysis maps
- Create `screen_analysis.py`.
- Inputs: `rgb_u8`, `classical_alpha`/background mask when available, keyer settings, optional user-picked screen color, optional keep/remove masks.
- Outputs: `screen_color_rgb`, `screen_plate_rgb`, `screen_probability`, `screen_distance`, `spill_probability`, `classical_confidence`, `edge_mask`, and `fringe_mask`.
- Algorithm:
  - estimate screen color from border/corner samples using trimmed median,
  - reject foreground-contaminated samples using saturation/hue/chroma outlier filtering,
  - support green, blue, and cyan-ish screens,
  - compute screen distance in normalized RGB and optional Lab/YCrCb distance,
  - build `screen_probability` from smoothstep distance to estimated screen color,
  - build low-frequency `screen_plate_rgb` resolver for uneven lighting by downsampling background candidates, filling holes, blur/interpolate, then resolving full-size/tile values only where needed,
  - build `edge_mask` from alpha/chroma gradient,
  - build `fringe_mask` from semi-transparent alpha band, edge dilation, and high spill probability.
- Storage/memory rules:
  - masks are bool/`uint8`,
  - scalar probability/distance maps are `uint8` unless a phase explicitly needs a temporary tile/ROI float,
  - do not retain full-image float32 Lab/linear RGB/screen plate for large images,
  - `screen_plate_rgb` must be represented as a capped low-res map or tile resolver for large sources; do not require retaining full-size HxWx3 RGB when source exceeds cap.

Acceptance:
- Works without BiRefNet and without torch.
- Stable on green, blue, cyan-ish, and uneven-lit screen fixtures.
- Exposes debug images for screen color, screen probability, screen distance, screen plate, spill probability, edge mask, and fringe mask.
- Export with `include_debug=False` must not retain these debug maps.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `screen_analysis.py` with robust screen-color estimation, uint8 screen probability/distance/spill/confidence maps, edge/fringe masks, and capped/low-res `ScreenPlateRGB` resolver storage for uneven lighting.



#### P6.1 - Add BiRefNet/classical trimap merge helper
- Create `hybrid_trimap.py`.
- Inputs:
  - `classical_alpha`,
  - `screen_probability`,
  - `screen_distance`,
  - `spill_probability`,
  - `classical_confidence`,
  - `background_mask`,
  - `edge_mask`,
  - `fringe_mask`,
  - `screen_plate_rgb` as a capped low-res/tile resolver, not necessarily full-size HxWx3 storage,
  - `biref_alpha`,
  - keep/remove masks.
- Return a durable result object/dataclass containing at least `known_bg`, `known_fg`, `unknown`, `conflict`, `soft_unknown`, `hard_unknown`, `spill_region`, `unmix_region`, `despill_region`, `protected_fg`, `safe_bg`, and any debug masks needed by Phase 8 diagnostics/RGB cleanup.
- Masks must be mutually exclusive after dilation and manual overrides; if in doubt, reapply `known_bg`/`known_fg` clamps after computing `unknown`.
- Define constants/inputs explicitly: `spill_threshold`, `manual_keep_core`, `strong_edge_band`, and the alpha used for candidate regions. Before final hybrid alpha exists, region outputs are candidates; Phase 8 must recompute/refresh final `unmix_region` and `despill_region` using final alpha.
- Conflict precedence: conflict/hard-unknown overrides automatic `known_fg`; manual keep is the only foreground override that can win over conflict.
- Logic:

```python
known_bg = (
    (screen_probability >= 245)
    & (classical_alpha <= 8)
    & (biref_alpha <= 24)
)
known_fg = (biref_alpha >= 220) | (classical_alpha >= 245)
conflict = (screen_probability >= 245) & (biref_alpha >= 96)
known_fg &= ~conflict
unknown = ~(known_bg | known_fg)
unknown |= dilate(edge_mask)
unknown |= dilate(conflict)
unknown &= ~(known_bg | known_fg)
soft_unknown = unknown | dilate(fringe_mask, r=2)
hard_unknown = dilate(conflict, r=4) | strong_edge_band
candidate_alpha = classical_alpha
spill_region = (
    (candidate_alpha > 0)
    & (candidate_alpha < 250)
    & (spill_probability > spill_threshold)
)
unmix_region = (
    hard_unknown
    | soft_unknown
    | ((candidate_alpha > 8) & (candidate_alpha < 245))
)
despill_region = (
    spill_region
    & ~known_bg
    & ~manual_keep_core
)
protected_fg = known_fg & (screen_probability < 128)
safe_bg = known_bg & (screen_probability >= 245)
```

- Manual masks override:
  - keep => foreground and overrides remove,
  - remove => background only where keep is not set.

Acceptance:
- Unit/smoke tests cover known BG, known FG, conflict, manual overrides, and detail preservation masks.
- Tests explicitly verify keep-over-remove priority.
- `unmix_region` and `despill_region` are available to Phase 8.
- Manual keep regions are protected from aggressive despill.
- Manual remove regions are eligible for hard background cleanup where not overridden by keep.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `hybrid_trimap.py` with explicit thresholds/inputs, mutually exclusive durable trimap classes, conflict/hard-unknown precedence over automatic foreground, keep-over-remove priority, and candidate spill/unmix/despill/protected/safe regions for later phases.


---

### Phase 7 - Hybrid alpha mode

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `keyer.py`, `smoke_test.py`. Preserve existing classical modes.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 7 completed `HybridBiRefNet` alpha mode and unknown-only alpha refinement with explicit BiRefNet/classical/manual/source-alpha ordering.

Verification:
- 2026-05-18: Passed `python smoke_test.py` including new Phase 7 HybridBiRefNet alpha tests.



#### P7.1 - Add `HybridBiRefNet` mode
- Add mode `HybridBiRefNet` in `keyer.py` while preserving existing classical modes.
- Do not use BiRefNet output directly as final alpha.
- Alpha merge:

```python
alpha = classical_alpha.copy()
alpha[known_bg] = 0
alpha[known_fg] = np.maximum(classical_alpha, biref_alpha)[known_fg]
w = smoothstep(64, 220, biref_alpha)
alpha[unknown] = classical_alpha[unknown] * (1 - w[unknown]) + biref_alpha[unknown] * w[unknown]
```

- Guided refine only in unknown if enabled.
- `known_bg` always clamps to 0.
- `known_fg` is preserved.
- Reapply `known_bg` and `known_fg` clamps after unknown blending so unknown cannot overwrite known decisions.
- Final alpha ordering must be explicit and tested:
  1. compute classical alpha/background/probability,
  2. merge BiRefNet with hybrid trimap,
  3. apply unknown-only refinement,
  4. reapply automatic known/background/foreground clamps,
  5. apply manual overrides with current priority: keep wins over remove,
  6. apply original source alpha as the final cap so transparent/semi-transparent source pixels remain capped even in known-fg and manual-keep cases.

Acceptance:
- Hybrid mode improves detail retention on synthetic thin-detail fixtures without increasing confident key-background leaks.
- Classical modes remain unchanged.
- Hybrid tests cover full vs crop parity, tile seam diff, known-bg clamp, manual overrides, and `alpha==0 => RGB==0`.
- Original source alpha remains a cap in `HybridBiRefNet`, including known-fg and manual-mask cases.
- Unknown-only guided refine skips deterministically when ROI/pixel cap such as `guided_max_pixels` would be exceeded.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added `HybridBiRefNet` as a keyer mode requiring explicit `biref_alpha`; BiRefNet alpha is blended with classical alpha through `hybrid_trimap.py` instead of being used directly, known-bg/known-fg clamps are reapplied, manual keep wins over remove, and original source alpha caps final alpha last.


#### P7.2 - Unknown-only alpha refinement
- Implement optional classical alpha refinement after `HybridBiRefNet` merge.
- Work only inside `unknown` / `soft_unknown`.
- Use edge-aware guided filter or bilateral-like smoothing guided by RGB/luma; implement with OpenCV box filters/NumPy first, no new dependency such as `opencv-contrib` required.
- Use ROI/pixel caps such as `guided_max_pixels`; if the unknown ROI exceeds cap, skip deterministically or use an explicitly bounded ROI/stripe path.
- Never smooth across `known_bg` / `known_fg` clamps.
- Preserve thin details from BiRefNet by using `biref_alpha` as detail prior.
- Suggested flow:

```python
alpha_f = alpha.astype(np.float32) / 255.0
guide = linear_rgb_or_luma(rgb_u8)
refined = guided_filter(
    guide=guide,
    src=alpha_f,
    radius=8_to_32,
    eps=1e-4_to_1e-2,
    mask=unknown,
)
alpha[unknown] = blend(alpha_f, refined, refine_strength)[unknown]
```

- Reapply final clamps after refinement:
  - `known_bg => alpha = 0`,
  - `known_fg => max(classical_alpha, biref_alpha)`,
  - `manual_keep => alpha = 255`,
  - `manual_remove => alpha = 0 unless keep`,
  - original source alpha remains the final cap.

Acceptance:
- Reduces jagged alpha edge on synthetic fixtures.
- Does not expand background leak in confident screen areas.
- Does not erase thin BiRefNet-retained hair/detail.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Added optional unknown/soft-unknown-only guided alpha refinement using existing OpenCV/NumPy guided-filter helpers, ROI pixel-cap deterministic skip behavior, known-region clamps, and BiRefNet detail-prior preservation.


---

### Phase 8 - RGB cleanup with final hybrid alpha

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own final color cleanup path in `keyer.py` and tests. Alpha behavior from Phase 7 must remain stable.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-18: Phase 8 completed. HybridBiRefNet RGB cleanup now recomputes final unmix/despill/protection regions from final capped alpha, resolves a bounded local clean screen plate, performs linear-light alpha-aware unmix plus edge-only despill, and keeps classical modes isolated.

Verification:
- 2026-05-18: Added and passed smoke coverage for hybrid composite halo reduction, foreground-core color tolerance, source-alpha caps, low-alpha noise suppression, classical mode isolation, and UI mode selection/probe. Verification passed: `python smoke_test.py`, required `py_compile`, `import app, keyer`, extended default dependency fence, AI import fence, and headless UI probe.



#### P8.1 - Apply classical screen/clean-plate cleanup to hybrid alpha
- Use final hybrid alpha to drive RGB cleanup.
- Recompute/refresh final fringe/detail masks plus `unmix_region` and `despill_region` after final hybrid alpha, manual overrides, source-alpha cap, and clamp decisions; candidate regions from Phase 6 must not be used blindly if alpha changed.
- Build/use screen/clean plate from `known_bg`/local screen model.
- Unmix RGB in unknown/fringe/detail regions:

```python
F = (I - (1 - alpha) * B) / max(alpha, eps)
```

- Despill is RGB-only; never changes alpha.
- Keep `alpha == 0 => RGB = 0`.
- Additional requirements:
  - convert sRGB `uint8` to linear RGB before unmix/despill,
  - perform compositing math in `float32` linear RGB,
  - convert back to sRGB only at final output,
  - clamp safely after color reconstruction.

Acceptance:
- BiRefNet-retained details do not keep visible green/blue key halo on black/white/gray/checkerboard composites.
- Foreground core RGB delta remains within v5 tolerance.

Status:
- Completed

Current:
- No


#### P8.1a - Build local screen/clean plate
- Use `known_bg` and `safe_bg` to estimate local background color `B`.
- Start from `screen_plate_rgb` from Phase 6.
- For missing/uncertain areas, use estimated global screen color, nearest valid background sample, or low-res interpolation + blur.
- Keep the plate low-frequency to avoid copying texture/noise into foreground.
- Support user clean-plate later, but do not require it in v6.

Acceptance:
- Uneven green/blue screen lighting does not create harsh edge color error.
- `B` is stable on gradient screen fixtures.

Status:
- Completed

Current:
- No


#### P8.1b - Alpha-aware foreground unmix
- Apply unmix only in `unmix_region`.
- Algorithm:

```python
a = alpha_float[..., None]
eps = 1 / 255.0
F_unmixed = (I_linear - (1.0 - a) * B_linear) / np.maximum(a, eps)
F_unmixed = clamp(F_unmixed, 0.0, 1.0)
```

- Low-alpha stabilization:

```python
low_alpha = alpha < 0.15
mid_alpha = (alpha >= 0.15) & (alpha < 0.60)
F_repaired = nearest_or_blurred_solid_foreground_color
F = where(low_alpha, blend(F_unmixed, F_repaired, 0.7), F_unmixed)
F = where(mid_alpha, blend(F_unmixed, F_repaired, 0.25), F)
```

Acceptance:
- Semi-transparent hair/fringe loses green/blue contamination.
- Low-alpha pixels do not explode into noisy saturated colors.

Status:
- Completed

Current:
- No


#### P8.1c - Edge-only despill after unmix
- Despill must run after unmix, not before.
- For green screen: `spill = max(0, G - max(R, B)); G_new = G - spill * strength`.
- For blue screen: `spill = max(0, B - max(R, G)); B_new = B - spill * strength`.
- Rules:
  - strong despill only in `despill_region`,
  - medium despill in `fringe_mask`,
  - weak/no despill in protected foreground core,
  - do not alter alpha,
  - do not force the screen channel below natural foreground color if user keep mask covers that region.

Acceptance:
- Green/blue halo removed on black/white/gray/checker composites.
- Real green/blue foreground objects are not destroyed in opaque core.

Status:
- Completed

Current:
- No


#### P8.1d - Final RGBA invariants
- After all cleanup:
  - `alpha == 0 => RGB = 0`,
  - `known_bg => alpha = 0`,
  - `manual_remove => alpha = 0 unless keep`,
  - `manual_keep` may request protected foreground but must not raise alpha above the final P7 source-alpha-capped alpha,
  - source alpha is re-applied last; P8 must never raise alpha above the final P7 capped alpha,
  - output PNG stores straight alpha,
  - no NaN/Inf in RGB or alpha.

Acceptance:
- Exported PNG has clean transparent pixels.
- Compositing on black/white/gray/checker produces no hidden green/blue garbage.
- Semi-transparent source pixels remain capped in known-fg and manual-keep cases.

Status:
- Completed

Current:
- No


#### P8.2 - Wire hybrid mode into UI preview/export
- Update `app.py` so generated BiRefNet hints can drive `HybridBiRefNet` preview/export after Phases 7/8.
- Preserve manual alpha-hint import behavior and existing classical modes.
- UI should distinguish:
  - imported/manual AI hint,
  - generated BiRefNet alpha hint,
  - final HybridBiRefNet result.
- Export must use the same selected hybrid/classical mode as preview.

Acceptance:
- Generated BiRefNet hint is actually used by `HybridBiRefNet` for preview/export.
- Manual hint import remains compatible.
- Classical export remains unchanged when hybrid mode is not selected.
- Mode-isolation test proves generated BiRefNet hints do not change classical preview/export unless `HybridBiRefNet` is selected.

Status:
- Completed

Current:
- No


---

### Phase 9 - BiRefNet diagnostics

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `smoke_test.py` diagnostics and `.artifact/` output. No algorithm changes unless fixing test bugs.

Status:
- Planned

Current:
- Yes



#### P9.1 - Add diagnostics command
- Add:

```powershell
python smoke_test.py --write-birefnet-diagnostics
```

- Output under `.artifact/birefnet-diagnostics/`:
  - `source.png`,
  - `classical_alpha.png`,
  - `birefnet_alpha.png`,
  - `screen_probability.png`,
  - `screen_distance.png`,
  - `screen_plate.png`,
  - `spill_probability.png`,
  - `fringe_mask.png`,
  - `hybrid_known_bg.png`,
  - `hybrid_known_fg.png`,
  - `hybrid_unknown.png`,
  - `hybrid_conflict.png`,
  - `unmix_region.png`,
  - `despill_region.png`,
  - `hybrid_alpha.png`,
  - `rgb_before_cleanup.png`,
  - `rgb_after_unmix.png`,
  - `rgb_after_despill.png`,
  - `alpha_edge_overlay.png`,
  - `result.png`,
  - black/white/gray/checker composites,
  - `metrics.json`.
- Metrics:
  - detail retention,
  - background leak,
  - edge key-color residual,
  - foreground core RGB delta,
  - tile seam diff.
  - `transparent_rgb_residual_max`,
  - `edge_green_residual_mean` / `edge_blue_residual_mean`,
  - `low_alpha_noise_score`,
  - `despill_core_color_delta`,
  - `known_bg_false_positive_area`,
  - `known_fg_preservation_score`.

Acceptance:
- Diagnostics run without bundled model by using synthetic/mock BiRefNet alpha; real BiRefNet diagnostics run when model path is available.
- `smoke_test.py` CLI parser explicitly accepts `--write-birefnet-diagnostics` and works without a real model by using synthetic/mock BiRefNet alpha.
- On black/white/gray/checker composites, edge key-color residual decreases after P8 cleanup.
- Foreground core RGB delta stays within tolerance.
- `alpha == 0` pixels have exact RGB zero.
- Low-alpha pixels do not contain saturated green/blue noise.

Status:
- Planned

Current:
- Yes


---

### Phase 10 - Packaging flavors

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own requirements/spec/workflow/docs. Keep model scope BiRefNet-only.

Status:
- Planned

Current:
- No



#### P10.1 - Keep three build flavors
- Maintain:
  1. `ImgKey classical`: no torch, no model.
  2. `ImgKey GPU runtime`: torch CUDA, no model.
  3. `ImgKey GPU BiRefNet`: torch CUDA + BiRefNet only.
- Add separate files as needed:
  - `requirements-gpu-birefnet-cu128.txt`,
  - `ImgKey-GPU.spec`,
  - `ImgKey-GPU-BiRefNet.spec`.
- Before bundling model weights, verify and record exact license/notice/size for the selected BiRefNet package/weights.
- No other model packages or weights.

Acceptance:
- Classical EXE remains non-AI and dependency-fenced.
- GPU runtime EXE can probe CUDA without model.
- GPU BiRefNet EXE can generate alpha hint with bundled/local BiRefNet, subject to license gate.

Status:
- Planned

Current:
- No


---

## 5) Classical algorithm addendum - keep AI scope unchanged

This plan keeps BiRefNet as the only AI model. All additions below are deterministic/classical image-processing steps and must not add torch/model dependencies to the default app.

Core classical maps:
- `screen_color_rgb`: robust estimated green/blue/cyan-ish screen color.
- `screen_plate_rgb`: local low-frequency clean screen estimate.
- `screen_probability`: probability/confidence that a pixel belongs to key background.
- `spill_probability`: probability that a foreground/fringe pixel contains screen-color contamination.
- `edge_mask`: chroma/alpha edge band.
- `fringe_mask`: semi-transparent/detail band requiring unmix/despill.
- `unmix_region`: pixels where alpha-aware color reconstruction is allowed.
- `despill_region`: pixels where RGB-only spill suppression is allowed.

Color math:
- All compositing/unmix math must be done in linear RGB `float32`.
- Convert back to sRGB only at output.
- `alpha == 0 => RGB == 0` is a hard invariant.

Recommended order:
1. Estimate screen color / local screen plate.
2. Compute classical alpha and screen probability.
3. Generate BiRefNet alpha hint.
4. Build hybrid trimap.
5. Merge final hybrid alpha.
6. Refine alpha only inside unknown/fringe regions.
7. Reconstruct foreground RGB using alpha-aware unmix.
8. Apply edge-only despill.
9. Enforce RGBA invariants.
10. Write diagnostics and composites.

---

## 6) Verification floor

Always run after source phases:

```powershell
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py gpu_runtime.py ai_worker.py screen_analysis.py hybrid_trimap.py ai_backends/__init__.py ai_backends/birefnet_adapter.py
python -c "import app, keyer; print('import ok')"
python -c "import sys, app, keyer; blocked={'torch','torchvision','transformers','timm','kornia','einops','accelerate','huggingface_hub','safetensors','skimage','onnxruntime','onnxruntime_gpu','pymatting','scipy','numba'}; loaded=sorted(m for m in blocked if m in sys.modules); assert not loaded, f'blocked optional/heavy modules imported at default startup: {loaded}'; print('default dependency fence ok')"
```

Omit newly planned files from `py_compile` only until the phase that creates them.

When AI modules exist, default-startup fence must also prove importing light modules does not import AI stacks even in an environment where they are installed:

```powershell
python -c "import sys, app, keyer, ai_assist, gpu_runtime, screen_analysis, hybrid_trimap; import ai_backends; blocked={'torch','torchvision','transformers','timm','kornia','einops','accelerate','huggingface_hub','safetensors','skimage','scipy','onnxruntime','onnxruntime_gpu'}; loaded=sorted(m for m in blocked if m in sys.modules); assert not loaded, f'AI/heavy modules imported at startup: {loaded}'; print('AI import fence ok')"
```

Default `ImgKey.spec` must continue excluding AI/GPU packages including `torch`, `torchvision`, `transformers`, `timm`, `kornia`, `einops`, `accelerate`, `huggingface_hub`, `safetensors`, `skimage`, `onnxruntime`, and model packages.

GPU probe:

```powershell
python -m gpu_runtime --probe --json
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); x=torch.randn(1024,1024,device='cuda'); print((x@x).mean().item())"
```

BiRefNet diagnostics:

```powershell
python smoke_test.py --write-birefnet-diagnostics
```

Only required after Phase 9 creates the command.

Offline/no-hidden-download verification for real BiRefNet runs:

```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; python ai_worker.py --request .artifact/birefnet-request.json
```

Use an empty temporary HF cache for one test; valid local model path must succeed and missing path must fail cleanly.

Packaging:

```powershell
python -m PyInstaller --noconfirm --clean ImgKey.spec
python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec
python -m PyInstaller --noconfirm --clean ImgKey-GPU-BiRefNet.spec
```

Packaging must also be tested on a clean Windows target with NVIDIA driver only: no Python packages and no CUDA Toolkit on PATH.

---

## 7) Immediate next step

Execute Phase 9/P9.1 next: add the BiRefNet diagnostics command and artifact output without changing the Phase 8 algorithm unless fixing diagnostics test bugs.
