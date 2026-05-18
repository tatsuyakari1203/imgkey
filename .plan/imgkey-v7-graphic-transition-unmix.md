# 06 - ImgKey v7 No-AI Classical GPU + Transition Unmix

Date: 2026-05-18
Status: Completed
Owner: ImgKey Classical Keyer / Classical GPU Runtime
Scope: Remove AI entirely, improve deterministic transition/fringe cleanup, and focus GPU work on classical CUDA acceleration.

---

## 1) Goal

Make ImgKey a **no-AI classical keyer** and fix the current classical quality/performance gaps:

```text
Hard/non-transition edge: already good; preserve it.
Transition/anti-aliased edge: still has light green/blue cast; repair it.
Alpha/detail: do not erode, despeckle, or threshold harder.
AI: removed entirely from product/runtime/source surface.
GPU: used only for deterministic classical math in a separate `ImgKey-GPU.exe` flavor.
```

Core model:

```text
I = alpha * F + (1 - alpha) * B
I = source pixel
B = sampled key plate / screen color
F = clean foreground reference
alpha = coverage
```

v7 treats edge green/blue pixels as **semi-transparent foreground mixed with key background**, not as extra background to delete.

Definition of “no AI” for v7:
- No model weights.
- No BiRefNet/CorridorKey/SAM/Matting Anything/U2Net/MODNet/ViTMatte.
- No Hugging Face/Transformers/timm/kornia/einops/accelerate/safetensors.
- No AI worker subprocess or model manifest.
- No AI wording in public UI/docs/specs.
- `torch` may remain only in the separate GPU flavor as a CUDA tensor/numerical runtime for classical kernels; default `ImgKey.exe` must not import or bundle it.

---

## 2) Context / current architecture

- Current branch: `feature/birefnet-detail-keyer`, latest known commit `6aaa420 Improve BiRefNet startup and mask quality`.
- Existing v5/v6 classical pipeline in `keyer.py` already has:
  - `KeySettings` color/edge controls,
  - global screen probability and connected background,
  - fringe masks,
  - global or tile-local nearest-inner foreground reference (`_nearest_inner_rgb_for_slice`, `_build_tile_local_nearest_inner_rgb`),
  - linear-light edge repair helpers,
  - tiled render via `_render_tiled_rgba()` and `_process_color_tile()`.
- v6 added hybrid/AI paths, but v7 supersedes them: remove those paths instead of improving them.
- Local GPU evidence: RTX 5060 Ti works with PyTorch CUDA cu128; `cupy` is not installed; current pip `opencv-python` has no usable CUDA devices, so the first practical GPU backend is a torch CUDA tensor runtime packaged only in `ImgKey-GPU.exe`.
- The target issue is graphic/poster assets: red/white/black hard shapes over green/blue/cyan key backgrounds with anti-aliased transition pixels retaining slight key color.

---

## 3) Risks / constraints

- Do **not** add AI/model work: remove BiRefNet/AI paths, no Matting Anything/SAM/etc., no hidden downloads.
- Do **not** increase deletion strength: no new erosion, no harder background threshold, no despeckling of foreground detail.
- Transition repair may only raise alpha when reconstruction is plausible; it must never reduce alpha.
- RGB repair must be region-gated to transition/fringe/detail pixels, not global color grading.
- Use linear RGB for compositing math; output remains straight-alpha PNG.
- Large-image rules still apply: no full-image float32 RGB retention in export; work per tile/ROI and keep masks compact.
- `alpha == 0 => RGB == 0` remains a hard invariant.
- Original source alpha is a hard cap: transition alpha recovery must run before final alpha is exposed to tiled rendering, and `_apply_original_alpha()` or an equivalent cap must be reapplied after recovery.
- Manual masks remain authoritative: keep protects foreground color/core, remove forces alpha/RGB to background unless keep overrides it.
- Default `ImgKey.spec` remains non-AI and dependency-fenced; source verification must prove no torch/transformers import at startup.
- GPU acceleration must be optional and isolated: `ImgKey-GPU.spec` may bundle torch CUDA only as a numerical runtime, with all model/AI packages excluded.
- Stop and ask if a proposed fix would resurrect fully transparent pixels globally; optional alpha resurrection is out of scope for v7 unless tightly edge-gated and explicitly approved.

---

## 4) Phases

Phase execution rule:
- Each implementation phase is a commit boundary. Phase owner must run that phase's targeted verification, inspect `git status --short --branch`, `git diff`, and `git log --oneline -10`, stage only intended source/plan/docs/test files, commit, and leave the branch clean before planner advances to the next phase.
- Never stage `build/`, `dist/`, `.artifact/`, model snapshots, wheels, or caches.

### Phase 0 - Remove AI product surface and runtime code

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own AI removal across `app.py`, `keyer.py`, `smoke_test.py`, docs/specs/requirements. No transition-unmix or GPU kernel implementation yet.

Status:
- Complete

Progress:
- 2026-05-18: Removed assisted-matte UI/runtime surface, deleted backend/worker/spec/requirements files, retained manual keep/remove and renamed retained matte import to `Imported Matte`.
- 2026-05-18: Updated docs/build specs/release workflow/repo context to list only `ImgKey.exe` and `ImgKey-GPU.exe` public build flavors; marked the v6 plan superseded/historical.
- 2026-05-18: Added smoke no-AI source/non-existence guards and updated verification to compile only retained helpers.


#### P0.1 - Remove AI UI and modes
- Remove UI/actions/status/view modes for Generate BiRefNet Hint, Cancel AI, BiRefNet Alpha, Hybrid BiRefNet output, Optional AI Adapters, legacy BiRefNet, and CorridorKey.
- Remove `biref_alpha` state flow from preview/export jobs.
- Remove `HybridBiRefNet` and `AIHint` public output modes, or rename any retained manual matte import to `Imported Matte` with no AI wording.
- Preserve manual keep/remove masks.

Execution:
- Serial

Isolation:
- `app.py`, associated tests. Do not change core keying math in this task.

Acceptance:
- UI contains no AI/BiRefNet/CorridorKey/model wording.
- Preview/export no longer requires or consumes `biref_alpha`.
- Existing classical modes still work.

Status:
- Complete


#### P0.2 - Remove AI worker/backend/source files
- Delete or retire:
  - `ai_assist.py`,
  - `ai_worker.py`,
  - `ai_backends/`,
  - `requirements-gpu-birefnet-cu128.txt`,
  - `ImgKey-GPU-BiRefNet.spec`,
  - BiRefNet diagnostics command/output references,
  - BiRefNet manifest/hash/model-path/runtime hook logic.
- Keep `gpu_runtime.py` only as generic GPU probe/runtime code with no AI language.
- Delete `hybrid_trimap.py` if it only serves BiRefNet/hybrid AI behavior; otherwise rename/refactor it into a strictly classical trimap helper with no AI/model terminology.
- Replace AI import-fence tests with no-AI source/import guards.

Acceptance:
- Deleted AI files/build specs/requirements no longer exist, or are explicitly renamed into no-AI equivalents.
- No app/test/spec imports deleted AI files.
- `grep`/tests show no public AI/model strings except historical `.plan/` files explicitly marked superseded.
- `python smoke_test.py` passes after test cleanup.

Status:
- Complete


#### P0.3 - Update docs and repo context for no-AI direction
- Update `AGENTS.md`, README/build docs, and plan references:
  - `ImgKey.exe`: classical CPU/default, no torch/CUDA/AI.
  - `ImgKey-GPU.exe`: classical GPU only, CUDA tensor runtime, no AI/model stack.
  - `ImgKey-GPU-BiRefNet.spec`: removed/superseded.
- Mark `.plan/imgkey-v6-birefnet-detail-keyer.md` as superseded by v7/no-AI.

Acceptance:
- A new agent reading repo context will not try to work on BiRefNet/AI.
- Docs list only no-AI build flavors.

Status:
- Complete


---






Current:
- No
### Phase 1 - Transition model and settings

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `keyer.py` settings/helpers and targeted smoke tests. No UI yet. No AI files/specs.

Status:
- Complete

Progress:
- 2026-05-18: Added diagnostic transition-unmix graphic fixtures/metrics, appended compatible v7 settings, and added covered transition/core mask helpers without wiring behavior.


#### P1.0 - Add baseline graphic transition fixtures first
- Add synthetic fixtures before algorithm wiring:
  - red anti-aliased slash on blue/green key,
  - white/black 1px text or barcode-like lines,
  - black tape edge,
  - source-alpha cap and transparent-RGB invariant cases.
- Build source by physically compositing `I = alpha*F + (1-alpha)*B` so expected alpha and foreground RGB are known.
- Add baseline metrics that can run before v7 behavior is enabled:
  - hard-edge/core RGB delta,
  - transition key residual,
  - alpha/detail recall,
  - background alpha leak,
  - composite residuals on black/white/gray/checker.

Execution:
- Serial

Isolation:
- `smoke_test.py` fixture/metric helpers only; no keyer behavior changes.

Acceptance:
- Baseline metrics reproduce the current issue: hard edge/core is clean but anti-aliased transition residual is measurable.
- Existing smoke tests still pass.

Status:
- Complete


#### P1.1 - Add v7 transition-unmix settings
- Add fields to `KeySettings`, appended for positional compatibility:
  - `transition_unmix: bool = True`
  - `alpha_recover_strength: float = 0.85`
  - `foreground_reference_pull: float = 0.65`
  - `key_vector_despill: float = 0.75`
  - `transition_spill_threshold: float = 0.08`
  - `transition_reconstruction_error: float = 0.08`
  - `foreground_reference_radius: int = 96`
  - `foreground_candidate_count: int = 4`
  - `transition_alpha_min: int = 2`
  - `transition_alpha_max: int = 253`
  - `preserve_foreground_luma: float = 0.85`
- Keep defaults aligned with current High Accuracy Graphic Blue usage; do not change user-approved key color/default preset unless needed to expose new controls.

Execution:
- Serial

Isolation:
- `keyer.py` settings only; no behavior change until helper is wired.

Acceptance:
- Existing calls to `KeySettings` remain compatible.
- `python -m py_compile keyer.py smoke_test.py` passes.

Status:
- Complete


#### P1.2 - Add transition/fringe region helpers
- Add helpers in `keyer.py`; reuse/extend any existing `_compute_key_spill_strength()` helper instead of creating duplicate names:

```python
def _build_transition_repair_mask(
    alpha_u8,
    edge_mask,
    fringe_mask,
    spill_strength,
    background_mask,
    keep_mask,
    remove_mask,
    foreground_core_mask,
    settings,
): ...
def _build_foreground_core_mask(alpha_u8, background_mask, probability, fringe_mask, keep_mask, remove_mask, settings): ...
```

- Transition mask logic:
  - base live mask is `alpha > 0`, not known background, not manual remove, and not opaque protected core,
  - eligible if semi alpha in `transition_alpha_min..transition_alpha_max`, or live edge/fringe pixel, or live spill pixel above `transition_spill_threshold`,
  - final mask must exclude protected opaque core unless also semi/fringe.
- Foreground core logic:
  - `alpha >= 250`,
  - not background,
  - low screen probability (`<= 48` or settings-derived equivalent),
  - low fringe (`fringe <= 24`).
- Do not use these masks to reduce alpha.
- Helper signatures must carry the protection inputs they need: `background_mask`, `keep_mask`, `remove_mask`, and foreground-core/protected-core mask. Do not rely on prose-only protections.

Acceptance:
- Unit/smoke helper coverage verifies transition mask catches anti-aliased edge pixels and ignores opaque core/background.
- Foreground core mask finds red/white/black solid core on synthetic graphic fixtures.
- Manual keep/remove behavior is covered: keep protects core from aggressive RGB pull; remove remains alpha/RGB zero unless keep overrides.
- Tests prove `_build_transition_repair_mask()` excludes known background, manual remove, and protected opaque core while still including semi-transparent transition/fringe pixels.

Status:
- Complete


---






Current:
- No
### Phase 2 - Foreground reference and alpha recovery

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `keyer.py` transition repair internals and smoke fixtures. No UI. Preserve existing tiled/crop contracts.

Status:
- Complete

Progress:
- 2026-05-18: Added radius-valid foreground reference maps, deterministic tile-local fallback, and global transition alpha recovery with source-alpha/manual-mask clamps before tiled rendering.


#### P2.1 - Build radius-aware foreground reference
- Reuse existing nearest-inner mechanisms where possible, but decouple v7 reference availability from old `inner_color_pull` / `edge_color_repair` gates. If `transition_unmix` is enabled, reference maps must be available even when legacy color-pull sliders are low.
- Extend or wrap global labels from `_build_nearest_inner_label_map()` / `_nearest_inner_rgb_for_slice()` to also provide a radius check:
  - store a compact clipped distance map (`uint16` or similar) when under cap, or
  - compute radius-limited tile-local references for transition regions when global distance would exceed memory caps.
- Tile-local fallback from `_build_tile_local_nearest_inner_rgb()` must reject references farther than `foreground_reference_radius`.
- Alpha recovery must have a pre-render reference strategy:
  - preferred: build a compact global radius-valid foreground reference/distance map under existing caps,
  - fallback: run a deterministic pre-render tiled/striped alpha-recovery pass that computes tile-local references with sufficient overlap and writes only to the global `matte.alpha`,
  - if neither path can produce radius-valid references within caps, skip alpha recovery for those pixels and still run RGB-only transition repair later.
- For v7, the reference is not only color-pull; it is used to solve alpha.
- Honor `foreground_reference_radius`; ensure `_tile_extra_overlap()` includes this radius when transition unmix is enabled and global labels are unavailable.
- Return/consume:

```python
foreground_ref_rgb: HxWx3 uint8
foreground_ref_valid: HxW bool
foreground_ref_distance: HxW clipped distance or equivalent validity gate
```

- First implementation may use one nearest reference; keep `foreground_candidate_count` as a future-compatible setting unless multi-candidate selection is cheap and testable.

Acceptance:
- Full render, tiled render, and crop render have matching alpha/RGBA within existing tile tolerance.
- No seam appears when transition pixels depend on tile-local foreground references.
- Large uniform gaps do not borrow unrelated far foreground references; far references are invalidated by radius.
- When no valid foreground reference exists, alpha recovery skips deterministically rather than using a far/unbounded color.

Status:
- Complete


#### P2.2 - Implement global alpha solve anti-erosion
- Add `_recover_transition_alpha_global(...)` or equivalent in global matte construction, not inside `_process_color_tile()`.
- Reason: `_render_tiled_rgba()` writes `rgba[:, :, 3]` from `matte.alpha` before tile color processing, so alpha recovery must update `matte.alpha` before tiled rendering. Tile color helpers must not secretly return a new alpha that the output ignores.
- This helper may process the source in bounded stripes/tiles, but its output is the global uint8 `matte.alpha` before `_render_tiled_rgba()` starts. It must not allocate full-image float32 RGB.
- In transition pixels with valid foreground reference, work in linear RGB:

```python
v = F_ref - B
alpha_solved = dot(I - B, v) / dot(v, v)
alpha_solved = clamp(alpha_solved, 0, 1)
I_recon = alpha_solved * F_ref + (1 - alpha_solved) * B
err = norm(I - I_recon)
```

- Plausibility rule:

```python
if err < settings.transition_reconstruction_error:
    recovered = alpha_current + settings.alpha_recover_strength * max(alpha_solved - alpha_current, 0)
    alpha_final = max(alpha_current, recovered)
```

- Never reduce alpha in this phase.
- Do not resurrect `alpha == 0` pixels except optionally edge-gated in a later explicit subphase; default v7 leaves fully transparent pixels transparent.
- Reapply known-background/manual-remove clamps after recovery.
- Reapply manual keep behavior without exceeding source-alpha cap.
- Reapply original source alpha cap last:
  - recovered alpha `<= original_alpha * 255`,
  - source alpha 0 keeps alpha 0 and RGB 0.

Acceptance:
- On red/white/black anti-aliased fixtures, edge alpha is `>=` baseline and detail recall does not decrease.
- Confident background alpha remains 0.
- `alpha == 0` RGB remains 0 after final render.
- Source-alpha semi-transparent and fully transparent regression tests pass.

Status:
- Complete


---






Current:
- No
### Phase 3 - Linear RGB transition unmix and key-vector despill

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `_process_color_tile()` transition repair path in `keyer.py`; all retained behavior must be classical/no-AI. Do not preserve AI/hybrid-specific branches.

Status:
- Complete

Progress:
- 2026-05-18: Added linear RGB transition unmix/color repair, key-vector despill, luma-preserving foreground-reference pull, final-alpha tile color wiring, and strict transition RGB smoke gates.


#### P3.1 - Add `_repair_transition_unmix()`
- Add main helper in `keyer.py`:

```python
def _repair_transition_unmix(
    rgb_u8: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_color: tuple[int, int, int],
    screen_tile: np.ndarray | None,
    nearest_fg_rgb: np.ndarray | None,
    nearest_fg_valid: np.ndarray | None,
    settings: KeySettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return repaired_rgb_u8, repair_mask_u8. Alpha has already been recovered globally."""
```

- Use constant sampled key color first when `screen_tile is None`; use screen tile/plate only when already available and stable.
- Convert `I`, `B`, `F_ref` to linear RGB float32.
- Use the already recovered/capped global alpha from Phase 2; do not modify alpha in this tile helper.
- Reconstruct foreground color:

```python
F_est = (I - (1 - alpha) * B) / max(alpha, eps)
```

- Clamp and guard all NaN/Inf.

Acceptance:
- Helper returns same shapes/dtypes as input.
- If no valid foreground reference exists, helper safely returns original RGB and zero repair mask.
- Does not retain full-image float32 RGB outside tile scope.

Status:
- Complete


#### P3.2 - Add key-vector spill removal and luma-preserving chroma pull
- Implement vector spill removal, not single-channel global clamp:

```python
key_luma = sum(key_linear * luma_w)
key_vec = normalize(key_linear - key_luma)
pix_luma = sum(rgb_linear * luma_w)
pix_chroma = rgb_linear - pix_luma
spill = max(dot(pix_chroma, key_vec), 0)
out = rgb_linear - key_vec * spill * settings.key_vector_despill
```

- Add luma helper/matcher if existing `_protect_luminance()` cannot be reused directly.
- Pull repaired transition chroma toward foreground reference:

```python
pull = transition_mask * spill_strength * settings.foreground_reference_pull
F_ref_luma_matched = match_luma(F_ref, luma(F_clean))
F_final = lerp(F_clean, F_ref_luma_matched, pull)
```

- Preserve foreground luma with `preserve_foreground_luma`.
- Do not apply to opaque protected core except where explicitly in transition/fringe mask.

Acceptance:
- Transition key residual decreases on black/white/gray/checker composites.
- Foreground core RGB max delta stays `<= 3..5` for graphic fixtures.
- Black edges do not get lifted/yellowed; white edges stay white.

Status:
- Complete


#### P3.3 - Wire into classical graphic color path
- Integrate transition repair into `_process_color_tile()` after existing edge repair or as a replacement for the transition/fringe part when `settings.transition_unmix` is true.
- Keep behavior gated:
  - only Graphic/classical modes by default,
  - no alpha reduction,
  - no tile-side alpha mutation,
  - no global RGB grading,
  - no changes for `alpha == 0` pixels except RGB zeroing.
- Ensure `despill_mask`/debug repair mask reflects transition repair where possible.

Acceptance:
- Existing v5/v6 tests still pass.
- Hard-edge pixels and opaque foreground core are unchanged within tolerance.
- Preview/export parity remains intact.
- `rgba[:, :, 3]` always comes from final global `matte.alpha`; tile color repair only writes RGB and repair mask.

Status:
- Complete


---






Current:
- No
### Phase 4 - Diagnostics and regression gates

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `smoke_test.py` and `.artifact/` diagnostics only. No algorithm changes except test-driven bug fixes.

Status:
- Complete

Progress:
- 2026-05-18: Promoted transition-unmix fixtures into strict before/after regression gates, added manual keep/remove and source-alpha assertions, retained Imported Matte coverage, and added `--write-transition-unmix-diagnostics` under `.artifact/transition-unmix-diagnostics/`.


#### P4.1 - Promote baseline fixtures into strict regressions
- Reuse the Phase 1 baseline fixtures and convert them into pass/fail gates after Phase 2/3 behavior is wired.
- Add manual-mask and source-alpha variants:
  - keep mask over transition/core,
  - remove mask over transition/background,
  - semi-transparent original source alpha,
  - fully transparent source pixels with nonzero RGB.

Acceptance:
- Baseline metrics remain available for before/after comparison.
- New strict gates cover manual masks and source-alpha cap.

Status:
- Complete


#### P4.2 - Add required assertions and diagnostics
- Metrics:
  - `key_residual_on_transition` before/after,
  - `alpha_detail_recall` before/after,
  - `foreground_core_rgb_delta`,
  - `transparent_rgb_residual_max`,
  - composite residual on black/white/gray/checker.
- Assertions:
  - `alpha_out >= alpha_baseline` in transition/detail region,
  - foreground core delta `<= 3..5`,
  - background alpha remains 0,
  - key residual decreases after transition repair,
  - 1px details recall does not decrease,
  - `rgba[alpha == 0, :3].max() == 0`.
  - source-alpha cap: recovered alpha never exceeds original source alpha,
  - manual keep protects foreground color/core; manual remove remains alpha 0/RGB 0 where keep is absent,
  - Imported Matte behavior, if retained under no-AI naming, is not regressed by classical v7 changes.
- Optional diagnostics under `.artifact/transition-unmix-diagnostics/`:
  - source,
  - baseline alpha/RGB,
  - transition mask,
  - foreground reference validity,
  - alpha recovered,
  - repaired RGB,
  - composites,
  - metrics JSON.

Acceptance:
- `python smoke_test.py` passes with new tests.
- Optional diagnostics command/flag, if added, writes only under `.artifact/`.

Status:
- Complete


---






Current:
- No
### Phase 5 - UI controls and defaults

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `app.py` inspector controls and UI smoke probe. No AI UI changes.

Status:
- Complete

Progress:
- 2026-05-18: Added no-AI transition-unmix controls under Spill Cleanup, wired them into `KeySettings`, presets/defaults, tooltips, and headless UI probe coverage.


#### P5.1 - Add transition repair controls
- Add controls under `Spill Cleanup` or an `Advanced Graphic` subsection:
  - `Transition Unmix` ON/OFF,
  - `Alpha Recover` `0.0..1.0`, default `0.85`,
  - `Key Vector Despill` `0.0..1.0`, default `0.75`,
  - `FG Color Pull` `0.0..1.0`, default `0.65`.
- Preserve minimal UI style and no zoom reset on slider changes.
- Keep plus/minus slider stepping and reset behavior consistent with existing `SliderRow` patterns.
- Update `current_settings()`, preset/reset/default handling, tooltips, and UI probe expectations for every new setting.
- Do not place controls in AI section; v7 is classical/non-AI.

Acceptance:
- UI controls update `KeySettings` and schedule preview without resetting zoom/pan.
- Headless UI probe covers controls exist and default values match settings.
- Reset/preset paths restore the approved High Accuracy Graphic defaults plus v7 transition defaults.

Status:
- Complete


---






Current:
- No
### Phase 6 - Classical GPU runtime and kernels

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `gpu_runtime.py`, new `gpu_accel.py`/similar, optional dispatch hooks in `keyer.py`, and GPU parity tests. CPU path remains the reference.

Status:
- Complete

Progress:
- 2026-05-18: Added lazy `gpu_accel.py` torch/CUDA tensor backend API, optional transition-repair GPU dispatch, GPU acceleration UI controls/status, fallback/import-fence coverage, CUDA parity tests, and `.artifact/gpu-benchmarks/` benchmark reporting. Shipped only the parity-tested transition color-tile repair kernel; screen-probability, smoothstep, and preview-composite probes remain unshipped benchmark-only paths.


#### P6.1 - Define no-AI GPU backend API
- Add a small backend module such as `gpu_accel.py` with lazy imports only inside functions.
- First backend candidate is torch CUDA as a numerical runtime because it is already proven on RTX 5060 Ti; do not import torch at default startup.
- API shape:

```python
def is_available() -> dict: ...
def process_color_tile_gpu(rgb_tile, alpha_tile, screen_tile, settings, ...) -> dict: ...
def process_preview_gpu(rgb_u8, settings, masks...) -> dict: ...
```

- Return structured fallback reasons; never crash UI/export.

Acceptance:
- Importing app/keyer/gpu_accel does not import torch/CUDA.
- CPU fallback works when torch/CUDA unavailable.
- No AI/model package is imported or referenced.

Status:
- Complete


#### P6.2 - Benchmark and ship only worthwhile kernels
- Benchmark CPU vs GPU under `.artifact/gpu-benchmarks/`:
  - screen distance/probability arithmetic,
  - alpha matte gamma/smoothstep arithmetic,
  - linear RGB transition unmix/despill/color repair per tile,
  - preview composition if useful.
- Measure transfer cost separately.
- Ship only kernels that beat CPU for representative large preview/export tiles; keep CPU for connected components/branchy morphology unless a GPU path is clearly safe.

Acceptance:
- Benchmark artifacts stay under `.artifact/` and are not committed.
- First shipped GPU path has CPU/GPU parity within tolerance and a measured speed win or documented fallback rationale.

Status:
- Complete


#### P6.3 - Add GPU UI controls/status
- Add no-AI controls/status:
  - GPU Acceleration: Auto / Off / Force GPU,
  - GPU Status,
  - backend/fallback message.
- Preserve viewer-first UI behavior and preview cancellation.

Acceptance:
- User can tell whether GPU is being used.
- Auto falls back cleanly; Force GPU reports clear errors.
- No AI wording appears.

Status:
- Complete


---






Current:
- No
### Phase 7 - No-AI packaging

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own specs, requirements, splash, docs. No algorithm changes except packaging fixes.

Status:
- Complete

Progress:
- 2026-05-18: Phase 7 completed no-AI packaging cleanup. Kept public outputs to `ImgKey.exe` and `ImgKey-GPU.exe`, left the GPU requirement torch-only, tightened default/GPU PyInstaller excludes plus static smoke guards, preserved GPU splash/progress and CUDA probe wiring, and refreshed packaging/build docs.


#### P7.1 - Keep only two public build flavors
- Keep:
  1. `ImgKey.exe`: classical CPU/default, no torch/CUDA/AI.
  2. `ImgKey-GPU.exe`: classical GPU, CUDA tensor runtime only, no AI/model stack.
- Remove `ImgKey-GPU-BiRefNet.exe` build path.
- Update `requirements-gpu-runtime-cu128.txt` to minimal GPU numerical runtime, preferably `torch>=2.7` only unless `torchvision` is justified by build evidence.
- `ImgKey-GPU.spec` must exclude all AI/model packages.

Acceptance:
- Default EXE stays lightweight relative to GPU build.
- GPU EXE starts with splash/progress and probes CUDA.
- No model files are bundled.

Status:
- Complete


---






Current:
- No
### Phase 8 - Full verification, build, and phase commit hygiene

Category:
- Review-heavy

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own final verification/build only. Do not introduce new feature work unless fixing a verification failure.

Status:
- Complete

Progress:
- 2026-05-18: Completed final no-AI classical GPU v7 verification, diagnostics, default/GPU PyInstaller builds, built-EXE probes, archive exclusion checks, EXE size/SHA256 capture, and phase commit hygiene.


#### P8.1 - Verification floor
- Run:

```powershell
python smoke_test.py
python smoke_test.py --gpu-parity
python -m gpu_runtime --probe --json
python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py
python -c "import app, keyer; print('import ok')"
python -c "import sys, app, keyer, gpu_accel, gpu_runtime, screen_analysis; blocked={'torch','torchvision','transformers','timm','kornia','einops','accelerate','huggingface_hub','safetensors','skimage','onnxruntime','onnxruntime_gpu','pymatting','scipy','numba'}; loaded=sorted(m for m in blocked if m in sys.modules); assert not loaded, f'heavy modules imported at default startup: {loaded}'; print('default dependency fence ok')"
python -c "from pathlib import Path; roots=[Path(p) for p in ['app.py','keyer.py','smoke_test.py','README.md','AGENTS.md','gpu_accel.py','gpu_runtime.py','screen_analysis.py','ImgKey.spec','ImgKey-GPU.spec','requirements.txt','requirements-gpu-runtime-cu128.txt']]; roots += list(Path('docs').glob('**/*')) if Path('docs').exists() else []; forbidden=['BiRefNet','CorridorKey','Matting Anything','SAM','U2Net','MODNet','ViTMatte','Hugging Face','transformers','AI Hint','Hybrid BiRefNet']; hits=[]; [hits.append((str(p),s)) for p in roots if p.is_file() for s in forbidden if s in p.read_text(encoding='utf-8', errors='ignore')]; assert not hits, hits; print('public no-AI source guard ok')"
python -c "from pathlib import Path; forbidden=[Path('ai_worker.py'),Path('ai_assist.py'),Path('ai_backends'),Path('ImgKey-GPU-BiRefNet.spec'),Path('requirements-gpu-birefnet-cu128.txt')]; existing=[str(p) for p in forbidden if p.exists()]; assert not existing, existing; print('AI files removed ok')"
git diff --check
```

- If `gpu_accel.py` or a renamed classical trimap helper exists, include it in `py_compile` and the import-fence command.

- If diagnostics flag is added, also run it and confirm outputs stay under `.artifact/`.

Acceptance:
- All verification passes.
- No AI/heavy import regression and no public no-AI guard hits.
- No generated artifacts are staged.

Status:
- Complete


#### P8.2 - Build smoke
- Build default non-AI app:

```powershell
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

- Build no-AI GPU app after Phase 7:

```powershell
python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec
dist\ImgKey-GPU.exe --gpu-probe --json
```

Acceptance:
- Default `dist\ImgKey.exe` builds and starts/probes as expected.
- GPU build contains no AI/model stack and passes CUDA probe on RTX machine.

Status:
- Complete


#### P8.3 - Commit boundaries
- Treat each implementation phase as a clean commit boundary.
- Before each commit inspect:

```powershell
git status --short --branch
git diff
git log --oneline -10
```

- Stage only source/plan/docs/test files. Never stage `build/`, `dist/`, `.artifact/`, model snapshots, wheels, or caches.

Acceptance:
- Final branch is clean after source commits.
- Commit messages describe the phase, e.g.:
  - `Add transition unmix settings and masks`,
  - `Recover transition alpha without erosion`,
  - `Repair graphic transition RGB`,
  - `Add transition unmix diagnostics`,
  - `Add transition unmix UI controls`.

Status:
- Complete


---

## 5) Immediate next step

Phase 8 complete; planner can push the final phase commit after review.








Current:
- No
