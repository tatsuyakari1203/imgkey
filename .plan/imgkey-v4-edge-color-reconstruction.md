# 03 - ImgKey v4 Edge Color Reconstruction

Date: 2026-05-17
Status: Completed
Owner: ImgKey Engine/UI
Scope: Add production-style edge color repair/fringe decontamination to remove blue/green key-color halos while preserving the existing large-image workflow, v3 UI, and default non-AI packaging.

---

## 1) Goal

Improve output edge quality by replacing key-color contaminated fringe pixels with reconstructed foreground colors. The target is to eliminate visible blue/green color cast at soft edges using classical VFX/matting algorithms first, while leaving AI as an optional future path.

---

## 2) Research synthesis

- The real issue is not only alpha quality; it is **foreground RGB estimation**. A soft pixel follows the matting equation `I = alpha * F + (1-alpha) * B`. If PNG export keeps contaminated `I` as RGB, old blue/green backing color remains visible when composited over a new background.
- Vlahos/blue-screen matting addressed spill by constraining/clamping the backing-dominant channel, e.g. reducing blue when it exceeds plausible foreground relationships. This is fast but can be crude if applied globally.
- Alpha-gated despill improves this: process only semi-transparent edge pixels (`0 < alpha < 1`) so foreground interior colors are not damaged.
- Foreground estimation libraries/papers, including PyMatting foreground estimation and Fast Multi-Level Foreground Estimation, solve for clean foreground colors from image + alpha. This is conceptually best but heavier, so use it optionally/ROI-only, not as the default full-image path.
- Production VFX workflows commonly build a spill/fringe map, then do adaptive color correction with luminance preservation.
- AI approaches such as FBA/Context-aware matting/CorridorKey predict foreground RGB + alpha together and help with hair, translucency, or ambiguous foregrounds. They are not required for the current poster/graphic use case and should not be bundled into the default EXE.

---

## 3) Constraints / non-goals

- Do not add PyTorch/CUDA/model dependencies.
- Do not break current v3 UI defaults or Space-pan/zoom preservation.
- Keep source image as `uint8`; avoid full-image float32 RGB allocations outside bounded preview/tile operations.
- Edge repair must be alpha/edge-gated by default. Do not globally alter opaque foreground colors.
- Tiled export must stay seam-free: any nearest-inner color propagation must be computed from global masks or use sufficient overlap/core-write logic.
- Existing `process_chroma_key()` compatibility must continue to return RGBA.
- Stop and ask before adding AI foreground reconstruction or bundling third-party model weights.
- Dependency fence: implementation must use only current default dependencies (`numpy`, `opencv-python`, `Pillow`, `PySide6`, stdlib). No PyMatting/SciPy/numba/model dependency without explicit approval.
- v4 must extend the existing `_process_color_tile()` unmix/despill path rather than stack a second independent full despill pass. Avoid double-correction by making edge repair the single final color-reconstruction stage inside tile color processing.

---

## 4) Target algorithm

Add an `Edge Color Reconstruction Pro` stage after alpha/trimap generation and before final RGBA write.

```text
Input RGB + alpha + key color + edge/background masks
-> build alpha edge/fringe band
-> compute key-color spill/fringe strength
-> alpha-aware foreground unmix
-> alpha-gated Vlahos/AGED channel clamp
-> nearest-inner foreground color pull via distance-transform labels
-> luminance-preserving blend
-> optional edge-aware smoothing inside fringe band only
-> zero RGB where alpha == 0
```

Proposed controls:

```python
fringe_remove: float = 0.75          # strength of key-color removal / channel clamp
edge_color_repair: float = 0.65      # blend from original/unmixed to repaired foreground RGB
inner_color_pull: float = 0.45       # pull toward nearest clean foreground core color
luminance_protect: float = 0.80      # UI alias/extension of existing luminance_restore for repair luma preservation
fringe_band_radius: int = 3          # local expansion around semi-transparent edge
```

Detailed constants/default formulas:
- `alpha_min = 2/255`, `alpha_max = 253/255` for semi-transparent edge detection.
- Use the engine-detected/sampled `screen_color` from `KeyResult`/global matte, not only `settings.key_color`, so Auto/Pick modes repair against the actual plate color.
- Blue/green spill excess: `excess = key_channel - max(other_channels)` normalized by `max(key_channel, 1)` and clipped to `[0,1]`.
- Custom spill excess: project luminance-neutral RGB residual onto normalized key-color vector and clip to `[0,1]`.
- Final repair weight: `fringe_mask * edge_color_repair`, with channel clamp scaled by `fringe_remove` and nearest-inner pull scaled by `inner_color_pull`.
- Luminance protection should preserve original/repaired perceived luma within a bounded blend; do not introduce a separate unrelated slider if existing `luminance_restore` already covers this in UI.

Tile strategy:
- Compute global alpha, edge/fringe mask, and nearest-inner label/reference maps before tiling when memory allows using `uint8` masks and `int32` labels only.
- Use OpenCV distance transform with labels on the global inverse inner-foreground mask to map fringe pixels to nearest clean foreground pixels. Do not allocate full-image float32 RGB for labels/foreground.
- During tiled color processing, read global labels/masks for that tile+overlap and gather nearest inner RGB from original `uint8` source. Write only tile core.
- If a tile has no valid nearby inner foreground labels, fall back to unmix + channel clamp for that tile. No per-tile independent semantic decisions.

Debug contract:
- `fringe_mask` is mandatory in preview `KeyResult` as `uint8`.
- `foreground_rgb`/`repaired_edge` are optional and should not be allocated full-res during export unless explicitly needed for preview/debug.

Debug outputs:

```python
fringe_mask: np.ndarray | None       # uint8 0-255, visible contamination/repair weight
repaired_edge: np.ndarray | None     # RGB/RGBA preview of repaired edge contribution if practical
foreground_rgb: np.ndarray | None    # optional repaired foreground RGB for debug/export inspection
```

---

## 5) Phases

### Phase 0 - Safety and baseline

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `.artifact/` backup/baseline artifacts and this plan status only. Do not change algorithm/UI yet.

Status:
- Completed


#### P0.1 - Backup and verify current app
- Create `.artifact/source-backup-edge-v4-*` containing `app.py`, `keyer.py`, `smoke_test.py`, `ai_assist.py`, `README.md`, `AGENTS.md`, `ImgKey.spec`, and active plan files.
- Run current verification baseline:
  - `python smoke_test.py`
  - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
  - `python -c "import app, keyer; print('import ok')"`

Execution:
- Serial

Isolation:
- `.artifact/` and plan status only.

Acceptance:
- Backup exists and baseline passes before engine changes.

Progress:
- 2026-05-17: Created `.artifact/source-backup-edge-v4-20260517-031705` with source files and active `.plan/*.md` files; baseline verification passed (`python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, and `python -c "import app, keyer; print('import ok')"`).

Status:
- Completed



---




Current:
- No
### Phase 1 - Engine edge color reconstruction

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `keyer.py` and smoke tests. Do not redesign UI in this phase except keeping existing app imports working.

Progress:
- 2026-05-17: Implemented v4 edge color reconstruction in `keyer.py`: compatible settings/result fields, global fringe mask, alpha-aware unmix, Vlahos/custom clamp, global nearest-inner color pull via OpenCV distance labels with a large-image memory cap/fallback, bounded luminance protection, tiled export integration, and transparent RGB zeroing. Added enforcing smoke coverage in `smoke_test.py` for blue/green fringe removal, interior preservation, nearest-inner pull, luma bounds, tile seam consistency, debug `fringe_mask`, despill/decontaminate sensitivity, compatibility wrapper, memory-cap fallback, and no heavy optional imports.

Status:
- Completed


#### P1.1 - Extend settings/result contract compatibly
- Add new `KeySettings` fields with defaults matching section 4.
- Add `KeyResult.fringe_mask`, `KeyResult.repaired_edge`, and/or `KeyResult.foreground_rgb` if practical.
- Preserve positional compatibility for existing first fields and `process_chroma_key()` return type.
- Decide/implement `luminance_protect` as an app/UI alias to `luminance_restore` unless a separate engine field is demonstrably needed.

Execution:
- Serial

Isolation:
- `keyer.py` dataclasses/API only.

Acceptance:
- Existing callers and smoke tests still pass with defaults; `process_chroma_key()` still returns plain RGBA; no optional/heavy dependency is imported.

Progress:
- 2026-05-17: Added `fringe_remove`, `edge_color_repair`, `inner_color_pull`, `fringe_band_radius`, and optional `luminance_protect` alias/override while preserving existing field order; `process_key_image()` now returns a populated `fringe_mask` and `process_chroma_key()` still returns RGBA.

Status:
- Completed


#### P1.2 - Implement fringe/spill mask
- Build semi-transparent/near-edge mask from alpha:
  - core edge: `alpha_min < alpha < alpha_max`,
  - optional dilation by `fringe_band_radius`, clipped away from fully transparent background unless needed for repair.
- Compute key-color spill strength:
  - green/blue: channel excess over max of other channels,
  - custom: projection onto normalized key-color vector after luminance-neutralization.
- Use the actual sampled `screen_color` from global matte/preview result.
- Combine alpha edge weight + spill strength into `fringe_mask`.

Execution:
- Serial

Isolation:
- `keyer.py` edge mask helpers and tests.

Acceptance:
- Synthetic blue/green fringe fixture produces high fringe mask on contaminated edge and low mask in opaque interior; alpha output is unchanged by mask creation.

Progress:
- 2026-05-17: Added global uint8 fringe map generation from alpha edge band, sampled screen color, green/blue excess, custom key-vector projection, probability, and edge weights.

Status:
- Completed


#### P1.3 - Add alpha-aware unmix and AGED/Vlahos channel clamp
- Improve/encapsulate unmix: `F = (I - (1-alpha)*K) / max(alpha, eps)` only in bounded edge/fringe regions.
- Add alpha-gated channel clamp:
  - blue screen: reduce blue toward `max(red, green)`,
  - green screen: reduce green toward `max(red, blue)`,
  - custom: reduce projection along key vector.
- Blend by `fringe_remove` and `fringe_mask`; never modify alpha==0 RGB except final zeroing and avoid damaging alpha==1 interior unless explicitly inside fringe band.
- Integrate into current `_process_color_tile()` so old unmix/despill/decontaminate and new repair do not double-correct the same edge pixels.

Execution:
- Serial

Isolation:
- `keyer.py` color repair helpers and tests.

Acceptance:
- Edge pixels reduce key-channel excess by at least 60% on synthetic fringe tests; opaque foreground interior RGB max delta stays <= 3 levels; alpha max diff is 0 except pre-existing matte operations.

Progress:
- 2026-05-17: Reworked `_process_color_tile()` into a single repair stage using fringe-gated unmix and Vlahos/custom clamp; existing `despill` and `decontaminate` controls now scale repair strength, and the old despill mask remains debug/output signal without a second color-correction pass.

Status:
- Completed


#### P1.4 - Add nearest-inner color pull and luminance protection
- Build clean inner foreground mask from high alpha and low spill, e.g. `alpha >= 0.98` and not background.
- Use OpenCV distance transform with labels or equivalent to map each fringe pixel to the nearest inner foreground pixel.
- Blend repaired edge RGB toward nearest inner RGB by `inner_color_pull * edge_color_repair * fringe_weight`.
- Preserve perceived luminance using `luminance_protect`; avoid over-brightening by clamping and safe eps.
- For tiles, compute required nearest-inner references globally or ensure overlap/core-write is sufficient and deterministic.
- Implement the global label-map strategy described in section 4; tile overlap is still used for color/refine context but must not change nearest-inner decisions.

Execution:
- Serial

Isolation:
- `keyer.py` repair stage and tests.

Acceptance:
- Synthetic poster/text edge fixture shows reduced blue halo on white/gray composite without flattening opaque interior texture; luma of repaired edge stays within 15% of pre-repair luma after `luminance_protect` unless clipped.

Progress:
- 2026-05-17: Added global nearest-inner label map using `cv2.distanceTransformWithLabels`; tiles gather nearest original uint8 RGB lazily and fall back to unmix+clamp if labels are unavailable or the image exceeds the memory cap. Luminance protection uses `luminance_protect` when provided, otherwise `luminance_restore`.

Status:
- Completed


#### P1.5 - Integrate with preview/export/tiled path
- Run repair in preview and full-res export.
- Ensure tiled export has no seams after repair.
- Keep progress/cancel hooks working.
- Keep transparent background RGB zeroed.
- Own required changes to `_process_color_tile()`, `_render_tiled_rgba()`, `_GlobalMatte`, and tile signatures/data flow.

Execution:
- Serial

Isolation:
- `keyer.py` pipeline/tile integration and tests.

Acceptance:
- Tile seam regression with small tile size and repair enabled has tiled-vs-reference RGBA max diff <= 1; output PNG remains straight alpha; transparent RGB remains zero.

Progress:
- 2026-05-17: Threaded fringe masks and nearest-inner references through `_GlobalMatte`, `_render_tiled_rgba()`, and tile processing; preview and export share the same path.

Status:
- Completed


#### P1.6 - Add enforcing edge-quality tests
- Move core edge-quality verification into Phase 1 rather than waiting for final packaging.
- Add synthetic tests for:
  - blue fringe removal,
  - green fringe removal,
  - opaque interior preservation,
  - nearest-inner color pull,
  - luminance protection avoiding dark/gray edges,
  - tiled repair seam consistency with small tile size,
  - debug `fringe_mask` presence/range,
  - `process_chroma_key()` compatibility and transparent RGB zeroing,
  - no heavy/optional modules imported in default path.

Execution:
- Serial

Isolation:
- `smoke_test.py`, `keyer.py` test helpers only.

Acceptance:
- Numeric tests enforce key-channel excess reduction, alpha stability, interior preservation, luma bounds, and tile consistency.

Progress:
- 2026-05-17: `run_v4_edge_repair_tests()` now enforces all Phase 1 edge-quality contracts, existing cleanup-control sensitivity, memory-cap fallback, and the existing v2 smoke suite.

Status:
- Completed


---




Current:
- No
### Phase 2 - UI controls and debug views

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `app.py` wiring for new settings/debug outputs only. Do not alter broader UI layout/style.

Progress:
- 2026-05-17: Completed Phase 2 UI-only wiring in `app.py` and added `.artifact/ui-v3-verification/v4_edge_repair_ui_probe.py`. Verification passed: `python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, `python -c "import app, keyer; print('import ok')"`, `python .artifact/ui-v3-verification/phase2_defaults_controls_probe.py`, `python .artifact/ui-v3-verification/phase4_final_ui_probe.py`, and `python .artifact/ui-v3-verification/v4_edge_repair_ui_probe.py`.

Status:
- Completed



#### P2.1 - Add edge repair controls
- In `Spill Cleanup` or `Edges`, add concise controls:
  - `Fringe Remove`,
  - `Edge Color Repair`,
  - `Inner Color Pull`,
  - `Luminance Protect`,
  - optional advanced `Fringe Band` if space allows.
- Defaults should match section 4 and reset correctly.
- `current_settings()` must pass the values into `KeySettings`.
- Update `APP_DEFAULT_SETTINGS`, `current_settings()`, High Accuracy/Fast/Clean presets, reset targets, and tooltips consistently. If `Luminance Protect` aliases existing `Luminance Restore`, do not duplicate confusing controls.

Execution:
- Serial

Isolation:
- `app.py` inspector/defaults/preset wiring.

Acceptance:
- Fresh launch shows tuned repair controls; changing them updates preview without zoom reset.

Progress:
- 2026-05-17: Added `Fringe Remove` (0.75), `Edge Color Repair` (0.65), `Inner Color Pull` (0.45), and `Fringe Band` (3) to Spill Cleanup. `Luminance Restore` is reused and tooltiped as luminance protection instead of adding a duplicate control. `APP_DEFAULT_SETTINGS`, `current_settings()`, High Accuracy/Fast/Clean presets, reset targets, and control tooltips now include the v4 repair settings.

Status:
- Completed


#### P2.2 - Add debug views
- Add view modes for `Fringe Mask` and, if available, `Foreground RGB` or `Repaired Edge`.
- Status/tooltips should explain that repair modifies RGB at soft edge, not alpha.
- Update `VIEW_MODES` and fallback behavior so missing optional RGB debug arrays do not crash the UI.
- Add/update offscreen UI probe assertions for new defaults/reset/debug view presence.

Execution:
- Serial

Isolation:
- `ImageCanvas`/view mode wiring only.

Acceptance:
- User can inspect where repair is applied and compare with Result.

Progress:
- 2026-05-17: Added `Fringe Mask` and `Foreground RGB` debug views, with safe fallbacks when optional RGB debug arrays are missing. The v4 offscreen UI probe asserts defaults/reset/preset wiring, debug view presence/fallbacks, and preview refresh without zoom reset after an edge repair control change.

Status:
- Completed


---




Current:
- No
### Phase 3 - Verification, docs, packaging

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `smoke_test.py`, docs/context updates, `.artifact/` verification outputs, `ImgKey.spec` rebuild output. Do not add new feature scope except fixing regressions.

Progress:
- 2026-05-17: Completed Phase 3 verification/docs/packaging. Updated `README.md` and `AGENTS.md` for v4 edge repair controls, RGB-only algorithm behavior, dependency fence, global/tile memory rules, and diagnostics. Verification passed: `python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, `python -c "import app, keyer; print('import ok')"`, all existing `.artifact/ui-v3-verification/*.py` probes including `v4_edge_repair_ui_probe.py`, and an explicit heavy optional import check. Built `dist\ImgKey.exe` with `ImgKey.spec`; the first rebuild was blocked by a running `ImgKey.exe`, stopped only that process, rebuilt successfully, and verified the EXE starts.

Status:
- Completed


#### P3.1 - Expand edge quality tests
- Confirm Phase 1 enforcing tests remain passing and add optional visual diagnostics: write before/after composites over black/white/gray/checkerboard to `.artifact/edge-repair-verification/`.

Execution:
- Serial

Isolation:
- `smoke_test.py`, `.artifact/` only.

Acceptance:
- Numeric tests remain passing and visual diagnostics are generated under ignored artifacts.

Progress:
- 2026-05-17: Added optional `python smoke_test.py --write-edge-repair-diagnostics` output and generated `.artifact/edge-repair-verification/` before/after composites over black, white, gray, and checkerboard plus `metrics.txt`. Phase 1 enforcing smoke tests passed during diagnostic generation.

Status:
- Completed


#### P3.2 - Update docs/context and rebuild EXE
- Update `README.md` and `AGENTS.md` with the new edge repair controls and algorithm notes.
- `AGENTS.md` must reflect any new dependency/default/memory/tile rule before this phase completes.
- Run verification:
  - `python smoke_test.py`
  - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
  - `python -c "import app, keyer; print('import ok')"`
  - existing UI probes if still applicable.
- Build default non-AI EXE with `ImgKey.spec`.
- Verify `dist\ImgKey.exe` starts.

Execution:
- Serial

Isolation:
- Docs/context/package output only.

Acceptance:
- Verification passes, EXE rebuilt, no AI/model dependencies added.

Progress:
- 2026-05-17: Documentation/context updated, all verification commands and UI probes passed, optional heavy import check passed, default non-AI PyInstaller build completed, and `dist\ImgKey.exe` start probe passed. Rebuilt EXE: `D:\keyphong\dist\ImgKey.exe` (99,817,243 bytes / 95.19 MiB).

Status:
- Completed


---

## 6) Verification commands

```powershell
cd D:\keyphong
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py
python -c "import app, keyer; print('import ok')"
python -m PyInstaller --noconfirm --clean ImgKey.spec
$p = Start-Process -FilePath ".\dist\ImgKey.exe" -PassThru; Start-Sleep -Seconds 6; if ($p.HasExited) { exit 1 } else { Stop-Process -Id $p.Id }
```

---

## 7) Immediate next step

Plan complete. No further phase remains in `.plan/imgkey-v4-edge-color-reconstruction.md`.
