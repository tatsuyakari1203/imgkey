# 04 - ImgKey v5 Classical Algorithm Upgrade

Date: 2026-05-17
Status: In progress
Owner: ImgKey Engine
Scope: Improve ImgKey's classical still-image chroma-key pipeline without AI, model runtimes, or new dependencies.

---

## 1) Goal

Raise output quality for large still-image chroma keying using classical algorithms only:

```text
Input RGB/RGBA
-> remove green/blue/custom key background
-> produce a smooth, non-jagged alpha matte
-> remove key-color fringe/halo at soft edges
-> preserve foreground core color
-> export straight-alpha PNG, with RGB=0 wherever alpha=0
```

No AI is in scope for this plan.

---

## 2) Current context

- Repo: `tatsuyakari1203/imgkey`, current local branch `main`, clean and tracking `origin/main`.
- Current release: `v1.0.0`, non-AI Windows EXE release is already published.
- Source of truth:
  - `keyer.py` — engine: sampling, probability, connected background, trimap/alpha, v4 fringe repair, tile export.
  - `app.py` — PySide6 UI, preview/export threads, debug views, full-crop preview.
  - `smoke_test.py` — synthetic tests and diagnostics.
  - `ImgKey.spec` — default non-AI onefile EXE packaging.
- `AGENTS.md`, `README.md`, `RELEASE.md`, `CHANGELOG.md` — current docs/context.
- GitNexus has no indexed repo for this workspace, so execution should rely on this plan, `AGENTS.md`, and focused source reads.
- Local git is expected after the public release push, but Phase 0 must verify it. If `.git`/`origin` is missing in an executor environment, execution may continue in backup-only mode with timestamped `.artifact/source-backup-*` snapshots, no commits, and explicit status in every phase completion report.

---

## 3) Hard constraints / stop conditions

- No AI changes: do not change `ai_assist.py` except to avoid import breakage.
- No new dependencies: keep default fence to `numpy`, `opencv-python`, `Pillow`, `PySide6`, and stdlib.
- Do not add PyTorch/CUDA/ONNX/BiRefNet/CorridorKey/model weights/PyMatting/SciPy/numba.
- Do not modify `ImgKey.spec` to bundle AI/deep runtimes.
- Source image stays `uint8`; masks stay `uint8`/bool; nearest-inner labels stay bounded `int32`.
- No full-image float32 RGB allocation for large images; float32 RGB work must be tile/ROI/crop only.
- Global semantic decisions — screen sampling, connected-background decisions, trimap, masks, fringe decisions where needed — happen before tiled export.
- Tiled export must remain seam-free: tiled vs reference max RGBA diff `<= 1` for covered test cases.
- For tile-local algorithms where exact tiled-vs-non-tiled equality is not meaningful, seam tests must compare boundary bands across multiple tile sizes and require max visible band delta `<= 1` for alpha and `<= 2` for RGB in non-fringe opaque areas, plus no checkerboard-visible halo discontinuity in diagnostics.
- `process_chroma_key()` compatibility must remain: returns `np.ndarray` RGBA `uint8`.
- Foreground core must not be color-graded: opaque foreground max RGB delta target `<= 3-5` levels.
- Transparent output RGB must remain zero: `rgba[alpha == 0, :3].max() == 0`.
- Stop and ask before any dependency, licensing, packaging, or AI/model decision that violates this section.

---

## 4) Target pipeline changes

1. **Baseline metrics first**: add hard fixtures and diagnostics before algorithm changes.
2. **Linear-light edge color repair**: move unmix/clamp/pull/luma protection from sRGB math to linear-light tile math.
3. **Guided alpha refinement**: optional grayscale guided filter applied only in edge/unknown bands, with exact core/background clamping.
4. **Tile-local screen model**: when no global `screen_map` exists, estimate local screen color from known background inside tile read regions.
5. **Crop-only full-resolution preview**: render only requested crop+margin instead of full-image RGBA then crop.
6. **Large-image nearest-inner fallback**: when global label map is skipped due to pixel cap, build tile-local labels in read tile with overlap.
7. **Docs/release prep**: update docs/context/tests, then optionally build and release a patch version after user approval.

---

## 5) Phases

### Phase 0 - Safety, branch, and baseline snapshot

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own repo safety only: git branch, `.artifact/` baseline outputs, plan status. Do not change algorithm yet.

Status:
- Completed

Current:
- No

#### P0.1 - Create implementation branch and backup
- Handle this plan file before enforcing clean git: either commit `.plan/imgkey-v5-classical-algorithm-upgrade.md` to `main` before execution, or create the feature branch and include it in the first planning/safety commit. The only acceptable pre-clean exception is this known plan file before it is intentionally tracked.
- Confirm `git status --short --branch` is otherwise clean.
- Confirm this is the real git checkout with `.git`, `origin`, and clean `main...origin/main`.
- If git/remotes are missing, switch to backup-only execution mode: create `.artifact/` backup snapshots before each deep phase, do not commit, and report the missing checkout as a release/merge blocker rather than blocking local algorithm work.
- Create a feature branch, recommended: `feature/classical-algorithm-upgrade`.
- Create `.artifact/source-backup-algo-v5-*` containing source files and `.plan/*.md`.
- Run current baseline commands:
  - `python smoke_test.py`
  - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
  - `python -c "import app, keyer; print('import ok')"`

Execution:
- Serial

Isolation:
- Git branch, `.artifact/`, plan status only.

Acceptance:
- Clean feature branch and backup exist; baseline verification passes before algorithm changes. If git is unavailable, backup-only mode is recorded and phase commits are skipped until a real checkout is restored.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-17: Confirmed `feature/classical-algorithm-upgrade` checkout with `origin`; created `.artifact/source-backup-algo-v5-20260517-173938`; baseline `smoke_test.py`, `py_compile`, and import checks passed.

---

### Phase 1 - Baseline fixtures and metrics

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `smoke_test.py` fixtures/metrics and optional `.artifact/algorithm-upgrade-baseline/` diagnostics. Do not alter core engine behavior.

Status:
- Completed

Current:
- No

#### P1.1 - Add hard fixtures and diagnostic metrics
- Add or extend synthetic fixtures:
  - `blue_gradient_screen_fixture`,
  - `green_gradient_screen_fixture`,
  - `same_key_foreground_core_fixture`,
  - `hair_lines_fixture`,
  - `semi_transparent_glass_fixture`,
  - `large_tile_gradient_fixture`,
  - `white_gray_black_composite_fixture`.
- Add metrics helpers:
  - `edge_key_residual`,
  - `opaque_foreground_max_delta`,
  - `alpha_soft_band_count`,
  - `transparent_rgb_zero`,
  - `tiled_vs_full_max_diff`,
  - `composite_black_white_gray_error`.
- New hardest checks may be diagnostics-first if current behavior cannot pass yet; clearly mark which become enforced in later phases.
- Save a machine-readable baseline metrics file, e.g. `.artifact/algorithm-upgrade-baseline/metrics.json`, and a human-readable summary before any algorithm change. Later phases must compare against this saved v4 baseline for “no worse than baseline” checks.
- Save reproducible baseline comparison artifacts for each enforced fixture, preferably `.npz` files containing alpha/RGBA/core masks plus hashes and metrics. Later phases that require exact/bounded diffs must compare against these saved artifacts, not memory or old code.
- Each fixture must include explicit masks/regions for known background, known foreground/core, soft edge, and optional expected foreground RGB where metrics need ground truth.
- Threshold policy:
  - Phase 1: diagnostics-only for new hardest fixtures unless current v4 already passes.
  - Phase 2+: enforce alpha unchanged for color-only changes, transparent RGB zero, foreground core max RGB delta `<= 5`, and fringe residual no worse than v4 baseline.
  - Phase 3+: enforce known BG/FG clamping and guided smoothness improvement on guided-specific fixtures.
  - Phase 4/6+: enforce seam metrics across at least two tile sizes.

Execution:
- Serial

Isolation:
- `smoke_test.py`, `.artifact/algorithm-upgrade-baseline/` only.

Acceptance:
- `python smoke_test.py` remains passing; baseline diagnostics, `metrics.json`, and per-fixture comparison artifacts/hashes are generated under `.artifact/algorithm-upgrade-baseline/`; no engine output changes yet.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-17: Added v5 diagnostic fixture set in `smoke_test.py` for blue/green gradient screens, same-key foreground core, hair-like thin lines, semi-transparent glass, large tile gradient runtime, and white/gray/black/checkerboard composite checks. Added metric helpers for edge residual, foreground delta, soft alpha band count, transparent RGB zero, tiled/full diff, and composite errors. Generated `.artifact/algorithm-upgrade-baseline/metrics.json`, per-fixture `.npz` artifacts, composite preview diagnostics, and `summary.md`; hardest new checks remain diagnostic-only for Phase 1.

---

### Phase 2 - Linear-light edge color repair

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `keyer.py` color repair helpers and `_process_color_tile()` path plus enforcing tests in `smoke_test.py`. Do not change alpha generation or UI controls in this phase.

Status:
- Planned

Current:
- Yes

#### P2.1 - Add sRGB/linear conversion helpers
- Add helper functions near existing color helpers:
  - `_srgb_to_linear_f32`,
  - `_linear_to_srgb_f32`,
  - `_srgb_u8_to_linear_f32`,
  - `_linear_f32_to_srgb_u8`.
- Keep helpers tile/ROI-friendly and avoid implicit full-image allocations from callers.

Execution:
- Serial

Isolation:
- `keyer.py` helper section only.

Acceptance:
- Helpers round-trip representative colors within expected `uint8` tolerance and do not affect existing outputs until wired.

Status:
- Planned

Current:
- Yes

#### P2.2 - Move color reconstruction math to linear light
- In `_process_color_tile()` convert `rgb_tile`, `screen_tile`/screen color, and nearest-inner RGB to linear before unmix/clamp/pull/luma protect.
- `_apply_vlahos_clamp()` and `_protect_luminance()` must receive linear RGB values and linear screen/key vectors; do not mix sRGB keys with linear pixels.
- Avoid unnecessary round-trip drift: pixels outside repair/fringe/despill masks should keep original `uint8` RGB unless the existing color pass already intentionally changes them.
- Keep alpha generation unchanged.
- Use Rec.709 luminance weights on linear RGB for luminance protection.
- Convert final RGB back to sRGB `uint8` only at tile output.
- Preserve transparent RGB zeroing.

Execution:
- Serial

Isolation:
- `_process_color_tile()`, `_protect_luminance()`, `_apply_vlahos_clamp()` or equivalent helpers.

Acceptance:
- Alpha max diff is `0` vs pre-phase behavior for baseline fixtures; transparent RGB remains zero; key-channel fringe excess is equal or better than v4; opaque foreground max RGB delta `<= 5`, target `<= 3`; tiled vs reference max diff `<= 1`.

Status:
- Planned

Current:
- No

---

### Phase 3 - Guided alpha refinement, edge-only

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own alpha-refine helpers/settings in `keyer.py` and tests in `smoke_test.py`. UI exposure is optional and deferred unless needed for verification.

Status:
- Planned

Current:
- No

#### P3.1 - Add optional guided alpha settings and helper
- Add compatible `KeySettings` fields:
  - `guided_alpha_refine: float = 0.0`,
  - `guided_radius: int = 8`,
  - `guided_eps: float = 1e-3`.
- Implement `_guided_filter_gray(guide, src, radius, eps)` using `cv2.boxFilter` with float32 grayscale/luma arrays.
- Default stays `0.0` at first to preserve existing behavior.
- Add `guided_max_pixels` or equivalent cap/fallback. If full-image guided filtering would exceed the cap, process only expanded edge-band ROI/stripes or skip guided refinement with deterministic unchanged output.

Execution:
- Serial

Isolation:
- `KeySettings`, guided helper only.

Acceptance:
- Existing tests pass unchanged with default `guided_alpha_refine=0.0`; no new dependencies.

Status:
- Planned

Current:
- No

#### P3.2 - Apply guided refinement only in edge/unknown band
- Add `_refine_alpha_guided()` after initial trimap alpha is built, preferably in `_build_global_matte` to avoid invasive `_build_alpha_from_trimap()` signature changes.
- Guide should be luma, preferably linear or consistent grayscale, not full-image float32 RGB.
- Large-image implementation must not allocate many full-frame float32 arrays without cap. Preferred order: edge-band bounding ROI with margin, then stripe fallback, then skip.
- Only blend refined alpha into edge/unknown mask.
- Clamp exact known regions after refinement:
  - known background remains `0`,
  - known foreground/core remains `255`,
  - connected/background policy remains authoritative.

Execution:
- Serial

Isolation:
- `keyer.py` global matte/alpha refine path and tests.

Acceptance:
- With guided off, outputs unchanged. With guided on in tests, soft-edge smoothness improves/increases without changing known BG/FG; tiled-vs-full alpha diff `<= 1` or exact for covered paths.

Status:
- Planned

Current:
- No

---

### Phase 4 - Tile-local screen model for gradient/shadow screens

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own local screen map helpers and tiled render data flow in `keyer.py`. Do not change probability model yet.

Status:
- Planned

Current:
- No

#### P4.1 - Add tile-local screen estimate fallback
- Reuse existing `local_screen_model` if already present and semantically equivalent; otherwise add `tile_local_screen_model: bool = True` with clear docs. Do not create duplicate/conflicting UI semantics.
- Implement `_estimate_screen_tile(rgb_tile, known_bg_tile, fallback_color, radius)` using normalized box filtering over known background pixels.
- `known_bg_tile` must come from connected/background-safe global matte masks, not raw high-probability pixels that may include protected foreground islands.
- In `_render_tiled_rgba()`, if `matte.screen_map is None` and local model is enabled, compute `screen_tile` from the read tile; otherwise use global screen color.
- Overlap/read margin must include the screen-estimation radius; write only core.

Execution:
- Serial

Isolation:
- `_estimate_screen_tile()`, `_render_tiled_rgba()`, `_process_color_tile()` inputs.

Acceptance:
- Gradient screen fixtures show lower edge residual than global screen fallback; no full-image float32 RGB allocation; boundary-band seam metrics pass across at least two tile sizes.

Status:
- Planned

Current:
- No

---

### Phase 5 - Crop-only full-resolution preview render

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `keyer.py` render crop support and minimal `app.py` preview job wiring. Preserve current UI behavior: no zoom reset, export progress/cancel remains intact.

Status:
- Planned

Current:
- No

#### P5.1 - Define and implement crop-render result contract in keyer.py
- Extend `_render_tiled_rgba()` with `render_crop: tuple[int,int,int,int] | None = None`, or add a wrapper if less invasive.
- When `render_crop` is set:
  - global matte still builds on the full image,
  - only tiles intersecting crop are color-rendered,
  - read region includes margin/overlap from all active local algorithms: edge radius, fringe band, guided radius, tile-local screen-estimation radius, tile-local nearest-inner radius after Phase 6, and tile overlap,
  - output array is crop-sized,
  - despill/fringe/debug masks are crop-sized or safely unavailable.
- Define crop result contract: when `settings.full_res_crop`/render crop is active, `KeyResult.rgba`, `alpha`, `despill_mask`, `fringe_mask`, optional debug RGB arrays, and `display_rgb` must all be crop-shaped and mutually aligned; metadata/status must still report original source size where UI needs it.

Execution:
- Serial

Isolation:
- `keyer.py` crop render path and crop-shape tests first.

Acceptance:
- Crop-only result matches crop from full render with max RGBA diff `<= 1`; all crop-shaped debug/view arrays align and do not crash view modes.

Status:
- Planned

Current:
- No

#### P5.2 - Wire Full Crop preview through app.py
- Update `app.py` full-crop preview job creation to request crop-only render while preserving v3 UX: no zoom reset, stable crop selection, progress/cancel behavior, and debug view fallback safety.

Execution:
- Serial

Isolation:
- `app.py` preview wiring and UI probes only.

Acceptance:
- Full Crop preview avoids full-image RGBA color pass; UI pan/zoom behavior remains stable; relevant UI probes pass.

Status:
- Planned

Current:
- No

---

### Phase 6 - Large-image nearest-inner fallback

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own nearest-inner fallback helpers and tiled color repair path in `keyer.py` plus tests.

Status:
- Planned

Current:
- No

#### P6.1 - Add tile-local nearest-inner pull when global labels are skipped
- Keep existing global label map for images below cap.
- When global nearest-inner labels are unavailable and `inner_color_pull > 0`, build tile-local inner labels inside read tile using `cv2.distanceTransformWithLabels`.
- Inner mask conditions should use high alpha, non-background, low fringe, and suitable probability thresholds.
- Define deterministic limits: local search uses read tile only, minimum inner pixels `>= 8`, max useful pull radius bounded by overlap/margin, and fallback to unmix+clamp if local labels are absent/too far.
- Read overlap must include edge radius, fringe band, guided radius, and local nearest-inner radius; write only tile core.

Execution:
- Serial

Isolation:
- `_build_nearest_inner_label_map()`, `_nearest_inner_rgb_for_slice()` or equivalent, `_render_tiled_rgba()` tile path.

Acceptance:
- Large synthetic edge repair has better residual than no-pull fallback; boundary-band seam tests pass across at least two tile sizes; crop-only render with tile-local nearest-inner enabled matches full-render crop within Phase 5 diff threshold; no full-image float32 RGB allocation.

Status:
- Planned

Current:
- No

---

### Phase 7 - Optional matte probability refinement research gate

Category:
- Review-heavy

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Investigation/prototype only unless metrics clearly justify implementation. Do not change tuned default probability behavior without review.

Status:
- Planned

Current:
- No

#### P7.1 - Decide whether to add local probability refinement
- Evaluate whether gradient/custom fixtures still fail after Phases 2-6.
- If needed, prototype a guarded/off-by-default probability improvement such as Lab/chroma distance or border-sample covariance/Mahalanobis blend in a branch or diagnostic path only.
- Keep connected-background policy as foreground-protection safety net.
- No production/default source change from this phase without explicit user approval after metrics show Phases 2-6 are insufficient.

Execution:
- Serial

Isolation:
- `keyer.py` probability helpers and diagnostics only, unless explicitly approved.

Acceptance:
- Either a measured, off-by-default improvement is proposed with tests, or the phase records that no probability change is recommended.

Status:
- Planned

Current:
- No

---

### Phase 8 - Documentation, release prep, and integration review

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own docs, final verification artifacts, packaging check, and release notes. Do not add new algorithm scope except fixing regressions.

Status:
- Planned

Current:
- No

#### P8.1 - Update docs/context and run full verification
- Update `README.md`, `AGENTS.md`, and `CHANGELOG.md` with:
  - linear-light repair,
  - guided alpha refine settings/default state,
  - tile-local screen model,
  - crop-only preview rendering,
  - tile-local nearest-inner fallback,
  - no-AI/no-new-dependency rule.
- `AGENTS.md` must explicitly add v5 rules for linear-light repair, guided-filter memory cap/fallback, tile-local screen/nearest-inner overlap rules, crop-render result-shape contract, and dependency-fence verification.
- Run:
  - `python smoke_test.py`
  - `python smoke_test.py --write-edge-repair-diagnostics`
  - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
  - `python -c "import app, keyer; print('import ok')"`
  - relevant UI probes if `app.py` preview path changed.
- After each implementation phase, also verify dependency fence with focused source/import checks: no new imports outside approved dependencies and no heavy optional modules imported by default.
- Build local EXE only if requested for validation or release prep:
  - `python -m PyInstaller --noconfirm --clean ImgKey.spec`

Execution:
- Serial

Isolation:
- Docs/tests/artifacts/package output only.

Acceptance:
- Full verification passes, docs describe the final pipeline, dependency fence remains unchanged, and `git status` contains only intended source/docs changes.

Status:
- Planned

Current:
- No

---

## 6) Required verification after every implementation phase

```powershell
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py
python -c "import app, keyer; print('import ok')"
```

Dependency-fence check after every implementation phase:

```powershell
python -c "import sys, app, keyer; blocked={'torch','torchvision','transformers','onnxruntime','onnxruntime_gpu','pymatting','scipy','numba'}; loaded=sorted(m for m in blocked if m in sys.modules); assert not loaded, f'blocked optional/heavy modules imported: {loaded}'; print('dependency fence ok')"
```

Also inspect source diffs for new imports outside the approved dependency fence before each phase commit/report.

When changing edge repair:

```powershell
python smoke_test.py --write-edge-repair-diagnostics
```

When changing preview/UI path:

```powershell
python app.py
```

When preparing release/build:

```powershell
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

---

## 7) Commit and release discipline

- Execute on a feature branch.
- If git is unavailable in the executor environment, execute in backup-only mode with no phase commits and report that merge/release requires restoring the real checkout.
- Each completed implementation phase should end with:
  - passing verification,
  - plan progress update,
  - one clean phase commit, unless the phase made no source changes.
- Do not push a release tag automatically from this plan. After all phases pass, ask before creating a patch release, likely `v1.1.0` or `v1.0.1` depending on scope.

---

## 8) Immediate next step

Start Phase 0 with `worker`, then Phase 1 with `worker`, then route Phases 2-6 to `deep-worker` one phase at a time. The first quality-impacting phase is **Phase 2: Linear-light edge color repair**.
