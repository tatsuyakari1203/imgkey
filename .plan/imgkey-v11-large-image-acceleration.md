# 11 - ImgKey Large-Image Acceleration

Date: 2026-05-21
Status: In progress
Owner: ImgKey Large-Image Pipeline + GUI Responsiveness
Scope: Make 25MP+ blue/green-key workflows feel materially faster by reducing CPU-global recomputation, improving preview responsiveness, and extending D3D12 acceleration where whole-pipeline profiling shows ROI.

---

## 1) Goal

Turn the D3D12 refactor into visible large-image speed gains in real GUI use, not just isolated kernel benchmarks.

Target outcome on the provided 6688x3776 blue-key PNGs:

- Keep D3D12 color-render parity at max RGB diff `<= 1` and alpha diff `0` for RGB-only paths.
- Reduce repeated preview/export recomputation by caching full-resolution global matte/transition data when source, masks, imported matte, and matte-affecting settings are unchanged.
- Make interactive preview responsive during slider/crop/viewport work with progressive/ROI behavior and stronger stale-job cancellation.
- Improve full export speed beyond the current ~2.4-2.7x CPU-vs-D3D12 end-to-end speedup by attacking the new CPU bottlenecks.
- Preserve the one-file `ImgKey.exe` CPU+D3D12 packaging, automatic CPU fallback, no-AI/no-heavy-dependency boundary, and existing output quality.

---

## 2) Context / problem summary

Recent profiling on the user's real images in `C:\Users\Admin\Downloads\zzz` showed:

- Inputs are all about `6688x3776` / `25.25 MP`.
- GUI and export now use D3D12 correctly when `GPU Acceleration=Auto` or `Force GPU`.
- D3D12 uses all full-export render tiles with zero fallbacks.
- The synthetic D3D12 full-color tile kernel is ~120x faster than CPU, but real full export including PNG is only ~2.4-2.7x faster.

Observed real-image ranges:

```text
Full export CPU+PNG:        ~80.8s - 120.6s
Full export D3D12 Auto+PNG: ~33.4s - 53.8s
Proxy preview CPU:          ~4.1s - 4.3s
Proxy preview D3D12 Auto:   ~1.9s - 2.1s
Full Crop Auto:             ~21.4s - 23.7s
```

Current D3D12 wins are real, but the accelerated color stage is no longer the dominant cost:

```text
global matte:        ~22s - 36s
transition alpha:    ~14s - 22s
tile-local screen:   ~5s - 8s
tile-local refs:     ~4s - 7s
D3D12 color stage:   ~0.9s - 2.2s
PNG encode:          ~3.6s - 8.7s
```

Large-image behavior matters because 25MP exceeds existing caps:

```text
max_local_screen_model_pixels = 12M
_MAX_INNER_LABEL_PIXELS       = 16M
```

So the engine deliberately skips some full-image maps and does seam-safe tile-local CPU screen/reference prep.

---

## 3) Risks / constraints

- CPU remains the correctness reference for global matte, connected-background decisions, manual keep/remove, imported matte priority, trimap semantics, and source-alpha caps.
- Do not introduce AI, Torch, CuPy, ONNX, PyOpenCL, SciPy/skimage, libvips, or other heavy runtime dependencies.
- Keep large-image memory discipline: source `uint8`, masks `uint8`, bounded `int32` labels only below cap, no full-image float32 RGB.
- Cache invalidation must be conservative. Wrong cached matte is worse than slower output.
- Full Crop preview currently means exact crop from full-image global matte. If a faster approximate crop mode is added, it must be clearly labeled and must not silently affect full PNG export.
- Do not run multiple competing 25MP preview jobs. Coalesce to latest generation and cancel stale work at safe checkpoints.
- D3D12 must not change alpha in RGB-only color repair paths, and RGB must remain zero where alpha is zero.
- `.artifact/`, `build/`, `dist/`, native build outputs, caches, and `.claude/` must not be staged unless explicitly intended.
- Phase commits are only allowed when execution is explicitly requested; this plan file itself may remain uncommitted until the user approves execution/commit.

---

## 4) Current architecture model

Relevant modules after v10:

- `app.py` - app shell/state, settings defaults, preview/export orchestration entry points.
- `ui/preview_controller.py` - preview worker thread, generation/stale-result handling, proxy/full-crop mode dispatch.
- `ui/export_controller.py` - export worker and PNG write flow.
- `ui/settings_mapper.py` / `ui/widgets.py` - control mapping, slider emission behavior, reset defaults.
- `keyer.py` - public facade and compatibility shims.
- `imgkey_engine/matte.py` - global matte/screen/trimap/alpha decisions.
- `imgkey_engine/screen_model.py` - screen model and tile-local plate estimation.
- `imgkey_engine/references.py` - nearest-inner/reference map paths.
- `imgkey_engine/transition_alpha.py` - transition alpha recovery.
- `imgkey_engine/color_repair.py` - CPU/D3D12 color tile work and backend dispatch.
- `gpu_backend.py` / `native/imgkey_gpu/` - backend-neutral probe and D3D12 native tile color backend.
- `smoke_test.py` - smoke, geometric, GPU parity, perf baseline, package guards.

The plan should add durable timing/cache abstractions instead of one-off monkeypatch profiling.

---

## 5) Definitions / gates

### Settings categories

The implementation must define and test all `KeySettings` fields against at least these categories:

- Source-affecting: image path/content, source alpha, orientation/decode result, proxy/full-resolution source generation.
- Mask-affecting: manual keep/remove masks, imported matte, alpha-hint thresholds/strength, mask generations, source alpha cap interactions.
- Matte/base-alpha-affecting: key color, sample/auto selection, tolerance, `brightness_tolerance`, softness, `edge_softness`, clip background/foreground, matte gamma, core strength, edge radius, erode/expand, `despeckle_min_area`, connected/aggressive interior mode, `aggressive_threshold`, `aggressive_min_area`, guided alpha settings, screen cleanup settings if they change alpha, and any foreground/background classification setting.
- Transition-alpha-affecting: `transition_unmix`, `alpha_recover_strength`, `transition_spill_threshold`, `transition_reconstruction_error`, `foreground_reference_radius`, `transition_alpha_min`, `transition_alpha_max`, source alpha cap behavior, and any transition setting that can change alpha/recovered masks.
- Tile-geometry/prep-affecting: `local_screen_model`, `max_local_screen_model_pixels`, nearest-inner caps/radii, `tile_size`, `tile_overlap`, crop/read-overlap policy, and local-screen/reference settings. These are not backend-only on large images because they can affect tile-local prep/output.
- Color-only: despill, decontaminate, luminance restore/protect, fringe remove/color repair strength, inner color pull when it only changes RGB, key-vector despill, foreground color pull, checker/composite view choices, PNG compression choice.
- Backend-only: Auto/Off/Force GPU, selected backend, backend probe result, debug/perf flags that do not change output.

If a setting is ambiguous, classify it as matte-affecting until tests prove otherwise.




Current:
- No
### Cache contracts

The implementation must keep cache layers separate and immutable to consumers:

- Source cache: decoded full RGB/source alpha and proxy RGB/source alpha by image identity/generation.
- Base matte cache: full-resolution global screen decisions, base alpha, foreground/background/core/trimap masks, manual/imported matte merge state. Export must never reuse a proxy matte as a full-resolution matte.
- Transition cache: recovered alpha/transition masks and transition-specific metadata derived from the base matte and transition-affecting settings.
- Reference/tile-prep cache: tile-local screen/reference artifacts keyed by full/proxy generation, tile geometry, caps, crop/read-overlap, and prep-affecting settings.
- Color render cache: optional rendered RGBA/crop/tile outputs keyed by matte/transition/prep generation plus color-only settings and backend.

Cache publication rules:

- Add explicit generation counters and increment them on image load, proxy rebuild, source alpha/decode change, keep/remove mask edit, imported matte load/clear/update, mask reset, and any in-place mask modification. Cache keys must use these generations rather than object identity alone.
- Cancelled or stale preview jobs must not publish partial cache entries.
- Cache arrays returned through `KeyResult` or UI paths must be treated as read-only or copied before mutation.
- Cache entries must record whether they are full-resolution or proxy-resolution; full export can only use full-resolution entries.
- Default memory budget target for one active 25MP image is roughly `300-600 MiB` for source + masks/cache metadata, excluding transient tile float buffers and final export RGBA. If a phase exceeds this, it must document measured peak memory and add eviction/release rules.
- Evict/release old full-resolution cache generations on image change, mask/imported matte change, or conservative settings invalidation.




Current:
- No
### Target metrics

Use the three user PNGs plus synthetic baselines:

- Proxy preview target: first visible response under `1s` where possible via cached/progressive behavior; final proxy under current `~1.9-2.1s` or clearly explained.
- Full Crop target: crop changes should render from a valid matte cache in a few seconds, not recompute `~21-24s`; when cache is invalid, UI must show stale/proxy progress honestly.
- Export target: if preview already computed a valid full matte, export should avoid the `~22-36s` global recompute.
- Full export fresh target: reduce D3D12 Auto+PNG below current `~33-54s` by optimizing transition/reference/prep and PNG options.




Current:
- No
### Required full verification floor

Unless a phase explicitly says targeted-only, final verification must include:

```powershell
python smoke_test.py
python smoke_test.py --write-geometric-benchmark
python smoke_test.py --tune-geometric-defaults
python smoke_test.py --gpu-parity
python smoke_test.py --gpu-benchmark
python smoke_test.py --write-perf-baseline
native/imgkey_gpu/build.ps1 -Clean   # required before PyInstaller when native D3D12 code changed
python -m gpu_runtime --probe --json
python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py gpu_backend.py native_toolchain.py subprocess_utils.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py <expanded imgkey_engine/*.py> <expanded ui/*.py>
python -c "import app, keyer; print('import ok')"
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

PowerShell py_compile must use expanded file lists, not literal wildcards.

---

## 6) Phases

Phase commit rule for execution:

- Each phase is a commit boundary when `/do-plan` or explicit execution is requested.
- Before each commit inspect `git status --short --branch`, `git diff`, and `git log --oneline -10`.
- Do not stage generated `.artifact/`, `build/`, `dist/`, native outputs, or `.claude/`.

Scheduling note:
- Default execution is serial by phase number, except Phase 6.1 (`Fast PNG compression option`) is low-risk and independent after Phase 1 profiling. Planner may pull only P6.1 forward before Phase 4/5 if the user wants an immediate export-save-time win.
- Phase 6.2 (`Export progress and cache visibility`) depends on Phase 2 cache metadata; before Phase 2 it may only add generic stage progress, not cache-hit/cache-miss UX.
- Phase 5 (`Persistent D3D12 large-image batch pipeline`) is conditional/follow-on: do not implement native batch/readback work unless Phase 4/P7 profiling proves color-stage overhead is still a meaningful whole-pipeline bottleneck after cache and CPU-prep improvements.




Current:
- No
### Phase 1 - Durable large-image profiler and cache-key classification

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own profiling utilities, smoke/perf tests, and classification helpers. No behavior-changing cache yet.

Status:
- Completed


Progress notes:
- Phase 1 added reusable in-engine profiling hooks plus `python smoke_test.py --profile-large-images <image_dir>`, writing JSON/markdown under `.artifact/large-image-perf/` with load/decode, proxy, global matte, transition alpha, screen/reference prep, D3D12 tile, allocation/composite, PNG, and GUI-adjacent conversion timings.
- Real-image profiling on `C:\Users\Admin\Downloads\zzz` completed and recorded CPU/D3D12 preview/export breakdowns with backend usage/fallback counts.
- Cache-key helpers now classify every `KeySettings` field into source/mask/base-matte/transition-alpha/tile-prep/color/backend categories and provide generation-key fingerprint helpers for source/proxy/original-alpha and masks/imported matte. No runtime cache behavior was enabled.

#### P1.1 - Add first-class real-image profiling command
- Add a durable CLI/test path for profiling arbitrary input directories, expected command shape: `python smoke_test.py --profile-large-images <image_dir>` or an equivalent documented command. If `C:\Users\Admin\Downloads\zzz` is absent, skip it gracefully and run synthetic fallback fixtures.
- Report stage timings for load/decode, proxy resize, global matte substages, transition alpha, screen/reference prep, D3D12 tile dispatch/readback, result allocation/composite, PNG encode, and GUI-adjacent image conversion where measurable.
- Write reports under `.artifact/large-image-perf/` only.
- Use reusable in-engine timing hooks or a small profiler module; do not rely only on monkeypatch-style smoke instrumentation.

Execution:
- Serial

Isolation:
- `smoke_test.py` perf helpers and/or a new small profiling module. Do not alter keyer output behavior.

Acceptance:
- Running the profiler on the three user PNGs, when that directory exists, produces JSON + markdown with CPU/D3D12 preview/export breakdowns and backend usage/fallback counts.
- Missing real-image directory produces a clean skip message and still writes synthetic fallback perf output.

Verification:
- `python smoke_test.py --write-perf-baseline`
- targeted real-image profiling command
- `python smoke_test.py --gpu-parity`
- `git diff --check`

Status:
- Completed


Progress notes:
- Implemented `imgkey_engine.profiling.PipelineProfiler` and instrumented keyer/global matte, transition-alpha, tile prep, CPU/D3D12 color, result allocation/composite, and smoke-test load/proxy/PNG/GUI-adjacent stages.
- Added `--profile-large-images <image_dir>` with synthetic fallback when the directory is missing or has no supported image files; reports are `.artifact/large-image-perf/large_image_profile.json` and `.artifact/large-image-perf/large_image_profile.md`.

#### P1.2 - Classify settings and cache invalidation keys
- Add explicit helpers/tests that split `KeySettings` into source/mask/matte/color/backend-affecting fingerprints.
- Expand classification across every `KeySettings` field, including transition-alpha, local-screen, tile geometry, imported matte/alpha hint, aggressive cleanup, and screen cleanup fields.
- Document conservative rules for ambiguous settings.
- Add regression tests proving color-only settings do not invalidate matte fingerprints, while source image generation/original alpha, key color/tolerance, keep/remove/imported matte, transition-alpha settings, tile geometry/local-screen caps, and mask generations do.

Execution:
- Serial

Isolation:
- New cache-key helpers and tests only; no persistent runtime cache yet.

Acceptance:
- Settings classification is test-covered and usable by preview/export controllers and keyer cache code in later phases.

Verification:
- `python smoke_test.py`
- focused fingerprint tests
- py_compile expanded file list

Status:
- Completed


Progress notes:
- Implemented `imgkey_engine.cache_keys` with conservative field classification, stable settings/cache fingerprints, and source/mask/imported-matte generation-key payloads.
- Added smoke regression coverage proving color-only/backend settings preserve matte fingerprints while source/original-alpha generations, key/tolerance/matte fields, imported matte/mask generations, transition-alpha fields, and tile geometry/local-screen caps invalidate the appropriate matte pipeline keys.



Current:
- No
### Phase 2 - Matte/transition cache split and export reuse

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own cache structures and preview/export integration. Preserve existing public `process_key_image` behavior for callers without cache.

Status:
- Completed


Progress notes:
- Added `imgkey_engine.cache` with internal source/base matte/reference-prep/transition-alpha/color-render cache records, staged transactions, read-only cached arrays, `ProcessCacheContext`, and UI-owned `ProcessingGenerations` counters.
- `process_key_image` now accepts optional cache/context/transaction inputs while preserving existing behavior for callers without cache; preview workers stage cache publication and only commit after latest-generation UI acceptance, while cancelled/stale work discards staged records.
- Full-resolution Full Crop preview and export share full-source matte/transition cache keys; proxy cache keys stay resolution-separated and cannot satisfy full export.
- Color-only changes reuse matte/transition records and rerun color render only; cache metadata reports `cache_hit` and `cache_miss_reason` through `KeyResult.cache_info` and profiler metadata.
- Focused smoke tests cover cache contracts, full/proxy boundaries, stale/cancel discard, synthetic parity, user PNG parity/timing, and color-only no-global-matte rerender.


#### P2.1 - Define internal cache API contract
- Define internal cache objects and optional `process_key_image` cache input/output contract before implementing runtime reuse.
- Keep existing callers working without a cache object.
- Separate base matte, transition-alpha/recovered alpha, reference/tile-prep, and color render cache records.
- Define where source/proxy/mask/imported-matte generation counters live and which UI/controller operations increment them.
- Specify immutable/copy-on-return behavior for arrays exposed through `KeyResult` or UI result paths.
- Specify cancelled/stale preview behavior: no partial cache publication.

Execution:
- Serial

Isolation:
- Cache contract/types and focused tests. No behavior-changing cache reuse yet.

Acceptance:
- Contract is documented in code and plan notes, with tests for full-vs-proxy cache boundaries and stale/cancelled publication rules.

Verification:
- focused cache contract tests
- `python smoke_test.py`

Status:
- Completed


#### P2.2 - Introduce source/base/transition cache records
- Add cache records for decoded source identity, original alpha, manual/imported mask generations, full-resolution base matte, alpha, trimap/core/background masks, transition/recovered alpha, and optional reference metadata.
- Cache keys must include source identity, full/proxy resolution, mask generations, imported matte generation, matte/transition-affecting settings fingerprints, tile/prep geometry where relevant, and algorithm version.
- Bound memory to the active image/generation by default; release old full-size arrays on image/mask/settings generation changes.

Execution:
- Serial

Isolation:
- `imgkey_engine` cache helpers plus preview/export integration points. No GUI redesign yet.

Acceptance:
- Reusing a valid full-resolution cache produces byte/tolerance-equivalent output to a cold run.
- Proxy cache cannot be used for full export.
- Cache invalidation is conservative and visible in perf report.

Verification:
- targeted cache-hit/cache-miss tests
- CPU vs cache parity on synthetic and one user PNG
- cancelled preview does not publish cache
- `python smoke_test.py`

Status:
- Completed



#### P2.3 - Reuse full matte between preview and export
- When preview has built a valid full-resolution matte for the current image/settings/masks, export must reuse it instead of recomputing global matte/transition prep.
- If proxy-only preview has no valid full matte, export runs cold and records that reason.
- Export progress should distinguish cache hit vs cold global matte.

Execution:
- Serial

Isolation:
- Preview/export controller cache handoff and keyer optional cache input/output contracts.

Acceptance:
- On the three 25MP user PNGs, export after a valid Full Crop/accurate matte preview avoids the `~22-36s` global matte recompute.
- Cold export output remains unchanged.

Verification:
- targeted user-image timing: cold export vs preview-then-export
- parity max diff within existing tolerance
- `python smoke_test.py --gpu-parity`

Status:
- Completed



#### P2.4 - Avoid matte rebuild for color-only changes
- For color-only slider changes, reuse matte/transition cache and rerun only color render/composite stages.
- Ensure color-only changes that currently affect alpha are reclassified as matte-affecting instead of risking stale alpha.
- Add debug/perf metadata showing `cache_hit: matte` / `cache_miss_reason`.

Execution:
- Serial

Isolation:
- Cache invalidation and render routing only.

Acceptance:
- Adjusting despill/fringe/color repair/foreground pull after a valid matte cache does not rerun global matte.
- Output equals cold run for the same final settings.

Verification:
- targeted color-only slider simulation tests
- `python smoke_test.py`
- `python smoke_test.py --write-geometric-benchmark`

Status:
- Completed





Current:
- No
### Phase 3 - Progressive preview and responsive large-image UX

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own preview scheduling, preview UI affordances, and slider debounce behavior. Do not change final export semantics.

Design brief:
- External patterns: large-image apps avoid whole-image display/render work; use tiles/pyramids/ROI first, background workers with cooperative cancellation, progressive passes, and debounced/commit-based expensive sliders.
- Local inventory: `PreviewThread` already has QThread, generations, progress, cancel flag, proxy max-side 1800, and `Full Crop`; but Full Crop still builds full-image matte and sliders can overschedule work.
- Gap: Full Crop sounds ROI-fast but currently costs `~21-24s`; status can imply GPU accelerates more than it does; cancellation is latest-wins but stale global matte work can occupy the lane; preview progress is less explicit than export progress.
- Recommendations: keep previous result visible, show stale/proxy/exact state, run proxy/draft first, render exact crop from cache when possible, add stronger cancellation checkpoints, throttle expensive sliders, expose `Refresh Crop`/`Pin Crop`/`Cancel Preview` as lightweight HUD/status actions if needed, and state `CPU global matte... GPU accelerates color tiles` honestly.
- Anti-patterns: do not auto-run 25MP Full Crop on every pan/slider tick, do not silently treat approximate crop as exact, do not let preview crop affect full export, do not spawn multiple 25MP preview jobs.

Status:
- Completed


Progress notes:
- Phase 3 clarified preview semantics in the inspector/HUD/status: Proxy is a fast whole-image preview, Full Crop is an exact pinned full-resolution ROI, cache state/crop dimensions are visible, and Refresh Crop explicitly recaptures the current viewport without affecting full PNG export.
- Preview scheduling is now progressive/latest-wins: cold Full Crop requests show a proxy draft first, exact crop runs after the full matte path, warm full matte cache goes straight to exact crop color render, only one preview worker runs at a time, stale results are ignored, staged stale cache is discarded, and the previous accepted preview remains visible during new work.
- Slider rows now keep cheap value/label updates live while active drags request debounced draft/proxy preview; committed exact work is scheduled on release while plus/minus buttons and numeric boxes remain immediate.
- Added stronger preview cancellation checkpoints after global matte substages, before GPU render setup/dispatch, and before each tile color render.



#### P3.1 - Define exact vs draft preview semantics
- Keep `Proxy` as fast whole-image preview.
- Rename or clarify `Full Crop` in UI/HUD as exact full-resolution ROI that may require full-image matte unless cache exists.
- Keep current pinned exact crop behavior for v11: `Full Crop` uses the pinned `_full_crop_rect`, not every pan/zoom movement. Make this explicit in HUD/status and add a lightweight `Refresh Crop` action if needed to recapture the current viewport rect.
- Add an optional draft/fast crop path only if clearly labeled approximate and never used for final export.
- Show cache state and crop dimensions in status/HUD.

Execution:
- Serial

Isolation:
- UI labels/status/HUD and preview mode metadata only.

Acceptance:
- User can tell whether preview is proxy, exact crop, cached, stale, or cold global matte.
- Pan/zoom does not silently trigger a new exact crop; preview crop state never silently changes export scope.

Verification:
- focused UI default/status tests
- manual GUI smoke with one user PNG
- `python smoke_test.py`

Status:
- Completed

Progress notes:
- Full Crop remains pinned to `_full_crop_rect`; pan/zoom only changes the pin after Refresh Crop or a mode switch.
- HUD/status now distinguish Proxy whole image vs exact pinned crop, show crop dimensions/origin, show cold/partial/matte-cached state, and state that export remains full image.



#### P3.2 - Progressive latest-wins preview scheduler
- Keep previous result visible during new work.
- For matte-cache-valid changes, prioritize visible/crop color render.
- For matte-cache-invalid changes, show proxy/draft quickly, then exact crop/full preview when global matte completes.
- Add cancellation checkpoints before/after global matte substages, before GPU dispatch, and every tile.
- Coalesce pending preview requests to newest generation only.

Execution:
- Serial

Isolation:
- Preview controller and keyer cancellation hooks. Export controller remains correctness-first.

Acceptance:
- Rapid slider movement/pan/crop changes do not queue multiple 25MP jobs.
- Stale jobs stop earlier than the current global-matte-bound behavior where safe.
- There is at most one running preview job; pending jobs are coalesced to newest generation; stale results are ignored; previous result remains visible during recompute.

Verification:
- synthetic/offscreen cancellation and coalescing tests
- stale-result ignored test
- previous-result-remains-visible probe where practical
- manual GUI drag/slider smoke
- `python smoke_test.py --gpu-parity`

Status:
- Completed

Progress notes:
- Preview requests coalesce by generation; running jobs are cancelled cooperatively and pending work restarts only for the newest generation.
- Cold Full Crop schedules Proxy draft first, then exact pinned crop; cached full matte skips straight to exact crop color render.
- Accepted results stay visible until a newer accepted result arrives; stale completions discard staged cache and do not update the canvas.



#### P3.3 - Expensive slider debounce/tracking policy
- Split cheap UI value updates from expensive processing for high-cost controls.
- During active drag, use draft/proxy preview and schedule exact update on release or trailing debounce (`300-500ms`) where appropriate.
- Preserve plus/minus stepping and numeric boxes.

Execution:
- Serial

Isolation:
- `ui/widgets.py`, settings mapper, preview scheduling only.

Acceptance:
- Slider drag feels responsive and does not repeatedly start cold 25MP matte jobs.

Verification:
- focused slider signal tests where possible
- manual GUI slider smoke
- py_compile/import

Status:
- Completed

Progress notes:
- Slider drag tracking now separates live widget value updates from expensive preview processing with a 400 ms draft debounce during active drags and committed preview scheduling on release.
- Plus/minus stepping and numeric spin boxes continue to emit immediate committed preview requests.



---


Current:
- No
### Phase 4 - CPU bottleneck reduction for transition/reference prep

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `imgkey_engine/transition_alpha.py`, `screen_model.py`, `references.py`, and tests. D3D12 native API changes are out of scope unless explicitly proven small.

Status:
- Completed



Progress notes:
- Phase 4 optimized the hot CPU matte-prep stages from profiler data without changing behavior intent: transition alpha now computes spill/linear solve only for eligible pixels and skips tile-local screen/reference prep for tiles with no transition candidates, while trimap alpha computes smoothstep/gamma only on the edge mask and blurs bounded edge ROIs.
- Added bounded tile-prep cache records keyed by transition matte generation and tile geometry so color-only rerenders reuse tile-local screen/reference artifacts without full-image float maps; stale/cancel publication remains transaction-gated.
- Real-image Phase 4 timing report is under `.artifact/phase4-large-image/phase4_timing_comparison.md`: compared 25MP user cases show D3D12 export process avg `42.50s -> 21.27s`, global matte `34.29s -> 16.03s`, transition alpha `19.70s -> 8.29s`, transition block `8.67s -> 0.49s`, and trimap alpha `8.87s -> 3.52s`.
- D3D12 prep-port decision: defer Phase 5. Remaining candidates are readback-heavy CPU-reference prep artifacts (local screen/reference maps, distance labels, trimap morphology) rather than RGB-only color work; if reopened, native work should be limited to a bounded tile-prep batch API with CPU fallback/parity gates.


#### P4.1 - Optimize transition alpha and global matte substages on CPU
- Use Phase 1 profiler to target the slowest substage first.
- Reduce redundant array allocations and passes.
- Prefer `uint8`/boolean masks and ROI operations; avoid full-image float32 RGB.
- Add timings around each optimized substage.

Execution:
- Serial

Isolation:
- CPU algorithm implementation only; no behavior-intent changes.

Acceptance:
- Measurable reduction in `transition alpha` / global matte time on the three 25MP user PNGs with parity within existing tolerances.

Verification:
- targeted user-image stage timings before/after
- `python smoke_test.py`
- `python smoke_test.py --write-geometric-benchmark`

Status:
- Completed

Progress notes:
- Optimized transition alpha candidate/linear solve and trimap alpha subpasses; added profiler stages `global_matte.trimap_morphology`, `global_matte.trimap_alpha_math`, `global_matte.trimap_blur`, `transition_alpha.tile_candidate_mask`, `transition_alpha.spill_mask`, `transition_alpha.eligible_mask`, `transition_alpha.linearize_eligible`, and `transition_alpha.solve_eligible`.



#### P4.2 - Reuse tile-local screen/reference prep inside a render generation
- Avoid recomputing tile-local screen/reference data when rendering multiple color-only variants for the same matte/tile geometry.
- Cache only bounded per-tile or per-generation compact artifacts; do not create large unbounded global float maps.

Execution:
- Serial

Isolation:
- Render generation cache and tile-local prep helpers.

Acceptance:
- Color-only rerenders reuse tile prep where valid and remain seam-free.

Verification:
- tile seam tests
- cache parity tests
- `python smoke_test.py --gpu-parity`

Status:
- Completed

Progress notes:
- Added per-generation `TilePrepRecord`/`TilePrepEntry` cache with compact uint8/bool per-read-tile screen/reference artifacts. Color-only rerenders with the same matte/tile geometry hit `cache_info["tile_prep"] == "hit"` and preserve seam/crop parity.



#### P4.3 - Decide D3D12 port candidates for remaining dense CPU stages
- Based on measured post-cache/post-CPU timing, decide whether to port transition alpha/screen probability/reference prep to D3D12.
- If the candidate requires readback-heavy intermediate maps with low reuse, document why not.
- If viable, create a small follow-on implementation spec for Phase 5 native API additions.

Execution:
- Serial

Isolation:
- Decision/spec only unless a tiny prototype is clearly isolated.

Acceptance:
- Plan has a data-backed decision for which CPU stage, if any, moves to D3D12 next.

Verification:
- updated profiling report
- reviewer sanity check before native API expansion if scope is large

Status:
- Completed

Progress notes:
- Phase 5 native D3D12 expansion is deferred: post-CPU/cache profiling shows transition solve math is no longer a large kernel target, and remaining prep stages would require CPU-owned alpha/reference-map readbacks with limited reuse. Follow-on spec, if ROI returns, is a bounded native tile-prep batch API only.



---


Current:
- No
### Phase 5 - Conditional persistent D3D12 large-image batch pipeline

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `gpu_backend.py`, `native/imgkey_gpu/`, and color-repair native dispatch. CPU fallback ABI and existing single-tile path must remain until batch path passes parity.

Precondition:
- Execute only if post-Phase-2/4 profiling shows native color-stage overhead, Python/native round trips, or GPU resource churn still materially affects end-to-end preview/export time. If D3D12 color remains `~1-2s` while CPU/PNG dominate, skip or defer this phase and document the reason.

Status:
- Deferred


Progress notes:
- Deferred after Phase 4 timing review. D3D12 color dispatch remains under roughly `0.9-1.0s` averaged on 25MP exports after CPU prep improvements, while the remaining expensive prep stages are CPU-reference/readback-heavy local screen/reference and trimap morphology work. No native API expansion is justified before the PNG/export UX phase.



#### P5.1 - Add native tile-batch ABI and persistent resource plan
- Add or finalize `imgkey_gpu_process_tile_batch_v1` behind capability flags.
- If the native ABI/header/spec changes, update `docs/build-gpu.md`, `ImgKey.spec`, native headers, and Python ctypes wrapper in the same milestone.
- Upload image/masks/constants once per render generation where practical.
- Reuse D3D12 context, root signatures, PSOs, command allocator pool, upload/readback buffers, and descriptor layouts.
- Keep dispatch chunks bounded for TDR safety.

Execution:
- Serial

Isolation:
- Native D3D12 + Python backend wrapper. No UI changes.

Acceptance:
- Batch path can be enabled behind capability detection and falls back to the existing path on failure.

Verification:
- native build
- focused D3D12 batch identity/color tests
- `python smoke_test.py --gpu-parity`

Status:
- Planned



#### P5.2 - Async/batched readback and reduced Python/native round trips
- Record many tile dispatches per command list and read back in batches.
- Avoid per-tile context setup or wait/readback loops where possible.
- Preserve progress/cancel granularity at safe batch boundaries.

Execution:
- Serial

Isolation:
- Native backend and `gpu_backend.py` session lifecycle.

Acceptance:
- Real-image D3D12 color stage and proxy preview improve beyond current `~0.9-2.2s` full color stage and `~1.9-2.1s` proxy where overhead is material.
- No parity regression.

Verification:
- targeted D3D12 batch benchmark
- `python smoke_test.py --gpu-benchmark`
- `python smoke_test.py --gpu-parity`

Status:
- Planned



#### P5.3 - Optional D3D12 port of selected dense prep stage
- Only execute if Phase 4 proves a dense stage is a good GPU-resident candidate.
- Keep CPU reference and fallback.
- Do not port connected-component/global semantic decisions in a tile-local way that risks seams.

Execution:
- Serial

Isolation:
- Native backend + selected engine stage. Requires explicit milestone update before implementation.

Acceptance:
- Stage speedup improves whole-export time, not just isolated kernel time.
- Parity/visual gates pass.

Verification:
- full large-image before/after profile
- smoke/geometric/GPU parity

Status:
- Planned



---



Current:
- No
### Phase 6 - Fast export path and PNG encode options

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own export options, PNG compression setting, docs, and tests. Do not change default output quality without explicit acceptance.

Status:
- Planned




#### P6.1 - Add Fast PNG compression option
- Add an export option for faster PNG compression level while preserving lossless pixels.
- Default may remain current compression unless benchmark and UX justify changing it.
- Show expected tradeoff: faster save, larger file.

Execution:
- Serial

Isolation:
- Export UI/settings and PNG writer only.

Acceptance:
- On the three user PNGs, fast PNG reduces `~3.6-8.7s` encode cost measurably.
- Pixel output is identical after decode; file size difference is documented.

Verification:
- targeted encode benchmark
- export/import pixel parity
- `python smoke_test.py`

Status:
- Planned



#### P6.2 - Export progress and cache visibility
- Export progress should show whether it is using cached matte, running CPU global matte, D3D12 color render, or PNG encode.
- Keep cancel support.

Execution:
- Serial

Isolation:
- Export controller/status only.

Acceptance:
- User can see why large export is waiting and whether GPU is active.

Verification:
- manual GUI export smoke
- focused status/progress tests where possible

Status:
- Planned



---




Current:
- Yes
### Phase 7 - Full verification, packaging, and release-readiness gate

Category:
- Review-heavy

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own verification, docs, packaging checks, plan completion. No broad new features.

Status:
- Planned



#### P7.1 - Full real-image benchmark matrix
- Run CPU/D3D12 Off/Auto/Force on the three user PNGs for proxy, exact crop, cold export, preview-then-export cache-hit export, and fast PNG export.
- Compare against v10/v11 baseline numbers.
- Include perceived-latency notes from GUI manual checks.

Execution:
- Serial

Isolation:
- `.artifact/large-image-perf/` outputs only.

Acceptance:
- Report clearly states where time is now spent and what improved.

Verification:
- generated benchmark report
- parity max channel diff within tolerance

Status:
- Planned



#### P7.2 - Full regression/build gate
- Run the full verification floor.
- If native D3D12 code changed, run `native/imgkey_gpu/build.ps1 -Clean` before PyInstaller.
- Rebuild primary `ImgKey.exe`.
- Probe packaged EXE from temp cwd and sanitized PATH.
- Run GUI lifetime smoke.
- Check archive excludes heavy runtimes and includes expected `imgkey_gpu.dll`.

Execution:
- Serial

Isolation:
- Build/verification only. Do not stage generated artifacts.

Acceptance:
- One-file CPU+D3D12 `ImgKey.exe` passes verification and has size/SHA256 recorded.

Verification:
- full verification floor listed in section 5
- packaged EXE probes
- archive exclusion checks
- `git diff --check`

Status:
- Planned



#### P7.3 - Complete docs and plan
- Update README/docs/build notes only where user-visible behavior changed.
- Update `AGENTS.md` after cache/preview architecture lands so future agents know v11 profiler/cache commands, cache semantics, and large-image workflow constraints.
- During v11 execution, this plan supersedes the current `AGENTS.md` wording that names v10 as the active refactor plan.
- Mark this plan `Completed` with final performance summary and known limitations.

Execution:
- Serial

Isolation:
- Docs and plan only.

Acceptance:
- Future agents can understand the cache/preview architecture and verification commands from repo docs.

Verification:
- doc diff review
- final `git status --short --branch`

Status:
- Planned



---

## 7) Stop-and-ask boundaries

Stop and ask the user before:

- Changing final export semantics or letting preview crop affect full PNG export.
- Adding a new runtime dependency outside the current dependency fence.
- Raising memory caps enough to risk high-RAM crashes on 25MP+ images.
- Shipping a default approximate matte/output path that differs visibly from current accurate output.
- Changing public release packaging away from one primary `ImgKey.exe`.

---

## 8) Immediate next step

Next execution target is Phase 6 using `worker`:

1. Add a fast PNG compression option while preserving lossless pixels.
2. Surface export progress/cache/GPU status during long saves.
3. Keep Phase 5 deferred unless new timing evidence reopens native D3D12 batch ROI.
