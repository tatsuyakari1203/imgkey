# 01 - ImgKey v2 Large-Image Accurate Keyer

Date: 2026-05-16
Status: Completed
Owner: ImgKey
Scope: Rebuild ImgKey into an image-only, high-accuracy chroma-key/matting tool optimized for large still images, with optional AI seams and a flat viewer-first UI.

---

## 1) Goal

Deliver a Windows Python app that lets a user load a large image, accurately remove a green/blue/custom keyed background, inspect the matte at preview and 100% crop quality, and export a full-resolution transparent PNG. The default path must be fast and reliable without AI dependencies; AI-assisted workflows must be optional and not bloat the default EXE.

---

## 2) Context / problem summary

- Current app files: `app.py`, `keyer.py`, `README.md`, `requirements.txt`, `smoke_test.py`.
- Current UI uses two `QLabel` previews and a fixed rounded settings panel. It wastes image space and has no zoom, pan, matte diagnostics, full-res crop preview, progress, or cancel.
- Current algorithm in `keyer.py` is simple chroma distance + channel dominance + morphology + Gaussian edge blur + despill. It leaves residual blue/green, eats foreground details, and cannot distinguish connected background from same-color foreground islands.
- User priority: large still images and high accuracy, not video. Tile processing is acceptable only if global semantic/mask steps remain global and tile seams are prevented.
- Research direction:
  - CorridorKey solves a harder neural unmixing problem: RGB plate + coarse alpha hint -> straight foreground color + linear alpha. It is useful as an optional plugin seam, not a default bundled dependency due model/runtime size and license constraints.
  - VFX keyers such as Keylight/Primatte/IBK use screen/clean-plate models, core matte + edge matte separation, clip black/white, despill, foreground/luminance restoration, and diagnostic matte views.
  - PyMatting can refine trimaps and estimate foreground, but should be used selectively around edge bands/ROIs for large images.
  - BiRefNet matting can be an optional MIT AI alpha-hint generator; RVM is GPL/human-video oriented; BackgroundMattingV2 needs a clean background plate.

---

## 3) Risks / constraints

- Accuracy risk: removing every pixel close to the key color will destroy legitimate foreground colors. Use connected-background logic and optional keep/remove hints instead of global color deletion.
- Tile risk: per-tile key color sampling or connected-component decisions will produce seams and inconsistent alpha. Global mask/trimap decisions must happen before tiled refinement.
- Memory risk: full-image float32 RGB is too expensive for 8K/12K images. Keep source as `uint8`, masks as `uint8`, and convert to float only per tile/edge ROI.
- UX risk: proxy preview can lie. UI must label preview scale and provide 100% crop/full-res ROI preview before export.
- Packaging risk: bundling PyTorch/CUDA/CorridorKey by default would make the EXE huge and complicate licensing. AI support is optional/plugin-style.
- Stop-and-ask boundary: before bundling or redistributing CorridorKey or noncommercial model weights, stop and ask about intended distribution/licensing.
- Baseline safety: `D:\keyphong` is not currently a git repo. Before deep rewrites, either initialize version control or create a timestamped source backup; do not rely on `build/` or `dist/` as recoverable source.
- Dependency gate: default install may add `opencv-contrib-python` only if guided filtering is chosen. `pymatting`, `onnxruntime-gpu`, `torch`, BiRefNet, and CorridorKey remain optional until explicitly approved.
- Large-image target: support at least 8K still images on normal RAM by avoiding full-image float32 RGB allocations; target preview updates under ~500ms for proxy changes and keep peak export memory roughly within source RGB + final RGBA + a small set of uint8 masks + one float tile.

---

## 4) Architecture target

### Core modules

- `keyer.py` or split into `keying/` package:
  - `settings.py`: typed settings/presets.
  - `screen_model.py`: key sampling, border sampling, HSV/Lab/RGB screen probability.
  - `matte.py`: hard background/foreground mask, trimap, connected components, edge band.
  - `refine.py`: edge-only alpha refinement, optional PyMatting/guided-filter path.
  - `color.py`: unmix, despill, luminance/foreground restore.
  - `tiles.py`: overlap tile iterator and core-write stitching.
  - `pipeline.py`: preview/export orchestration returning a shared result object.
- `app.py` or split UI files:
  - `ImageCanvas(QGraphicsView)`: zoom/pan, checkerboard, view modes, eyedropper mapping.
  - `InspectorPanel`: flat VFX-style controls.
  - `WorkerThread`: preview/export background execution with progress/cancel.







Current:
- No
### Shared result contract

```python
@dataclass
class KeyResult:
    rgba: np.ndarray                 # display/export RGBA, straight alpha
    alpha: np.ndarray                # uint8 or float alpha depending stage
    foreground: np.ndarray | None    # optional straight foreground RGB
    background_mask: np.ndarray | None
    edge_mask: np.ndarray | None
    despill_mask: np.ndarray | None
    preview_scale: float = 1.0
```







Current:
- No
### Large-image pipeline

```text
Load image as uint8 RGB/RGBA
-> global key sampling / screen model
-> global hard background mask from border-connected regions
-> global protected foreground/core mask
-> global trimap + edge band
-> preview proxy result for UI
-> export full-res with tiled edge/color processing:
   - tile with overlap
   - refine alpha only in edge/unknown band
   - unmix foreground color
   - despill/decontaminate
   - write only tile core to output RGBA
-> final PNG
```

---

## 5) Phases







Current:
- No
### Phase 0 - Repo context and baseline safety

Category:
- Repo-context

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `AGENTS.md`, `.gitignore`, `.artifact/`, `smoke_test.py`, and optional synthetic fixture helper only. Do not change algorithm/UI in this phase.

Status:
- Completed


Progress:
- Completed 2026-05-17: added repo context, artifact hygiene, baseline backup, and non-enforcing v2 diagnostic smoke fixtures.

#### P0.1 - Add lightweight repo context and artifact hygiene
- Because the folder is not a git repo, first create a safety point: ask whether to `git init` or create a timestamped backup copy of source files. If execution is authorized without user preference, create a non-destructive source backup outside `build/`/`dist/`.
- Create/refresh root `AGENTS.md` with project structure, run/build/test commands, and current constraints.
- Ensure `.gitignore` covers `.artifact/`, `build/`, `dist/`, `__pycache__/`, `.pytest_cache/`, generated test outputs, and model/download caches.
- Document that generated EXE/build artifacts are not source-of-truth.

Execution:
- Serial

Isolation:
- `AGENTS.md`, `.gitignore` only.

Acceptance:
- Future executors can start without rediscovering file roles and verification commands.

Status:
- Completed


Progress:
- Completed 2026-05-17: created `AGENTS.md`, `.gitignore`, ignored `.artifact/`, and a timestamped source backup under `.artifact/source-backup-*`.

#### P0.2 - Establish baseline fixtures
- Add or update smoke fixtures that create synthetic green/blue/custom keyed images for current and future checks with:
  - flat background,
  - uneven background gradient,
  - foreground same-color island that must be preserved,
  - semi-transparent/anti-aliased edge,
  - large-image synthetic case that avoids committing huge binaries.
- In Phase 0, keep new future-target checks non-enforcing or marked expected-fail/diagnostic until Phase 1 implements the engine. Do not make Phase 0 fail because v2 behavior is not implemented yet.
- Save generated debug outputs under `.artifact/` only.

Execution:
- Serial

Isolation:
- `smoke_test.py` and optional fixture-generation helper. No UI rewrite.

Acceptance:
- `python smoke_test.py` still passes on the current implementation and can generate/describe future v2 diagnostic fixtures without failing baseline.

Status:
- Completed


Progress:
- Completed 2026-05-17: `smoke_test.py` still enforces only the current flat-green baseline and can optionally write future-target diagnostics under `.artifact/smoke-fixtures/`.

---






Current:
- No
### Phase 1 - High-accuracy image keying engine

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own keying algorithm modules and tests. Preserve public app entrypoint `process_chroma_key()` until UI migration is ready, or provide a compatibility wrapper.

Status:
- Completed


Progress:
- Completed 2026-05-17: implemented the non-AI high-accuracy engine in `keyer.py`, expanded `smoke_test.py` to enforce v2 numeric coverage, and preserved the existing `app.py` compatibility API without new default dependencies.

#### P1.1 - Introduce explicit settings and result model
- Replace generic `KeySettings` with settings covering:
  - mode: `GraphicExact`, `ProChroma`, future `AIHint`,
  - key color / sample size / auto border sample,
  - clip background, clip foreground, matte gamma,
  - core strength, edge refine radius, edge softness,
  - erode/expand, despeckle min area,
  - despill amount, decontaminate strength, luminance restore,
  - preview scale / full-res crop options.
- Add `KeyResult` and debug outputs (`alpha`, `background_mask`, `edge_mask`, `despill_mask`).

Execution:
- Serial

Isolation:
- `keyer.py` or new `keying/*.py`; app compatibility wrapper retained.

Acceptance:
- Existing smoke test still passes through compatibility API; new settings can represent both simple and pro controls.

Status:
- Completed


Progress:
- Completed 2026-05-17: added expanded `KeySettings`, `KeyResult`, `process_key_image()`, debug masks, and retained `process_chroma_key()`/`checkerboard_composite()` compatibility.


#### P1.2 - Implement `GraphicExact` global mask path
- Sample global screen color from border and/or eyedropper; support green, blue, and custom colors.
- Add an uneven-screen model:
  - estimate a per-pixel screen color/strength field from border samples and background candidates,
  - optionally synthesize a clean-plate-like background color map by smoothing known background regions,
  - fall back to a single global key color only for flat backgrounds.
- Compute screen probability using a blend of:
  - normalized chroma distance,
  - HSV hue/saturation distance,
  - dominant channel/background channel contrast for green/blue,
  - Lab/linear RGB optional distance if it improves uneven lighting.
- Build hard background candidates from probability thresholds.
- Flood-fill from image borders to retain only background connected to image edges.
- Preserve foreground islands even if they are near key color unless user explicitly enables aggressive global removal.
- Add non-AI correction policy for interior regions:
  - `Connected Background` default preserves non-border-connected islands,
- `Aggressive Interior Removal` removes high-confidence key-colored islands below/above configurable area rules.
- Use connected components to remove small background/foreground speckles with area thresholds.

Execution:
- Serial

Isolation:
- Algorithm modules only.

Acceptance:
- Synthetic same-color foreground island is preserved in default mode; interior keyed background can be removed with aggressive non-manual policy; uneven gradient background fixture is mostly removed without eating foreground; no tile seams because this phase is global.

Status:
- Completed


Progress:
- Completed 2026-05-17: added border/eyedropper sampling, uneven-screen probability, border-connected background extraction, component cleanup, default island preservation, and aggressive interior removal.


#### P1.3 - Add core/edge trimap and edge-only refinement
- Build trimap:
  - certain background = alpha 0,
  - certain foreground/core = alpha 255,
  - edge/unknown band = needs refinement.
- Add edge band generation via distance transform / morphology, not global blur.
- Implement fast edge refinement:
  - local smoothstep alpha from screen probability inside unknown band,
  - optional guided/bilateral smoothing limited to edge band,
  - optional PyMatting/closed-form/KNN only for ROI/crop if performance is acceptable.
- Add clip black/clip white/matte gamma controls.

Execution:
- Serial

Isolation:
- Algorithm modules and tests.

Acceptance:
- Anti-aliased synthetic edges retain soft alpha while flat background becomes exactly alpha 0 and core foreground exactly alpha 255.

Status:
- Completed


Progress:
- Completed 2026-05-17: added morphology-based edge band trimap, exact bg/fg cores, smoothstep alpha refinement, clip controls, matte gamma, and edge-only smoothing.


#### P1.4 - Implement foreground unmix and despill/decontamination
- Apply compositing equation in edge pixels:
  - `C = alpha * F + (1-alpha) * K`
  - `F = (C - (1-alpha)*K) / alpha`
- Use safe alpha floor only in edge band; never brighten transparent background junk.
- Add channel-based despill for green/blue and vector projection despill for custom key.
- Add luminance restoration so despill does not gray out foreground edges.
- Output straight-alpha PNG; display composite only in UI.

Execution:
- Serial

Isolation:
- `color.py`/`keyer.py` and tests.

Acceptance:
- Residual keyed color on edge pixels is reduced in debug metrics; transparent background RGB is zeroed; exported PNG remains straight-alpha.

Status:
- Completed


Progress:
- Completed 2026-05-17: added edge-band foreground unmixing, channel/vector despill, luminance restore, straight-alpha output, and transparent RGB zeroing.


#### P1.5 - Implement tiled full-resolution export without seams
- Add tile iterator with configurable tile size and overlap:
  - default tile size: `2048` or adaptive from RAM,
  - overlap: `96-192px`, at least `4x` edge refine radius.
- Run global mask/trimap before tiling.
- In each tile, process only expensive edge/color operations; write only tile core to final output.
- Add progress callbacks and cancellation checks.
- Avoid full-image float32 RGB allocations.

Execution:
- Serial

Isolation:
- `tiles.py`, export pipeline, tests.

Acceptance:
- Large synthetic image export stays bounded in memory; seam test comparing tile boundaries finds no visible alpha discontinuity beyond tolerance.

Status:
- Completed


Progress:
- Completed 2026-05-17: added overlapped tile/core-write rendering with progress/cancel hooks; global mask/trimap runs before tiling and smoke tests compare tiled vs non-tiled output.


#### P1.6 - Add manual matte correction hooks, without AI
- Add optional keep/remove mask inputs in the engine:
  - keep mask forces foreground/core or protects same-color details,
  - remove mask forces background/unknown cleanup,
  - masks can be imported from grayscale PNG even before UI brush tools exist.
- Add merge rules between screen model, connected background, and manual masks.
- Expose enough API for Phase 2 to add simple brush/import controls.

Execution:
- Serial

Isolation:
- Engine mask merge and tests only.

Acceptance:
- A same-color foreground region can be protected by keep mask; an interior keyed background region can be removed by remove mask; app still works without masks.

Status:
- Completed


Progress:
- Completed 2026-05-17: added optional keep/remove mask inputs plus grayscale mask import/export helpers; smoke tests cover protection and forced removal.


---






Current:
- No
### Phase 2 - Viewer-first flat UI redesign

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `app.py` and any new UI modules. Do not change core algorithm semantics except wiring settings/result fields.

Design brief:
- Use image-editor/VFX viewer patterns: one large central viewer, right inspector, matte/status/debug views, zoom/pan/100% crop, thin toolbar, status bar.
- Prefer `QGraphicsView` over `QLabel` for accurate coordinate picking and scalable view transforms.
- Style: flat dark minimal, small `4-6px` radii, thin separators, compact controls; avoid rounded card-heavy layout.

Status:
- Completed


Progress:
- Started 2026-05-17 by deep-worker: created non-git source backup under `.artifact/source-backup-phase2-*`; implementing viewer-first QGraphicsView canvas, flat inspector, and threaded preview/export wiring without AI dependencies.
- Completed 2026-05-17: replaced the dual preview labels with a single `ImageCanvas(QGraphicsView)`, added VFX debug view modes/backgrounds/status readouts, rebuilt the inspector with Phase 1 settings/mask hooks, and moved export to a progress/cancel worker while keeping AI dependencies out of scope.


#### P2.1 - Replace dual-label previews with `ImageCanvas(QGraphicsView)`
- Implement zoom, pan, fit, 100%, mouse-coordinate mapping, and eyedropper sampling.
- Add checkerboard/black/white/gray/transparent display backgrounds.
- Add view modes: `Result`, `Source`, `Alpha`, `Background Mask`, `Edge Mask`, `Despill Mask`, `Split Compare`.
- Show current preview scale and cursor RGB/alpha in status bar.

Execution:
- Serial

Isolation:
- UI canvas files only.

Acceptance:
- User can inspect a crop at 100%, pick key color accurately, and switch alpha/debug views without exporting.

Status:
- Completed

Progress:
- Completed 2026-05-17: `ImageCanvas` now supports result/source/alpha/background/edge/despill/split views, checker/black/white/gray/transparent scene backgrounds, fit/100%/wheel zoom/pan, accurate mapped eyedropper sampling, and cursor RGB/alpha status.


#### P2.2 - Build flat VFX inspector
- Replace generic sliders with grouped controls:
  - Key: Auto / Green / Blue / Pick, sample size, key chip.
  - Matte: Clip Background, Clip Foreground, Matte Gamma, Core Strength, Despeckle.
  - Edge: Edge Radius, Edge Softness, Erode/Expand, Edge Restore.
  - Color: Despill, Decontaminate, Luminance Restore.
  - Output: Proxy/Adaptive/Full Crop preview, Export PNG.
- Add non-AI correction controls:
  - import keep/remove mask PNG,
  - export current matte,
  - simple `Connected Background` vs `Aggressive Interior Removal` toggle.
- Use `SliderRow` with numeric spinbox and reset/default buttons.
- Add presets: `Fast`, `Clean`, `High Accuracy`.

Execution:
- Serial

Isolation:
- Inspector/UI wiring only.

Acceptance:
- Controls map directly to engine settings; no unlabeled magic sliders; app remains usable with optimized defaults.

Status:
- Completed

Progress:
- Completed 2026-05-17: inspector is grouped into Key/Matte/Edge/Color/Output with direct `KeySettings` wiring, numeric slider rows with reset buttons, Fast/Clean/High Accuracy presets, connected-vs-aggressive policy, keep/remove mask import, and current matte export.


#### P2.3 - Add non-blocking preview/export UX
- Keep proxy preview threaded/debounced.
- Add full-res crop preview for current viewport/ROI.
- Run export in worker with progress, cancel, and elapsed-time status.
- Keep UI responsive while processing large images.

Execution:
- Serial

Isolation:
- UI worker/export wiring only.

Acceptance:
- Export no longer blocks UI thread; cancellation leaves app stable; progress reflects tile completion.

Status:
- Completed

Progress:
- Completed 2026-05-17: debounced preview still runs in a `QThread`, full-crop preview is selectable for the current viewport focus, and full-resolution export runs in `ExportThread` with Phase 1 progress/cancel callbacks.


#### P2.4 - Apply flat modern minimal style
- Palette:
  - background `#0B0D10`, canvas `#101318`, panel `#151922`, border `#2A3038`, text `#E7ECF3`, muted `#8B96A6`, accent `#4F8CFF`.
- Reduce border radius to `4-6px`.
- Use compact toolbar buttons and thin separators.
- Remove huge title/header from working view; prioritize image canvas.

Execution:
- Serial

Isolation:
- Stylesheet/layout only.

Acceptance:
- UI visually prioritizes the image, not the settings chrome; style is flat and minimally rounded.

Status:
- Completed

Progress:
- Completed 2026-05-17: working view now prioritizes the canvas with a compact toolbar/right inspector/status bar and flat dark palette using 4-6px radii, thin borders, and the requested accent colors.


---







Current:
- No
### Phase 3 - AI/hint integration seam

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Add optional integration points without requiring model dependencies in default install/build.

Status:
- Completed


Progress:
- Started 2026-05-17 by deep-worker: created a non-git source backup under `.artifact/source-backup-phase3-*`; kept the phase limited to optional seams, capability checks, docs, and UI affordances with no AI dependency install/download/bundling.
- Completed 2026-05-17: added grayscale AI Alpha Hint merge through the classical pipeline, disabled BiRefNet/CorridorKey affordances backed by no-import capability stubs, external adapter contracts, documentation, and smoke coverage for no-dependency behavior.


#### P3.1 - Add alpha hint import/export
- Build on Phase 1/2 mask import/export instead of duplicating it.
- Add AI-specific meaning for a grayscale mask PNG as a coarse alpha hint.
- Merge hint with chroma trimap conservatively:
  - hint can protect foreground/core,
  - background still constrained by connected screen model unless user overrides.

Execution:
- Serial

Isolation:
- Engine hint merge + UI hint section.

Acceptance:
- Imported hint can preserve foreground regions that color keying alone would eat; exported matte is reusable.

Status:
- Completed

Progress:
- Completed 2026-05-17: `alpha_hint` is accepted by `process_key_image()`/`process_chroma_key()`, high-confidence hint pixels protect foreground/core without letting low hint values remove background, the hint appears in the `AI Hint` debug view, and current matte export remains the reusable grayscale output path.


#### P3.2 - Prototype optional BiRefNet alpha assist
- Keep behind optional dependency group or external model folder.
- Use AI at low/mid resolution to create a coarse alpha hint, then refine full-res with classical image keying.
- Do not run full-res AI over the whole image by default.

Execution:
- Serial

Isolation:
- Optional module, no default EXE bundling.

Acceptance:
- App works without AI installed; a stable external-adapter seam reports missing deps/model path clearly and can later route a configured local BiRefNet runner into the AI Alpha Hint path without bundling a runtime or weights.

Status:
- Completed

Progress:
- Completed 2026-05-17: `ai_assist.BiRefNetAlphaAssist` checks optional `torch`/`torchvision`/`transformers`, local `IMGKEY_BIREFNET_MODEL`, and optional `IMGKEY_BIREFNET_ADAPTER=module:function` without importing heavy runtimes at startup; UI remains disabled by default and docs state that no model download occurs.


#### P3.3 - Evaluate CorridorKey plugin path
- Treat CorridorKey as external/plugin-style backend only.
- Support passing original RGB + coarse alpha hint and receiving FG/alpha/processed outputs if user installs CorridorKey separately.
- Stop and ask before bundling weights/runtime or changing license/distribution assumptions.

Execution:
- Serial

Isolation:
- Research/prototype adapter only.

Acceptance:
- Clear documentation of install path, requirements, license constraints, and when CorridorKey should be used.

Status:
- Completed

Progress:
- Completed 2026-05-17: added `ai_assist.CorridorKeyPlugin` external adapter contract for `rgb_u8` + `alpha_hint_u8` -> foreground/alpha/processed outputs, with README stop notes for license/runtime/weights before any bundling or redistribution.


---







Current:
- No
### Phase 4 - Verification, packaging, and release rebuild

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own tests, docs, packaging spec, and build artifacts. Do not introduce new algorithm/UI scope except bug fixes from verification.

Status:
- Completed

Progress:
- Completed 2026-05-17: expanded smoke coverage for import/py_compile checks, refreshed diagnostics and visual artifacts under `.artifact/`, updated README/AGENTS/spec around v2 and non-AI packaging, built `dist/ImgKey.exe` from `ImgKey.spec`, and verified startup.


#### P4.1 - Expand automated verification
- Add smoke tests for:
  - green/blue/custom key,
  - connected-background preservation,
  - anti-aliased edge alpha,
  - despill/decontamination metric,
  - tile seam consistency,
  - import/py_compile.
- Keep large generated test images synthetic and temporary.
- If code is split into packages, replace narrow `py_compile` calls with `python -m compileall .` excluding `build/`, `dist/`, and `.artifact/`.

Execution:
- Serial

Isolation:
- `smoke_test.py`, test helpers.

Acceptance:
- `python smoke_test.py` and `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py` pass.

Status:
- Completed

Progress:
- Completed 2026-05-17: `smoke_test.py` covers green/blue/custom keying, connected-background preservation, anti-aliased edge alpha, despill, tiled consistency, optional AI no-dependency behavior, and import/py_compile checks; required smoke/compile/import commands passed.


#### P4.2 - Manual visual verification
- Create `.artifact/` verification outputs:
  - alpha view,
  - result over checkerboard/black/white/gray,
  - full-res crop before/after,
  - large-image export timing/memory note.
- Use the user's problematic poster-style case if available, otherwise generate a close synthetic case.

Execution:
- Serial

Isolation:
- `.artifact/` only, not committed unless user requests.

Acceptance:
- Visual checks show clean transparent background, preserved foreground details, no tile seams, and materially reduced keyed-color residue.

Status:
- Completed

Progress:
- Completed 2026-05-17: refreshed `.artifact/smoke-fixtures/` and generated `.artifact/phase4-verification/` with alpha/result/composite views plus large tiled timing/memory notes.


#### P4.3 - Rebuild Windows EXE
- Update `requirements.txt` and `README.md`.
- Choose one packaging source of truth:
  - update and use `ImgKey.spec`, or
  - delete/regenerate it intentionally and document the CLI build command.
- Build with PyInstaller default non-AI dependencies only.
- Verify `dist/ImgKey.exe` starts.
- Document optional AI install separately.

Execution:
- Serial

Isolation:
- Packaging/docs only.

Acceptance:
- New EXE exists, starts successfully, and default package size remains reasonable compared with bundling PyTorch/CUDA/models.

Status:
- Completed

Progress:
- Completed 2026-05-17: kept `ImgKey.spec` as packaging source of truth, excluded optional AI runtimes from the default build, built the onefile/windowed non-AI EXE, and verified it starts with a short start/stop probe.


---

## 6) Verification commands

Baseline and final:

```powershell
cd D:\keyphong
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py
# If split into packages during implementation:
# python -m compileall app.py keyer.py keying ui ai_assist.py smoke_test.py
python -c "import app, keyer; print('import ok')"
```

Packaging:

```powershell
cd D:\keyphong
python -m PyInstaller --noconfirm --clean ImgKey.spec
$p = Start-Process -FilePath ".\dist\ImgKey.exe" -PassThru; Start-Sleep -Seconds 6; if ($p.HasExited) { exit 1 } else { Stop-Process -Id $p.Id }
```

Manual UI verification:

- Load large image.
- Fit to view, zoom to 100%, pan, eyedropper sample.
- Switch `Result`, `Source`, `Alpha`, `AI Hint`, `Background Mask`, `Edge Mask`, `Despill Mask`, `Split Compare`.
- Run full-res crop preview.
- Export PNG and inspect over checkerboard/black/white/gray.

---

## 7) Immediate next step

Plan completed 2026-05-17. Continue to stop before bundling or redistributing CorridorKey, PyTorch/CUDA stacks, BiRefNet weights, or other optional AI runtimes.
