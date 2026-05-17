# 02 - ImgKey v3 UI/UX Redesign

Date: 2026-05-17
Status: Completed
Owner: ImgKey UI
Scope: Redesign the existing v2 app UI/UX into a friendlier flat-design image-keying workspace while preserving the current tuned keying behavior and default values.

---

## 1) Goal

Make ImgKey feel like a polished image/VFX tool: image-first canvas, clear controls, precise parameter editing, spring-loaded pan with Space, direct eyedropper sampling, and no zoom/pan reset while tuning sliders. Preserve the user’s current good parameter values as the new default preset.

---

## 2) Context / problem summary

- Current v2 engine is working and should not be reworked in this plan.
- Current UI is functional but still feels cramped and technical:
  - dense right inspector at fixed `346px`, small labels, too much vertical scrolling,
  - sliders hide spinbox arrows and only expose tiny numeric boxes plus reset,
  - holding `Pick` disables panning; user wants Space to temporarily drag the image,
  - changing sliders triggers preview refresh that can zoom out/reset the view,
  - eyedropper should point/sample directly on the image,
  - disabled AI controls are visible enough to distract,
  - defaults are not the user’s tuned values.
- Relevant local code:
  - `app.py::ImageCanvas` owns the `QGraphicsView` canvas.
  - `app.py::SliderRow` owns slider/numeric/reset UI.
  - `app.py::_build_ui`, `_build_toolbar`, `_build_inspector`, `_apply_theme` define layout/style.
  - `app.py::_set_current_source` and `on_preview_done` currently call canvas image setters during preview updates.

---

## 3) Constraints / non-goals

- Do not change keying algorithm semantics except setting UI defaults and wiring existing settings.
- Do not add AI/model dependencies.
- Do not break v2 smoke tests, export worker, alpha/debug views, or mask import/export.
- Keep flat/minimal dark style with small `4-6px` radii; do not return to large rounded cards.
- Preserve zoom/pan for ordinary preview refreshes; only reset on explicit user actions or image/preview-source changes.
- Because this folder may not be a git repo, create a timestamped source backup under `.artifact/` before modifying `app.py` deeply.

---

## 4) User-approved default values

Set these as the app-level **High Accuracy Graphic** defaults and reset targets. Prefer an `APP_DEFAULT_SETTINGS = KeySettings(...)` plus explicit default UI state in `app.py`; do not necessarily change low-level `keyer.py` defaults.

```python
key_mode = "Blue"
key_color = (30, 80, 235)
sample_size = 10
tolerance = 0.45
softness = 0.01
clip_background = 0.97
clip_foreground = 0.00
matte_gamma = 2.20
core_strength = 0.38
despeckle_min_area = 0
aggressive_interior_removal = True
edge_radius = 32
edge_softness = 0.00
erode_expand = -8
despill = 0.70
decontaminate = 0.50
luminance_restore = 0.76
```

Mapping notes:
- `key_mode` is UI-only; map it to `self.key_mode.setCurrentText("Blue")`, `self.settings.key_color = (30, 80, 235)`, and `auto_detect_key_color=False` unless user later chooses Auto.
- `edge_radius` maps to `KeySettings.edge_refine_radius=32` and compatibility `edge_blur=(32 - 1) / 4` where current UI uses `edge_blur` to derive radius.
- Presets, `current_settings()`, `self.settings.key_color`, and every reset button must all use the same app-level defaults.

---

## 5) Target UX model

```text
Top command bar:
  Open | Pick | Pan | Fit | 100% | View | BG | Export PNG

Main area:
  Large canvas with checkerboard/image result
  Small floating HUD: zoom %, preview mode, “Hold Space to pan” hint

Right inspector, resizable 400-460px:
  Screen
  Matte
  Edges
  Spill Cleanup
  Masks & Export
  Optional AI status collapsed/secondary

Bottom status:
  file, resolution, preview scale, cursor RGB/A, progress/cancel
```

Interaction rules:
- `Pick` means click samples color.
- Holding `Space` is a temporary hand/pan tool even while Pick is active.
- Releasing `Space` restores the previous tool without toggling toolbar buttons.
- Sliders/numeric edits update preview but do not change zoom/pan/center.
- Fit/100% buttons are the only normal controls that intentionally change zoom.
- Full Crop preview may reframe only when switching preview source/crop, not when editing a parameter.

---

## 6) Phases

### Phase 0 - Safety and UI baseline probe

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `.artifact/` backup/probe outputs and this plan file only, unless a tiny probe script is needed. Do not modify UI yet.

Status:
- Completed




#### P0.1 - Create backup and record current UI defaults
- Create `.artifact/source-backup-ui-v3-*` containing source files that will be edited.
- Capture current `KeySettings()` and current UI default values so regressions are obvious.
- Confirm current verification baseline passes before UI changes.

Execution:
- Serial

Isolation:
- `.artifact/`, optional notes only.

Acceptance:
- Backup exists; baseline commands pass:
  - `python smoke_test.py`
  - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
  - `python -c "import app, keyer; print('import ok')"`

Status:
- Completed


Progress:
- 2026-05-17: Created source backup at `.artifact/source-backup-ui-v3-20260517-020349` covering `app.py`, `keyer.py`, `smoke_test.py`, `ai_assist.py`, `README.md`, `AGENTS.md`, `ImgKey.spec`, and this plan.
- 2026-05-17: Captured current `KeySettings()` and fresh UI defaults at `.artifact/ui-v3-verification/baseline-ui-defaults-20260517-020349.md` / `.json`; verification report saved at `.artifact/ui-v3-verification/baseline-verification-20260517-020349.md`.
- 2026-05-17: Baseline verification passed: `python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, and `python -c "import app, keyer; print('import ok')"`.


---




Current:
- No
### Phase 1 - Interaction fixes: preserve view, Space-pan, eyedropper

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `ImageCanvas` behavior and preview image update plumbing in `app.py`. Do not alter inspector layout/style in this phase except minimal hooks.

Status:
- Completed


Progress:
- 2026-05-17: Implemented preserve-view preview refreshes, stable Full Crop cache for parameter-only edits, spring-loaded Space-pan, centered custom picker cursor, and separate cursor/sample/engine RGB status updates in `app.py`.
- 2026-05-17: Added and passed offscreen Phase 1 UI probe at `.artifact/ui-v3-verification/phase1_interaction_probe.py`; required smoke, compile, and import checks passed.

#### P1.1 - Stop slider refreshes from resetting zoom/pan
- Refactor `ImageCanvas.set_images()` / `_refresh_pixmap()` to support `reset_view=False` and preserve scene center + zoom transform.
- Fix the actual current reset path: `on_preview_done()` calls `_set_current_source()` and then `canvas.set_result()`, while `_set_current_source()` calls `canvas.set_images()` and `set_images()` currently resets/refits.
- Reset view only on image load, explicit Fit, explicit 100%, or deliberate preview source/crop switch.
- On ordinary preview result updates from slider changes, replace pixmap/result while preserving current transform and center.
- Remove behavior that refits merely because `_fit_mode` is true during result refresh unless the user explicitly requested Fit mode and the image dimensions changed.
- In `Full Crop` mode, cache/stabilize the current crop rect during parameter edits; recompute crop only when the user switches preview source/crop, uses an explicit refresh-crop action, or changes viewport intentionally before requesting Full Crop.

Execution:
- Serial

Isolation:
- `ImageCanvas`, `_set_current_source`, `on_preview_done`, preview mode wiring.

Acceptance:
- User can zoom to 100%, pan, change `Screen Tolerance`, and the canvas stays at the same zoom/center.

Status:
- Completed


Progress:
- 2026-05-17: `ImageCanvas.set_images()` and `_set_current_source()` now accept `reset_view`; ordinary `on_preview_done()` updates preserve transform/scene center, while image load and source/crop switches still reset intentionally. Full Crop reuses a cached crop rect across parameter-only previews.

#### P1.2 - Add spring-loaded Space-to-pan
- Add key press/release handling or an event filter so Space temporarily activates hand/pan.
- Add a minimal tool-state model before visual toolbar work: current persistent tool (`Pan` or `Pick`) plus transient Space-pan override.
- Ignore key auto-repeat.
- Do not steal Space while any interactive inspector/toolbar control has focus, including spinboxes, combos, sliders, buttons, and text-like widgets.
- In Pick mode, holding Space pans; releasing Space restores eyedropper cursor and click-to-sample behavior.
- Toolbar Pick checked state must not toggle when Space is held/released.

Execution:
- Serial

Isolation:
- `ImageCanvas`, `MainWindow` event filter/shortcuts only.

Acceptance:
- With Pick enabled: hold Space + drag pans; release Space + click samples. Without Pick: Space still works as temporary pan and restores normal pan/tool state.

Status:
- Completed


Progress:
- 2026-05-17: Added persistent `Pan`/`Pick` canvas tool state plus transient Space-pan via `MainWindow.eventFilter`; Space autorepeat is ignored and focused inspector/toolbar controls keep Space.

#### P1.3 - Improve eyedropper targeting
- Use a small custom `QCursor` with a centered hotspot; fall back to a precise/cross cursor only if the custom cursor cannot be created.
- Ensure `mousePressEvent` samples only when picker is active and Space-pan is not active.
- Keep `mapToScene()` / pixmap item mapping for accurate image coordinates.
- Update status/HUD to show sampled RGB and engine screen color after preview completes.
- Because `sample_size=10` uses a patch/median, display both cursor pixel RGB and sampled/key-color RGB so users understand exact pointer vs averaged sample behavior.

Execution:
- Serial

Isolation:
- `ImageCanvas` cursor/sampling and status messages only.

Acceptance:
- User can visually target a pixel on the image; sampled RGB in status matches the pixel under the cursor.

Status:
- Completed


Progress:
- 2026-05-17: Picker sampling is disabled while Space-pan is active; cursor RGB, sampled/key RGB, and engine screen RGB are surfaced separately in status UI.

---



Current:
- No
### Phase 2 - Precision controls and defaults

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own `SliderRow`, app-level defaults, preset/reset behavior, and related tests/probes. Do not change engine math.

Status:
- Completed


Progress:
- 2026-05-17: Redesigned `SliderRow` with label/value display, minus/plus step buttons, wider non-keyboard-tracking numeric boxes, reset controls, synchronized typed/slider values, and Ctrl/Shift 10× nudge tooltip.
- 2026-05-17: Added `APP_DEFAULT_SETTINGS`/`APP_DEFAULT_KEY_MODE` tuned Blue defaults, initialized fresh UI to those values, mapped edge radius 32 to `edge_refine_radius=32` and `edge_blur=7.75`, made High Accuracy match tuned defaults, and kept Fast/Clean presets available.
- 2026-05-17: Added and passed offscreen Phase 2 probe at `.artifact/ui-v3-verification/phase2_defaults_controls_probe.py`; required smoke, compile, and import checks passed.




#### P2.1 - Add step buttons to slider rows
- Redesign each `SliderRow` as:
  - label + current value,
  - `−` button, slider, `+` button,
  - wider numeric box, reset button.
- Use explicit `nudge(-1)` / `nudge(+1)` controls with the row’s configured step.
- Support accelerated nudge if practical (`Ctrl` or `Shift` = 10x step); document tooltip.
- Use `setKeyboardTracking(False)` for spinboxes to avoid previewing every partial typed number.
- Keep typed values and slider position synchronized.

Execution:
- Serial

Isolation:
- `SliderRow` only plus stylesheet for step/reset buttons.

Acceptance:
- Every slider can be adjusted precisely via minus/plus buttons, typed numeric input, slider drag, and reset.

Status:
- Completed


Progress:
- 2026-05-17: Every `SliderRow` now supports precise minus/plus nudge, typed numeric entry, slider drag, and reset while keeping displays synchronized.




#### P2.2 - Preserve tuned defaults and reset targets
- Add `APP_DEFAULT_SETTINGS` in `app.py` matching the user-approved values.
- Set initial combo/slider state to:
  - `Screen Mode = Blue`, sample size 10,
  - exact values listed in section 4,
  - policy `Aggressive Interior Removal`.
- Update `High Accuracy` preset to match the same values.
- Ensure each row’s reset button returns to the new tuned default, not old `KeySettings()` values.
- Update `current_settings()`, `apply_preset()`, `set_key_color()`, combo initialization, and `self.settings.key_color` together so no old green/default values leak into UI state.
- Keep `Fast`/`Clean` presets available but do not override the initial tuned default.

Execution:
- Serial

Isolation:
- `MainWindow.__init__`, defaults/preset methods, inspector initialization.

Acceptance:
- Fresh launch shows the exact values from section 4 before opening an image; reset buttons restore those defaults.

Status:
- Completed


Progress:
- 2026-05-17: Fresh launch, High Accuracy preset, and all row reset targets now use the tuned Blue defaults from section 4.




---



Current:
- No
### Phase 3 - Visual redesign and inspector UX

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own layout, inspector grouping, toolbar/HUD/status styling in `app.py`. Do not change engine settings beyond already defined defaults.

Status:
- Completed


Progress:
- 2026-05-17: Rebuilt the top command bar, added Pan and Export PNG actions, added a canvas HUD for zoom/preview/Space-pan hint, widened/reorganized the inspector, moved optional AI adapter runtime status behind collapsed secondary controls, refreshed the flat dark QSS palette/states, and added `.artifact/ui-v3-verification/phase3_visual_layout_probe.py`.
- 2026-05-17: Phase 3 verification passed: `python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, `python -c "import app, keyer; print('import ok')"`, Phase 1 probe, Phase 2 probe, and Phase 3 visual layout probe.






#### P3.1 - Rebuild command bar and canvas HUD
- Make top toolbar read like a product command bar:
  - `Open`, `Pick`, `Pan`, `Fit`, `100%`, `View`, `BG`, `Export PNG`.
- Wire command-bar `Export PNG` to the same enabled/disabled state and export handler as the inspector export control.
- Add a small flat HUD overlay or status strip near canvas with:
  - zoom percent,
  - preview mode (`Proxy` / `Full Crop`),
  - hint: `Hold Space to pan`.
- Keep export accessible without scrolling the inspector.

Execution:
- Serial

Isolation:
- Toolbar/canvas/status UI only.

Acceptance:
- Main actions are discoverable at the top; user does not need to scroll to export.

Status:
- Completed

Progress:
- 2026-05-17: Top toolbar now orders primary actions as `Open`, `Pick`, `Pan`, `Fit`, `100%`, `View`, `BG`, `Export PNG`; toolbar export mirrors inspector export enablement/handler, and the canvas HUD shows zoom, preview mode, and `Hold Space to pan` without intercepting canvas input.




#### P3.2 - Redesign inspector readability
- Increase inspector width to a resizable target of `400-460px` instead of fixed `346px`.
- Rename/organize sections:
  - `Screen`, `Matte`, `Edges`, `Spill Cleanup`, `Masks & Export`.
- Put disabled AI adapter status behind a collapsed/secondary details area or move it below normal workflow.
- Improve spacing: 16px panel margins, 12px section gaps, 8px control gaps.
- Use readable 13px controls, 14px section titles, 16-18px panel title.

Execution:
- Serial

Isolation:
- Inspector layout/style only.

Acceptance:
- Inspector is readable, less cramped, sections are easy to scan, and important export/mask actions remain visible.

Status:
- Completed

Progress:
- 2026-05-17: Inspector is resizable in the 400-460px target range, uses 16px panel margins and readable section spacing/type, is grouped as `Screen`, `Matte`, `Edges`, `Spill Cleanup`, `Masks & Export`, and keeps optional AI adapter status collapsed below the normal workflow.




#### P3.3 - Refine flat visual language
- Maintain dark flat design:
  - main `#0B0D10`, canvas `#101318`, panel `#151922`, section surface `#181D26`, border `#2A3038`, text `#E7ECF3`, muted `#9AA6B6`, accent `#4F8CFF`.
- Use only `4-6px` radius.
- Add consistent hover/pressed/focus states.
- Style step buttons, combo boxes, numeric fields, toolbar actions, status bar, splitter, and scrollbars.
- Avoid glossy gradients and oversized rounded cards.

Execution:
- Serial

Isolation:
- Stylesheet and object names only.

Acceptance:
- UI looks cohesive and friendlier while remaining flat/minimal.

Status:
- Completed

Progress:
- 2026-05-17: Updated the stylesheet to the flat dark palette, 4-6px radii, and consistent hover/pressed/focus states for toolbar actions, step/reset buttons, combos, numeric fields, sliders, status bar, splitter, and scrollbars.




---


Current:
- No
### Phase 4 - Verification and rebuild

Category:
- Standard

Executor:
- Worker

Execution:
- Serial

Isolation:
- Own verification artifacts, docs updates if needed, and packaging. Do not add UI feature scope except fixing regressions found in verification.

Status:
- Completed

Current:
- No

Progress:
- 2026-05-17: Phase 4 automated checks passed: `python smoke_test.py`, `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`, `python -c "import app, keyer; print('import ok')"`, Phase 1/2/3 UI probes, and the new Phase 4 final UI probe.
- 2026-05-17: Added `.artifact/ui-v3-verification/phase4_final_ui_probe.py` covering exact defaults, High Accuracy preset, reset targets, SliderRow plus/minus behavior, transform preservation, Full Crop stability, Space-pan behavior, and export command availability.
- 2026-05-17: Recorded the UX checklist at `.artifact/ui-v3-verification/phase4-manual-ux-checklist-20260517.md`; updated `AGENTS.md` with v3 defaults, Space-pan behavior, and the aggressive interior-removal default.
- 2026-05-17: Rebuilt the default non-AI EXE with `ImgKey.spec`; `D:\keyphong\dist\ImgKey.exe` starts under the short start/stop probe and is `99,811,847` bytes (`95.19 MiB`).

#### P4.1 - Automated and UI behavior verification
- Run baseline commands:
  - `python smoke_test.py`
  - `python -m py_compile app.py keyer.py smoke_test.py ai_assist.py`
  - `python -c "import app, keyer; print('import ok')"`
- Add or run an offscreen UI probe if practical to verify:
  - default values are exact,
  - `SliderRow` nudge/reset works,
  - preview updates can replace images without forced fit.
- Save screenshots/probe artifacts under `.artifact/ui-v3-verification/` only.
- Make UI probes required unless the local environment cannot instantiate Qt; if blocked by environment, report it explicitly as an environment blocker, not a UI pass.
- Required probe assertions:
  - exact default values and High Accuracy preset values,
  - reset buttons return to the tuned defaults,
  - `SliderRow` minus/plus nudge changes by the configured step,
  - canvas transform scale and scene center survive the preview result update path,
  - Full Crop bounds remain stable during parameter-only edits,
  - Space-pan behavior works with picker active and does not fire while inspector controls have focus.

Execution:
- Serial

Isolation:
- Tests/probes/artifacts only.

Acceptance:
- Automated checks pass and UI probe confirms defaults/precision control behavior.

Status:
- Completed

Progress:
- 2026-05-17: Required automated commands and offscreen UI probes passed, including `.artifact/ui-v3-verification/phase4_final_ui_probe.py`.




#### P4.2 - Manual UX checklist and EXE rebuild
- Manual checklist:
  - fresh launch shows Blue/default values exactly,
  - zoom to 100%, pan, adjust multiple sliders: zoom/center do not reset,
  - Pick enabled + hold Space + drag pans, release Space + click samples,
  - eyedropper cursor targets image directly,
  - plus/minus buttons step values correctly,
  - export remains easy to find.
- Rebuild default non-AI EXE with `ImgKey.spec`.
- Verify EXE starts.
- If PyInstaller or packaging environment fails independently of the app, report it as an environment blocker and still return UI verification status clearly.
- Update `AGENTS.md` after successful implementation to mention v3 UI defaults, Space-pan behavior, and the user-approved aggressive interior-removal default.

Execution:
- Serial

Isolation:
- Packaging and `.artifact/` verification only.

Acceptance:
- `D:\keyphong\dist\ImgKey.exe` is rebuilt and starts; UI checklist passes.

Status:
- Completed

Progress:
- 2026-05-17: Manual UX checklist record saved under `.artifact/ui-v3-verification/`; PyInstaller rebuild passed; rebuilt EXE start/stop probe passed; `AGENTS.md` updated.




---

## 7) Verification commands

```powershell
cd D:\keyphong
python smoke_test.py
python -m py_compile app.py keyer.py smoke_test.py ai_assist.py
python -c "import app, keyer; print('import ok')"
python -m PyInstaller --noconfirm --clean ImgKey.spec
$p = Start-Process -FilePath ".\dist\ImgKey.exe" -PassThru; Start-Sleep -Seconds 6; if ($p.HasExited) { exit 1 } else { Stop-Process -Id $p.Id }
```

---

## 8) Immediate next step

Completed 2026-05-17. No further plan phases remain.
