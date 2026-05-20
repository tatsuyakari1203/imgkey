# 10 - ImgKey D3D12/Vulkan Backend Refactor

Date: 2026-05-20
Status: Planned
Owner: ImgKey Native GPU + App Architecture
Scope: Refactor ImgKey away from god components and CUDA-specific GPU seams toward a backend-neutral, one-EXE CPU/GPU app with D3D12 as the primary Windows-native GPU backend and Vulkan as an optional portable backend.

---

## 1) Goal

Ship one maintainable ImgKey app that can run CPU-only or GPU-accelerated on a broad range of Windows GPUs while keeping the no-AI/no-heavy-dependency product surface.

Target architecture:

```text
ImgKey.exe
  Python/PySide shell
  CPU reference pipeline
  native imgkey_gpu.dll
    D3D12 compute backend         # primary long-term Windows GPU path
    Vulkan compute backend        # optional portable path after abstraction stabilizes
    CUDA backend compatibility    # optional NVIDIA fast/fallback path during migration
  automatic CPU fallback
```

The GPU rewrite is not just a shader port. It must also remove current architecture bottlenecks:
- app/UI god object,
- engine god module,
- CUDA-specific backend coupling,
- per-tile allocate/copy/sync/readback overhead,
- narrow GPU coverage of only transition RGB repair.

---

## 2) Architecture assessment

### Current god components

- `app.py` (~2.3k lines): UI construction, settings/defaults, image state, preview scheduling, worker threads, export, masks, GPU probe, canvas, slider widgets, event filtering. Risk: UI behavior, engine state, and threading are coupled in one class.
- `keyer.py` (~2.6k lines): settings/results, image I/O, global matte, screen probability, trimap/refine, local screen model, nearest-inner reference, transition alpha, color repair, tiled export, CUDA dispatch. Risk: algorithm phases, memory policy, tiling, and backend selection are interwoven.
- `smoke_test.py` (~5.3k lines): smoke tests, benchmark generators, diagnostics, packaging guards, UI probes, private helper tests. Risk: protects behavior well, but imports many private `keyer.py` internals, so refactor needs compatibility shims.
- `gpu_accel.py` (~770 lines): concrete compact CUDA backend, ctypes ABI, validation, CPU mirror, transition repair dispatch. Risk: it is not a backend abstraction and duplicates keyer color math.

### Current GPU bottlenecks

- GPU is optional/off by default and currently CUDA-only.
- Current DLL accelerates only transition RGB repair.
- Global matte, screen probability, local screen model, nearest-inner references, alpha recovery, guided refinement, morphology, and most color repair remain CPU.
- CUDA v1 falls back when `screen_tile` is present; default local screen modeling can therefore bypass GPU.
- Each GPU tile does allocate/copy/kernel/sync/copy/free; no persistent context, buffers, async queue, or multi-pass batching.

---

## 3) Technical direction

### Decision

- Do not make CUDA mandatory.
- Keep CUDA only as a migration/reference backend while D3D12/Vulkan mature.
- Build a backend-neutral native layer first; then add D3D12 and Vulkan behind the same ABI.
- D3D12 is the primary Windows-native backend because ImgKey is a Windows app and D3D12 is broadly available with Windows GPU drivers.
- Vulkan is valuable for portability and cross-vendor compute, but on Windows it adds loader/ICD/synchronization complexity; implement after D3D12 and only through the same backend contract.
- D3D11 compute may be used as a pragmatic compatibility fallback/prototype if D3D12 implementation risk blocks progress, but it is not the final target of this plan.

### Shader/tooling strategy

- Use HLSL as the shared shader source where possible.
- D3D12: precompile HLSL to DXIL/DXBC at build time; no runtime shader compiler in the EXE unless later justified.
- Vulkan: compile HLSL to SPIR-V with DXC (`-spirv`) after binding layout is stabilized.
- Prefer buffers/structured buffers for parity and simple ctypes-style tile data before optimizing texture paths.

### Native ABI direction

Create `native/imgkey_gpu/` with a stable C ABI:

```c
imgkey_gpu_version()
imgkey_gpu_probe_v1()
imgkey_gpu_create_context_v1()
imgkey_gpu_destroy_context_v1()
imgkey_gpu_process_color_tile_v1()
imgkey_gpu_process_tile_batch_v1()      // later, after MVP
imgkey_gpu_last_error()
```

Capabilities must be explicit:
- backend id: `d3d12_compute`, `vulkan_compute`, `cuda_compat`, `cpu_fallback`;
- device name/vendor;
- supports `screen_tile`;
- supports persistent sessions;
- max tile pixels;
- shader/kernel version;
- fallback reason.

---

## 4) Risks / constraints

- No AI/model runtime, Torch, CuPy, PyOpenCL, ONNX Runtime, or other heavy default dependency.
- CPU remains the correctness reference and final fallback.
- Keep large-image memory rules: source RGB as `uint8`, masks as `uint8`, no full-image float32 RGB, tile/ROI float only.
- Manual keep/remove/imported matte must remain authoritative.
- Transparent RGB zero invariant remains required.
- GPU backend must not require CUDA Toolkit, Vulkan SDK, DXC, or shader compiler on user machines.
- D3D12/Vulkan float math may differ from CPU; tests need tolerance but must catch visible regressions.
- Windows TDR risk: kernels must be tile-bounded and avoid long single dispatches.
- Do not break existing benchmark/diagnostic commands while extracting modules; keep re-export shims if tests still import private helpers.
- `.claude/`, `.artifact/`, `build/`, `dist/`, native build outputs, and caches must not be staged unless explicitly intended.

### Definitions / gates

- Compact size target: one-EXE CPU/GPU release should stay below `150 MB` preferred and `250 MB` hard stop unless the user explicitly approves a larger runtime.
- Representative large synthetic images: at minimum 4096x4096 and 8192x8192 generated fixtures with geometric benchmark features, flat/gradient key backgrounds, and tiled export enabled.
- Broad Windows GPUs: at minimum NVIDIA RTX local CUDA/D3D12 hardware, plus clean fallback on no-D3D12/no-Vulkan machines; release-quality claims require at least one AMD or Intel D3D12/Vulkan validation machine or an explicitly documented gap.
- Parity tolerance defaults:
  - alpha exact for RGB-only backend paths unless the kernel intentionally changes alpha,
  - alpha max diff `<= 1` for alpha-producing paths,
  - RGB max diff `<= 2` and p99 diff `<= 1` for deterministic color repair unless documented shader math requires a reviewed tolerance,
  - repair mask max diff `0`,
  - transparent RGB remains zero,
  - crop/tile parity must match existing CPU gates.
- Hardware-skip policy: missing D3D12/Vulkan hardware must produce clean skip/fallback in tests, but release promotion of a backend requires at least one real hardware pass.

---

## 5) Phases

Phase commit rule:
- Each phase is a commit boundary.
- Before commits inspect `git status --short --branch`, `git diff`, and `git log --oneline -10`.
- Do not stage generated artifacts/build outputs.

### Phase 1 - Baseline profiling and behavior freeze

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own benchmark/profiling instrumentation and documentation only. No behavior-changing refactor yet.

Status:
- Planned

Current:
- Yes

#### P1.1 - Add pipeline timing report
- Add timing instrumentation for:
  - image load/preview resize,
  - global matte,
  - screen model/local plate,
  - nearest-inner reference,
  - transition alpha recovery,
  - per-tile color render,
  - PNG encode/export,
  - GPU transfer/dispatch/readback when available.
- Add CLI/report output under `.artifact/perf/`.

Acceptance:
- A benchmark report identifies the top CPU and transfer bottlenecks on representative geometric and large synthetic images.
- Existing smoke/geometric/GPU tests still pass.

Verification:
- `python smoke_test.py`
- `python smoke_test.py --write-geometric-benchmark`
- `python smoke_test.py --gpu-parity`
- `python smoke_test.py --gpu-benchmark`
- `python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py`
- `python -c "import app, keyer; print('import ok')"`

Status:
- Planned

Current:
- Yes

#### P1.2 - Freeze visual and parity baselines
- Generate current geometric benchmark, tuning summary, GPU benchmark, transition diagnostics.
- Save only code/reporting changes; do not commit `.artifact/` outputs.

Acceptance:
- Baseline metrics and inspected visual failure modes are summarized in the phase commit message or docs.

Verification:
- `python smoke_test.py --tune-geometric-defaults`
- `python -m gpu_runtime --probe --json`
- `git diff --check`

Status:
- Planned

Current:
- No

---

### Phase 2 - Engine module extraction without behavior change

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `keyer.py` extraction and new engine modules. Keep public `keyer.py` facade and private compatibility aliases until tests are migrated.

Status:
- Planned

Current:
- No

#### P2.1 - Extract pure leaf modules
- Extract low-risk helpers first:
  - `imgkey_engine/color_math.py`,
  - `imgkey_engine/image_io.py`,
  - `imgkey_engine/tiling.py`,
  - `imgkey_engine/types.py`.
- Keep `keyer.py` imports/re-exports compatible.

Acceptance:
- No behavior change in smoke/geometric/GPU parity.

Status:
- Planned

Current:
- No

#### P2.2 - Extract matte/screen/reference/color phases
- Extract:
  - `matte.py`,
  - `screen_model.py`,
  - migrate/reuse `screen_analysis.py` logic so benchmark screen plate analysis does not diverge from engine screen plate logic,
  - `references.py`,
  - `transition_alpha.py`,
  - `color_repair.py`.
- Keep `process_key_image()` and `render_tiled_rgba()` facade stable.

Acceptance:
- `keyer.py` becomes a facade/orchestrator rather than a god module.
- Tests that import private helpers either use compatibility aliases or are migrated intentionally.

Verification:
- `python smoke_test.py`
- `python smoke_test.py --write-geometric-benchmark`
- `python smoke_test.py --gpu-parity`
- `python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py`
- `python -c "import app, keyer; print('import ok')"`
- dependency/no-AI guards
- `git diff --check`

Status:
- Planned

Current:
- No

---

### Phase 3 - UI/controller refactor without UX change

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `app.py` split and UI modules. Do not alter user-facing behavior except bug fixes required by extraction.

Status:
- Planned

Current:
- No

#### P3.1 - Split UI primitives and controllers
- Extract:
  - `ui/canvas.py` (`ImageCanvas`),
  - `ui/widgets.py` (`SliderRow` and reusable controls),
  - `ui/settings_mapper.py`,
  - `ui/preview_controller.py`,
  - `ui/export_controller.py`,
  - `ui/gpu_probe_controller.py`.
- Keep `app.py` as startup and `MainWindow` composition layer.

Acceptance:
- Viewer-first UX, spring-loaded space pan, pick behavior, default/reset behavior, preview/export flows remain unchanged.

Verification:
- `python smoke_test.py`
- `python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py`
- `python -c "import app, keyer; print('import ok')"`
- UI smoke/lifetime probe if PySide6 is available
- `git diff --check`

Status:
- Planned

Current:
- No

---

### Phase 4 - Backend-neutral native GPU abstraction

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `gpu_backend.py`, native ABI scaffolding, packaging hooks, and existing CUDA adapter wrapping. No D3D12/Vulkan shader port yet.

Status:
- Planned

Current:
- No

#### P4.1 - Add backend protocol and session lifecycle
- Replace direct `gpu_accel.process_color_tile_gpu()` calls with a backend registry/session API:
  - `probe_backends()`;
  - `select_backend(Auto/Off/Force, required_capabilities)`;
  - `begin_render(settings, image_shape)`;
  - `process_color_tile(...)`;
  - `end_render()`.
- Wrap current CUDA DLL as `CudaCompatBackend` behind this protocol.
- Define and test the native C ABI structs before D3D12 implementation:
  - versioned input/output structs with `struct_size` and `version`,
  - explicit strides/dimensions/dtypes,
  - owned/thread-local error strings,
  - status enum and fallback reasons,
  - capability flags (`constant_screen`, `screen_tile`, `persistent_session`, `tile_batch`, `alpha_write`, `rgb_only`),
  - no exceptions across native boundary,
  - CPU/fake backend ctypes tests for bad dtype/shape/stride/version/null-like inputs before unsafe calls.

Acceptance:
- Behavior remains equivalent to current CUDA/CPU fallback.
- Probe JSON reports backend list and selected backend.
- Fake backend and CUDA compatibility backend pass ABI validation tests.

Status:
- Planned

Current:
- No

#### P4.2 - Native toolchain and packaging decision gate
- Add a native toolchain probe/report for:
  - MSVC Build Tools,
  - Windows SDK and DirectX headers/libs,
  - DXC/FXC shader compiler availability for build time,
  - Vulkan headers/lib/loader strategy if Vulkan phase is enabled,
  - `dumpbin`/dependency audit availability.
- Decide whether to merge GPU packaging into `ImgKey.spec` only after backend size/dependency/fallback checks pass.
- Until then, keep current CPU-lightweight + optional GPU spec policy intact.
- If one-EXE policy is activated, update `AGENTS.md`, `docs/build-gpu.md`, README/release packaging docs, and workflow/spec names in the same phase or a dedicated docs commit.

Acceptance:
- Toolchain report identifies available/missing build dependencies.
- Packaging dependency audit proves no runtime dependency on SDKs/shader compilers.
- One-EXE merge is either explicitly approved by size/dependency/fallback evidence or deferred with current two-flavor release unchanged.

Verification:
- `python smoke_test.py`
- `python -m gpu_runtime --probe --json`
- packaging archive/dependency check if specs change
- sanitized-PATH EXE probe if specs change
- `git diff --check`

Status:
- Planned

Current:
- No

---

### Phase 5 - D3D12 compute MVP

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `native/imgkey_gpu/` D3D12 backend, shader build scripts, and Python backend adapter. CPU remains reference.

Status:
- Planned

Current:
- No

#### P5.1 - D3D12 identity backend
- Implement D3D12 device selection, adapter probe, context lifecycle, command queue/list/fence, descriptor/root signature/PSO setup.
- Add build script using MSVC + Windows SDK + DXC/FXC as appropriate.
- Precompile/embed shader bytecode.
- Add an identity/copy kernel first to prove upload/dispatch/readback and fallback/error paths before porting keying logic.

Acceptance:
- `python -m gpu_runtime --probe --json` lists D3D12 backend on capable GPUs and falls back cleanly otherwise.
- No runtime shader compiler dependency in the packaged app.
- Identity kernel returns exact byte-for-byte output for RGBA/tile buffers.

Status:
- Planned

Current:
- No

#### P5.2 - D3D12 constant-screen transition kernel
- Port current CUDA transition repair and screen-residue cleanup to D3D12 compute.

Acceptance:
- D3D12 parity against CPU passes benchmark tolerances.
- D3D12 does not regress current CUDA/CPU output visually on geometric/transition diagnostics.

Status:
- Planned

Current:
- No

#### P5.3 - D3D12 `screen_tile` / local plate support
- Add D3D12 support for per-pixel/per-tile local screen plate inputs.
- Ensure default local screen model does not force CPU fallback when backend supports this capability.

Acceptance:
- D3D12 default-quality path runs on cases that currently force CUDA fallback due to `screen_tile`.
- Local-screen geometric/crop/tile parity passes.

Verification:
- `python smoke_test.py`
- `python smoke_test.py --gpu-parity`
- `python smoke_test.py --gpu-benchmark`
- `python smoke_test.py --write-geometric-benchmark`
- `python -m gpu_runtime --probe --json`
- py_compile/import checks
- `git diff --check`

Status:
- Planned

Current:
- No

---

### Phase 6 - Full GPU tile color pipeline

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own tile color render graph and backend capabilities. Do not move global connected components/distance transform until color path is stable.

Status:
- Planned

Current:
- No

#### P6.1 - GPU linear conversion / unmix foundation
- Move the linear/sRGB conversion and base unmix math into backend-supported tile kernels.
- Use LUTs or matched shader math to satisfy parity thresholds.

Acceptance:
- RGB parity thresholds pass on geometric and transition diagnostics.

Status:
- Planned

Current:
- No

#### P6.2 - GPU despill/luma/color cleanup
- Move:
  - sRGB/linear conversion LUTs,
  - Vlahos clamp/unmix,
  - despill/decontaminate,
  - luminance protect,
  - alpha/RGB invariant enforcement.

Acceptance:
- GPU utilization improves in large export benchmark.
- CPU/GPU geometric parity remains within tolerance.
- Unsupported operations fall back by capability, not crash.

Status:
- Planned

Current:
- No

#### P6.3 - GPU nearest-inner/transition-reference integration
- Integrate existing nearest-inner reference inputs into the GPU tile pipeline.
- Do not port CPU distance-transform/label generation until the color path proves value.

Acceptance:
- Inner color pull and transition references match CPU within tolerance.

Status:
- Planned

Current:
- No

#### P6.4 - Screen tile/local plate + screen cleanup fusion
- Fuse local screen plate, screen cleanup, and transition repair into one backend-supported tile graph where possible.
- Upload tile inputs once, run multiple dispatches, read final RGBA once.

Acceptance:
- Default local screen model uses GPU for supported backends.
- Large synthetic export shows less transfer overhead than current one-kernel CUDA path.

Status:
- Planned

Current:
- No

#### P6.5 - Persistent buffers and batching
- Add render-session persistent buffers sized for max tile.
- Avoid per-tile allocation/free.
- Use async copy/dispatch/readback where beneficial without risking TDR.

Acceptance:
- Benchmarks show reduced transfer/dispatch overhead vs current compact CUDA path.

Status:
- Planned

Current:
- No

---

### Phase 7 - Vulkan backend behind the same ABI

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own Vulkan backend only; no algorithm changes unless parity requires shared shader fixes.

Status:
- Deferred until D3D12 gate passes

Current:
- No

#### P7.1 - Vulkan probe/context and SPIR-V shader path
- Hard stop before this milestone: do not start Vulkan until D3D12 MVP has passed parity, performance, packaging dependency audit, and backend ABI review.
- Runtime-load `vulkan-1.dll`.
- Enumerate physical devices/queues.
- Compile shared HLSL to SPIR-V at build time with DXC.
- Use validation layers in development only, never packaged.

Acceptance:
- Probe reports Vulkan availability or clear fallback reason.
- Packaged app does not require Vulkan SDK; only installed Vulkan driver/loader.

Status:
- Planned

Current:
- No

#### P7.2 - Vulkan parity for tile color backend
- Implement the same tile color operations supported by D3D12.
- Compare output vs CPU and D3D12.

Acceptance:
- Vulkan passes geometric/transition parity on a machine with Vulkan driver, or is marked experimental with explicit skip when no compatible device exists.

Status:
- Planned

Current:
- No

---

### Phase 8 - Final release hardening

Category:
- Review-heavy

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Verification/build/docs only.

Status:
- Planned

Current:
- No

#### P8.1 - Verification matrix
- Run:
  - `python smoke_test.py`,
  - `python smoke_test.py --write-geometric-benchmark`,
  - `python smoke_test.py --tune-geometric-defaults`,
  - backend parity/benchmark for CPU/D3D12/Vulkan/CUDA where available,
  - `python -m gpu_runtime --probe --json`,
  - py_compile/import checks,
  - no-AI/no-heavy-dep/default startup guards,
  - PyInstaller one-EXE build,
  - EXE probes with normal and sanitized PATH,
  - archive checks for no Torch/model/heavy runtime.

Acceptance:
- One `ImgKey.exe` is the intended release artifact.
- CPU fallback works on machines with no supported GPU.
- D3D12 works on supported Windows GPUs.
- Vulkan works where available or reports a clean skip/fallback.
- Docs explain backend priority, fallback behavior, and known limitations.

Status:
- Planned

Current:
- No

---

## 6) Immediate next step

Start Phase 1 with `deep-worker`: add profiling/timing reports and freeze current visual/performance baselines before refactoring. Do not begin D3D12/Vulkan implementation until god-component extraction and backend abstraction are in place.
