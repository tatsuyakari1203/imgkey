# 08 - ImgKey Compact Classical GPU DLL

Date: 2026-05-18
Status: Completed
Owner: ImgKey Classical GPU Runtime
Scope: Replace the large torch-based GPU build with a compact NVIDIA CUDA DLL backend while keeping ImgKey no-AI.

---

## 1) Goal

Reduce GPU build size while preserving GPU acceleration for deterministic classical image math.

```text
Current GPU build: ~2.86 GiB because PyTorch CUDA is bundled.
Target GPU build: compact `ImgKey-GPU.exe` using a custom CUDA DLL, no torch, no AI, no model stack.
Default build: keep `ImgKey.exe` CPU classical/lightweight.
```

Runtime target:
- User needs only NVIDIA display driver.
- No CUDA Toolkit, Python packages, PyTorch, or model downloads on user machine.
- GPU acceleration remains optional/fallback-safe.

---

## 2) Context / evidence

- Current branch: `feature/birefnet-detail-keyer`, clean at v7 no-AI GPU commit `3c5ccd9`.
- Current built sizes:
  - `dist\ImgKey.exe`: ~100 MB.
  - `dist\ImgKey-GPU.exe`: ~2.86 GB.
- Local `torch` package is ~4.4 GB; size comes from torch/CUDA framework, not ImgKey algorithms.
- Local toolchain evidence:
  - CUDA Toolkit v12.6 exists at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`.
  - Visual Studio 2019 BuildTools exists at `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`.
  - `cl` is not on the default PATH; build scripts must call VS `vcvars64.bat`/`VsDevCmd.bat` before `nvcc`.
- `opencv-python` CUDA is not usable; current `cv2.cuda` device count is 0.
- CuPy remains a backup/prototype option only; custom DLL is the target for smallest EXE.

---

## 3) Architecture decision

Use a committed native CUDA source tree and build script:

```text
native/imgkey_cuda/
  imgkey_cuda.cu
  imgkey_cuda.h
  build.ps1
  README.md
```

Python loads the DLL lazily:

```text
gpu_accel.py
  ctypes load imgkey_cuda.dll only when GPU path is requested
  never imports torch
  falls back to CPU on missing DLL/CUDA/error
```

CUDA DLL API v1:
- expose `imgkey_cuda_version()`.
- expose `imgkey_cuda_device_count()` / basic probe.
- expose one shipped kernel matching v7 GPU path: transition color-tile repair / key-vector despill on tile-sized arrays.

Build/linking preference:
- Use `nvcc` with MSVC host compiler via VS BuildTools env.
- Prefer static CUDA runtime (`-cudart static`) if compatible to avoid bundling `cudart64_*.dll`.
- If static runtime is not viable, bundle only the minimal CUDA runtime DLL required by the custom DLL, not torch/cuDNN/cuBLAS.

---

## 4) Risks / constraints

- No AI/model packages or public AI wording may reappear.
- Do not add torch/cupy/pycuda/pyopencl to default or GPU requirements unless an explicit later fallback decision is made.
- Keep CPU path as correctness reference.
- GPU DLL work must be tile-bounded; no full-image float32 GPU retention.
- DLL ABI must validate shapes/strides/dtypes and return clear error codes; Python must not crash on bad input.
- Support RTX 5060 Ti / compute capability 12.0. If CUDA 12.6 cannot compile `sm_120`, compile PTX/fatbin strategy that runs on installed driver, or document/adjust to supported arch.
- Generated native build outputs stay untracked under ignored build folders.
- Stop and ask only if CUDA compiler cannot produce a working DLL with available toolchain.

---

## 5) Phases

Phase execution rule:
- Each phase is a clean commit boundary and should be pushed by planner after completion.
- Before commits inspect `git status --short --branch`, `git diff`, and `git log --oneline -10`.
- Never stage `build/`, `dist/`, `.artifact/`, native build outputs, wheels, or caches.

### Phase 1 - Toolchain gate and native CUDA DLL prototype

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `native/imgkey_cuda/`, `.gitignore`, and minimal ctypes probe/test scaffolding. Do not change PyInstaller packaging yet.

Status:
- Completed


#### P1.0 - Toolchain and architecture gate
- Before accepting the DLL prototype, inspect:

```powershell
nvcc --list-gpu-arch
nvcc --version
```

- Local GPU is RTX 5060 Ti / compute capability 12.0. CUDA 12.6 may only support through `compute_90`.
- Execute this milestone before writing/accepting the kernel implementation.
- If a newer CUDA Toolkit is required for `compute_120`/`sm_120` and is not already available locally, stop and report the exact installer/toolkit requirement before installing anything.
- Phase 1 is accepted only if one of these is true:
  - install/use a newer CUDA Toolkit after explicit approval if needed, or
  - build a PTX/fatbin strategy that actually runs the kernel on RTX 5060 Ti and passes the ctypes parity test.
- A DLL that merely builds is not enough; at least one real CUDA kernel invocation must pass on the RTX 5060 Ti.

Acceptance:
- Toolchain arch support or required toolkit upgrade is explicitly recorded.
- Execution does not proceed past P1.0 if a new toolkit install is required and not approved.

Status:
- Completed

Progress:
- Completed 2026-05-18. CUDA 12.6 (`nvcc` V12.6.20) lists through `compute_90`, not `compute_120`/`sm_120`. A `compute_90` PTX forward-JIT probe launched on RTX 5060 Ti / compute 12.0 and returned the expected kernel result, so no toolkit install was required for Phase 1.


#### P1.1 - Add buildable CUDA DLL source and script
- Create native source exposing a tiny C ABI:

```c
#define IMGKEY_CUDA_API __declspec(dllexport)
#define IMGKEY_CUDA_CALL __cdecl

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ImgKeyCudaStatus {
    IMGKEY_CUDA_OK = 0,
    IMGKEY_CUDA_INVALID_ARGUMENT = 1,
    IMGKEY_CUDA_NO_DEVICE = 2,
    IMGKEY_CUDA_LAUNCH_FAILED = 3,
    IMGKEY_CUDA_UNSUPPORTED_VERSION = 4
} ImgKeyCudaStatus;

typedef struct ImgKeyCudaTransitionParamsV1 {
    int struct_size;
    int version;
    int width;
    int height;
    int rgb_stride_bytes;
    int alpha_stride_bytes;
    int mask_stride_bytes;
    int out_stride_bytes;
    float foreground_reference_pull;
    float key_vector_despill;
    float preserve_foreground_luma;
    float transition_spill_threshold;
    unsigned char screen_r;
    unsigned char screen_g;
    unsigned char screen_b;
} ImgKeyCudaTransitionParamsV1;

IMGKEY_CUDA_API int IMGKEY_CUDA_CALL imgkey_cuda_version(void);
IMGKEY_CUDA_API int IMGKEY_CUDA_CALL imgkey_cuda_device_count(void);
IMGKEY_CUDA_API const char* IMGKEY_CUDA_CALL imgkey_cuda_last_error(void);
IMGKEY_CUDA_API ImgKeyCudaStatus IMGKEY_CUDA_CALL imgkey_cuda_transition_repair_v1(
    const ImgKeyCudaTransitionParamsV1* params,
    const unsigned char* rgb,
    const unsigned char* alpha,
    const unsigned char* transition_mask,
    const unsigned char* foreground_ref_rgb,
    const unsigned char* foreground_ref_valid,
    unsigned char* out_rgb,
    unsigned char* out_repair_mask
);

#ifdef __cplusplus
}
#endif
```

- ABI rules:
  - no C++ exceptions across boundary,
  - validate null pointers, dimensions, struct size/version, positive strides, and max tile size before launching,
  - synchronize before return so Python can safely read outputs,
  - store a thread-local or otherwise owned last-error string,
  - return status codes instead of crashing Python.
- Kernel contract:
  - Python wrapper precomputes `transition_mask` as the final repair/eligibility mask using current v7 CPU logic, including background/protected-core/manual/source-alpha gates.
  - DLL receives only the pixels already eligible for repair; it must not independently decide background/foreground membership.
  - Parity target is therefore the CPU transition repair helper for the same precomputed `transition_mask`, alpha, screen color, and foreground-reference buffers.
  - Future ABI versions may add more masks, but v1 stays compact and explicit.
- Export rules:
  - declarations and definitions must use `extern "C"`/C linkage,
  - verify unmangled exports with `dumpbin /exports` or equivalent; names must include `imgkey_cuda_version`, `imgkey_cuda_device_count`, and `imgkey_cuda_transition_repair_v1`.
- Implement the transition repair kernel for contiguous tile buffers first; Python can pass equal width-derived strides, but ABI must still validate stride arguments.
- Add `build.ps1` that:
  - locates VS BuildTools using `vswhere` or known path,
  - initializes MSVC x64 env via `VsDevCmd.bat`/`vcvars64.bat` in the same `cmd /c` invocation that runs `nvcc`, because `cl` is not on default PATH,
  - invokes `nvcc` from CUDA v12.6 or `CUDA_PATH`,
  - logs `cl` and `nvcc` versions,
  - writes output under `native/imgkey_cuda/build/`.
- Add `.gitignore` entries for native build outputs.

Acceptance:
- `native/imgkey_cuda/build.ps1` builds `imgkey_cuda.dll` locally.
- A tiny ctypes probe can load the DLL and report version/device count.
- Invalid-input ctypes/Python-wrapper tests cover null/shape/stride/version errors plus NumPy dtype, contiguity, dimensionality, and shape mismatches before unsafe ctypes calls.
- `dumpbin /exports` or equivalent proves exported names are unmangled.
- A real kernel run passes on RTX 5060 Ti / compute 12.0 using supported arch/PTX strategy.
- Built DLL and intermediates are untracked.

Status:
- Completed

Progress:
- Completed 2026-05-18. Added `native/imgkey_cuda/` source/header/build script/README. `build.ps1` locates VS BuildTools and CUDA, logs `cl`/`nvcc`, builds `imgkey_cuda.dll` under the ignored native build folder, and `dumpbin /exports` shows unmangled C ABI names.


#### P1.2 - Add Python loader and parity harness
- Update `gpu_accel.py` to prefer the CUDA DLL backend over torch.
- Keep torch backend disabled/removed for GPU build direction; if retained temporarily, it must be fallback-disabled by default and not bundled.
- Add smoke coverage that loads DLL if present and compares DLL output to CPU on a small tile.
- Skip cleanly if DLL is absent.

Acceptance:
- Importing `gpu_accel` does not load CUDA DLL or torch.
- DLL parity test passes when DLL exists.
- Existing CPU smoke tests pass when DLL is absent.

Status:
- Completed

Progress:
- Completed 2026-05-18. Replaced `gpu_accel.py` with a lazy ctypes CUDA DLL backend (no torch import), added Python validation and smoke parity coverage, and verified ctypes version/device_count plus real kernel parity on the local RTX 5060 Ti.


---






Current:
- No
### Phase 2 - Replace torch GPU path with compact DLL backend

Category:
- Deep

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `gpu_accel.py`, `gpu_runtime.py`, `keyer.py` dispatch hooks, `smoke_test.py` GPU parity/benchmark/probe expectations, and related tests. No packaging yet except removing torch assumptions from source.

Status:
- Completed


#### P2.1 - Make DLL backend the only active GPU implementation
- Remove active torch CUDA kernel usage from `gpu_accel.py`.
- Rewrite `gpu_runtime.py` so `python -m gpu_runtime --probe --json` reports compact DLL/nvidia-smi status and does not import torch.
- Rewrite `smoke_test.py --gpu-parity` and `--gpu-benchmark` so they use the DLL backend and never require torch.
- GPU availability/probe uses:
  - DLL load + `imgkey_cuda_device_count()`, and/or
  - `nvidia-smi` as user-facing status.
- Keep CPU fallback for missing DLL/no device/errors.

Acceptance:
- `python smoke_test.py --gpu-parity` uses DLL backend when built, otherwise skips/falls back with clear reason.
- No source import path requires torch.
- `python -m gpu_runtime --probe --json` works without torch installed.
- GPU benchmark output names the backend as compact CUDA DLL, not torch.

Status:
- Completed

Progress:
- Completed 2026-05-18. `gpu_runtime.py` now probes only the compact CUDA DLL plus `nvidia-smi`, reports DLL load/version/device-count and a transition repair kernel smoke, and has no torch import path. `gpu_accel.py`/`keyer.py` now expose the compact CUDA DLL as the only active GPU backend, with CPU fallback on missing DLL/no device/errors. `smoke_test.py --gpu-parity` uses the DLL backend directly and skips cleanly when unavailable.


#### P2.2 - Benchmark compact DLL backend
- Add/update benchmark output under `.artifact/gpu-benchmarks/`:
  - CPU vs CUDA DLL transition repair tile,
  - transfer overhead included,
  - output diff vs CPU.

Acceptance:
- DLL backend speed and parity are recorded.
- If slower than torch but still useful/small, document tradeoff; if slower than CPU, keep backend behind Auto fallback.

Status:
- Completed

Progress:
- Completed 2026-05-18. `python smoke_test.py --gpu-benchmark` writes `.artifact/gpu-benchmarks/summary.json` for the compact CUDA DLL backend. Local RTX 5060 Ti run recorded 1024×1024 transition repair direct CPU reference 1247.65 ms vs CUDA DLL 5.67 ms including DLL host/device transfers, max RGB/mask diff 0, plus dispatch CPU 1361.61 ms vs CUDA DLL 856.98 ms including transfers, max RGB diff 4. Auto/CPU fallback remains the policy for unavailable or failed GPU work.


---






Current:
- No
### Phase 3 - Compact GPU packaging

Category:
- Migration

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Own `ImgKey-GPU.spec`, workflow/build docs, and native build integration. Do not change algorithms except packaging fixes.

Status:
- Completed


#### P3.1 - Remove torch from GPU requirements/spec
- Update `requirements-gpu-runtime-cu128.txt` into a compact no-PyTorch GPU packaging note or delete/replace it if no Python package is required.
- Update `ImgKey-GPU.spec`:
  - include `imgkey_cuda.dll` from `native/imgkey_cuda/build/` or `IMGKEY_CUDA_DLL`,
  - include `cudart64_*.dll` only if `dumpbin`/dependency checks prove dynamic CUDA runtime is required,
  - exclude torch/torchvision/CUDA Python model stacks,
  - keep splash/progress.
- Update release workflow/docs to build native DLL before PyInstaller GPU build.
- Update `packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py` or equivalent startup/DLL search setup so frozen app finds bundled `imgkey_cuda.dll` and any required CUDA runtime DLL without relying on CUDA Toolkit PATH.

Acceptance additions:
- Run dependency inspection such as `dumpbin /dependents native\imgkey_cuda\build\imgkey_cuda.dll` or an equivalent tool and document required DLLs.
- Prove dependencies are either Windows/NVIDIA driver provided or explicitly bundled; do not rely on CUDA Toolkit/MSVC PATH.

Acceptance:
- `ImgKey-GPU.exe` archive has no torch package entries.
- GPU EXE includes only ImgKey app + compact CUDA DLL/minimal runtime DLLs.

Status:
- Completed

Progress:
- Completed 2026-05-18. GPU packaging no longer installs or bundles torch/Python CUDA stacks. `requirements-gpu-runtime-cu128.txt` is a no-op compatibility note, `ImgKey-GPU.spec` bundles `imgkey_cuda.dll` plus MSVC runtime DLLs, keeps splash/progress, and excludes torch/nvidia Python packages. Runtime hook now adds the frozen extraction root for bundled DLL discovery. Release workflow/docs now build the native DLL before the GPU PyInstaller build and document dependency inspection: static-runtime local DLL depends on `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, Windows/UCRT DLLs, and no `cudart64_*.dll`.


#### P3.2 - Build and size gate
- Build:

```powershell
native/imgkey_cuda/build.ps1
python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec
```

- Test:

```powershell
dist\ImgKey-GPU.exe --gpu-probe --json
```

Acceptance:
- GPU EXE starts and probes compact DLL backend.
- Target size is substantially smaller than torch build; preferred `<= 300 MB`, acceptable first pass `<= 500 MB` if static runtime inflates.
- Run a sanitized-path probe on the build machine, with CUDA Toolkit/MSVC paths removed from PATH, and verify `dist\ImgKey-GPU.exe --gpu-probe --json` still works.
- Prefer an actual clean Windows target with NVIDIA driver only when available; if unavailable, document sanitized-path result as local substitute.

Status:
- Completed

Progress:
- Completed 2026-05-18. Built `native/imgkey_cuda/build/imgkey_cuda.dll`, built `dist\ImgKey-GPU.exe`, verified archive contains `imgkey_cuda.dll` and no torch/nvidia/cudart entries, and verified built EXE probe with normal and sanitized PATH runs. Local size gate result: `dist\ImgKey-GPU.exe` is 102,044,207 bytes (97.32 MiB), SHA256 `893071a8468e665bf2ead25451aab23fc3b013dcce68b3024f681ec2c5a66210`.


---






Current:
- No
### Phase 4 - Final verification and cleanup

Category:
- Review-heavy

Executor:
- Deep-worker

Execution:
- Serial

Isolation:
- Verification/build only; no feature expansion.

Status:
- Completed


#### P4.1 - Verification floor
- Run:

```powershell
python smoke_test.py
python smoke_test.py --gpu-parity
python smoke_test.py --gpu-benchmark
python -m gpu_runtime --probe --json
python -m py_compile app.py keyer.py smoke_test.py gpu_runtime.py screen_analysis.py gpu_accel.py packaging/pyinstaller/rthooks/imgkey_cuda_runtime.py
python -c "import app, keyer; print('import ok')"
```

- Run dependency/no-AI guards and ensure default startup imports no torch.
- Build both EXEs:

```powershell
python -m PyInstaller --noconfirm --clean ImgKey.spec
python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec
```

Acceptance:
- `ImgKey.exe` remains lightweight/no GPU runtime.
- `ImgKey-GPU.exe` is compact and no-torch.
- CPU/GPU parity passes or cleanly falls back with documented reason.
- Final branch clean and ready for planner push.

Status:
- Completed

Progress:
- Completed 2026-05-18. Final verification passed: native DLL rebuild, CPU smoke, GPU parity, GPU benchmark, runtime probe, py_compile, import check, dependency/no-AI guards, default no-torch import guard, native DLL export/dependency inspection, default and GPU PyInstaller builds, EXE probes, sanitized-PATH GPU probe, GPU GUI lifetime probe, and archive checks. `dist\ImgKey.exe` is 100,128,472 bytes (95.49 MiB), SHA256 `fc7703425099e0200198718edb760a74e400774509beb81823adcec49caa329f`; archive has no GPU DLL, torch/nvidia Python stack, cudart, or model stack. `dist\ImgKey-GPU.exe` is 102,042,819 bytes (97.32 MiB), SHA256 `dbbe955a14f3941aa3ce70c702fb4f6a94000e17b8d073e4dcb6602a1f7b0efd`; archive contains `imgkey_cuda.dll` and no torch/nvidia Python stack/cudart/model entries. DLL exports are unmangled and depend only on MSVC/UCRT/Windows DLLs; no `cudart64_*.dll` dependency. Planner push remains pending per handoff.


---

## 6) Immediate next step

Phase 4 is complete and ready for planner push after the phase commit.

Current:
- No
