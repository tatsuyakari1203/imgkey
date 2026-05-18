# ImgKey CUDA DLL prototype

This folder contains the compact native CUDA backend prototype for deterministic
classical transition color repair. It intentionally does not use torch, CuPy, or
model runtimes.

## Phase 1 toolchain gate

Local gate result on the RTX 5060 Ti / compute 12.0 machine:

- `nvcc --version`: CUDA compilation tools 12.6, V12.6.20.
- `nvcc --list-gpu-arch`: supports through `compute_90`; it does **not** list
  `compute_120` / `sm_120`.
- CUDA 12.6 cannot emit native `sm_120` SASS, but a `compute_90` PTX forward-JIT
  probe compiled with `-gencode=arch=compute_90,code=compute_90` launched and
  completed on the local RTX 5060 Ti (`device0=NVIDIA GeForce RTX 5060 Ti
  compute=12.0`, result `2,3,4,5`).

The Phase 1 build therefore uses `sm_90` plus `compute_90` PTX when the installed
toolkit does not yet expose `compute_120`. A future CUDA Toolkit with
`compute_120` support can be used by the same script and will select native
`sm_120` automatically.

## Build

From the repo root:

```powershell
native/imgkey_cuda/build.ps1
```

The script:

- locates CUDA via `CUDA_PATH`, the v12.6 default install path, or `nvcc` on PATH;
- locates Visual Studio Build Tools via `vswhere` or known BuildTools paths;
- calls `VsDevCmd.bat`/`vcvars64.bat` and `nvcc` in the same `cmd.exe` invocation;
- logs `cl` and `nvcc` versions to `native/imgkey_cuda/build/build.log`;
- writes `imgkey_cuda.dll` under `native/imgkey_cuda/build/`.

Build outputs are generated artifacts and must stay untracked.

## ABI v1 contract

Exports use C linkage, `__declspec(dllexport)`, and `__cdecl`:

- `imgkey_cuda_version`
- `imgkey_cuda_device_count`
- `imgkey_cuda_last_error`
- `imgkey_cuda_transition_repair_v1`

`imgkey_cuda_transition_repair_v1` validates null pointers, dimensions, ABI
version/struct size, strides, and max tile size before launch. It synchronizes
before returning and reports status codes instead of throwing across the ABI.

The Python wrapper precomputes the final transition repair strength/eligibility
mask from the classical CPU-side tile inputs. The DLL does not decide
background/foreground membership; it only applies the compact transition color
repair math to pixels already marked eligible.
