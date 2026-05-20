# ImgKey native GPU build and packaging notes

ImgKey now uses one primary Windows release artifact: `ImgKey.exe`. The EXE is a
one-file PyInstaller bundle that contains the classical CPU path plus the compact
native D3D12 backend DLL (`imgkey_gpu.dll`). CPU fallback is automatic when the
native DLL, D3D12 device, or requested backend capability is unavailable.

`ImgKey-GPU.exe` remains only as a legacy/development CUDA compatibility build for
comparing the old compact CUDA transition-repair DLL. Do not publish it as the
primary release artifact.

## 1. Primary `ImgKey.exe` CPU+D3D12 release build

Build the native D3D12 backend first, then build the one-file app:

```powershell
python -m venv .venv-classical
.\.venv-classical\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
pwsh -NoProfile -ExecutionPolicy Bypass -File native/imgkey_gpu/build.ps1 -Clean
python smoke_test.py
python -m gpu_runtime --probe --json
python -m PyInstaller --noconfirm --clean ImgKey.spec
.\dist\ImgKey.exe --gpu-probe --json
```

`ImgKey.spec` is the release source of truth. It explicitly bundles
`native/imgkey_gpu/build/imgkey_gpu.dll`, includes only MSVC runtime DLLs that the
native DLL actually imports, and rejects runtime imports for CUDA, Vulkan SDK,
shader compiler, or Python GPU package stacks. The current local D3D12 DLL imports
only Windows platform DLLs (`d3d12.dll`, `dxgi.dll`, `KERNEL32.dll`), so no MSVC
runtime DLLs are bundled for the D3D12 backend.

The primary bundle must stay under the one-EXE size gate: `150 MB` preferred and
`250 MB` hard stop unless the user explicitly approves a larger runtime.

## 2. D3D12 backend DLL

`native/imgkey_gpu/build.ps1` compiles HLSL with DXC/FXC at build time, embeds the
shader bytecode into `imgkey_gpu.dll`, and writes all generated files under ignored
`native/imgkey_gpu/build/`. End-user machines must not need DXC, FXC, Windows SDK,
or a runtime shader compiler.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File native/imgkey_gpu/build.ps1 -Clean
python -m gpu_runtime --probe --json
python smoke_test.py --gpu-parity
python smoke_test.py --gpu-benchmark
```

Backend status:

- `d3d12_compute` is the primary GPU backend and supports constant screen color,
  per-pixel `screen_tile` local plates, persistent sessions, RGB-only transition
  repair, and the fused full-color tile path.
- CPU remains the correctness reference and final fallback.
- Native D3D12 calls are capped at `512*512` pixels to avoid Windows TDR risk; the
  Python session splits larger full-color tiles into persistent-buffer
  subdispatches.

## 3. Vulkan status

Vulkan is runtime-probed but the native tile backend is deferred until Vulkan SDK
headers and `vulkan-1.lib` are available without adding packaged runtime
dependencies. The app may runtime-load the installed Vulkan loader/driver for
diagnostics, but packaged ImgKey must not include the Vulkan SDK, DXC, shader
compiler binaries, validation layers, or a fake Vulkan tile implementation.

Run the gate report with:

```powershell
python -m gpu_runtime --probe --json
```

The JSON includes:

- `backend_registry.backends` and `backend_registry.selected_backend`;
- `native_toolchain.components.msvc` for MSVC Build Tools;
- `native_toolchain.components.windows_sdk` for Windows SDK DirectX headers/libs;
- `native_toolchain.components.shader_compilers` for build-time DXC/FXC;
- `native_toolchain.components.vulkan` for the Vulkan SDK header/import-lib/loader
  gate;
- `vulkan_runtime` and `backend_registry.backends[].runtime_probe` for a no-SDK
  runtime-load probe of the installed Vulkan loader/driver;
- `native_toolchain.components.dependency_audit` for `dumpbin` or `llvm-objdump`;
- `native_toolchain.packaging_decision` for the one-EXE release policy.

Current local gate result: `vulkan-1.dll` is present and exposes one
compute-capable NVIDIA device, but Vulkan SDK headers and `vulkan-1.lib` are
missing. This is a clean deferred state: D3D12 remains primary and CPU fallback
remains active.

## 4. Legacy/dev CUDA compatibility build

The old compact CUDA DLL backend is still useful for compatibility tests and
benchmarks, but it is not the primary release packaging path.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
native/imgkey_cuda/build.ps1
python -m gpu_runtime --probe --json
python -m PyInstaller --noconfirm --clean ImgKey-GPU.spec
.\dist\ImgKey-GPU.exe --gpu-probe --json
```

`requirements-gpu-runtime-cu128.txt` is a no-op compatibility note. There are no
extra Python GPU packages for this flavor; build
`native/imgkey_cuda/build/imgkey_cuda.dll` before running PyInstaller. Set
`IMGKEY_CUDA_DLL` only when packaging a DLL from a non-default path.

## 5. RTX 5060 Ti / Blackwell constraints

- CUDA Toolkit 12.6 can build the legacy CUDA DLL with `sm_90` plus `compute_90`
  PTX forward-JIT, which has been verified on the local RTX 5060 Ti / compute
  12.0 machine.
- A future CUDA Toolkit with `compute_120` support can be used by
  `native/imgkey_cuda/build.ps1`; the script selects it automatically when
  `nvcc --list-gpu-arch` exposes it.
- The primary `ImgKey.exe` does not require a CUDA Toolkit, CUDA DLL, or NVIDIA
  CUDA runtime on the target machine.

## 6. Native DLL dependency inspection

Inspect every native DLL build before packaging:

```powershell
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"
cmd /d /s /c "call `"$vs`" -arch=amd64 -host_arch=amd64 >nul && dumpbin /dependents native\imgkey_gpu\build\imgkey_gpu.dll"
cmd /d /s /c "call `"$vs`" -arch=amd64 -host_arch=amd64 >nul && dumpbin /dependents native\imgkey_cuda\build\imgkey_cuda.dll"
```

Current local D3D12 backend dependents:

```text
d3d12.dll
dxgi.dll
KERNEL32.dll
```

Current local CUDA compatibility backend dependents:

```text
MSVCP140.dll
KERNEL32.dll
VCRUNTIME140.dll
VCRUNTIME140_1.dll
api-ms-win-crt-runtime-l1-1-0.dll
api-ms-win-crt-stdio-l1-1-0.dll
api-ms-win-crt-heap-l1-1-0.dll
api-ms-win-crt-convert-l1-1-0.dll
api-ms-win-crt-string-l1-1-0.dll
api-ms-win-crt-time-l1-1-0.dll
```

Packaging policy:

- `ImgKey.spec` bundles `imgkey_gpu.dll` and only MSVC runtime DLLs imported by
  that DLL. It must not bundle CUDA, DXC, Vulkan SDK files, Torch/model runtimes,
  CuPy, ONNX Runtime, PyOpenCL, or Python GPU package libraries.
- `ImgKey-GPU.spec` bundles `imgkey_cuda.dll`, the needed MSVC runtime DLLs, and a
  dynamic `cudart64_*.dll` only if the CUDA DLL import table requires it.

## 7. Clean-target testing expectations

Test the generated primary EXE on a clean Windows x64 target with no Python, no
pip packages, no CUDA Toolkit, no Windows SDK, and no shader compiler on PATH.

1. `ImgKey.exe --gpu-probe --json` reports `d3d12_compute` selected on supported
   hardware or a clear D3D12 unavailable/fallback reason with CPU available.
2. A sanitized-PATH probe still loads the bundled `imgkey_gpu.dll` and does not
   require DXC, FXC, Vulkan SDK files, CUDA Toolkit files, or Python GPU packages.
3. `ImgKey.exe` opens and survives a GUI lifetime smoke, then passes a manual
   import/export path without torch/model files.
4. Archive checks find `imgkey_gpu.dll` and no Torch, model, CuPy, ONNX, PyOpenCL,
   DXC, Vulkan SDK, or CUDA toolkit payloads in the primary bundle.
5. Confirm `build/`, `dist/`, wheels, caches, native build outputs, and
   `.artifact/` outputs remain ignored and are not committed.
