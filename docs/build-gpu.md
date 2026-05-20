# ImgKey GPU build notes

ImgKey currently has two public Windows build flavors. Keep them separated so the default app remains lightweight while native GPU backend packaging evidence is gathered.

## 1. Default `ImgKey.exe`

No torch and no CUDA runtime.

```powershell
python -m venv .venv-classical
.\.venv-classical\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python smoke_test.py
python -m PyInstaller --noconfirm --clean ImgKey.spec
```

`ImgKey.spec` is the default release source of truth and keeps optional GPU packages out of the lightweight bundle.

## 2. GPU runtime `ImgKey-GPU.exe`

Includes the custom `imgkey_cuda.dll` backend for compact GPU acceleration/probe support. The GPU spec bundles `imgkey_cuda.dll` plus required MSVC runtime DLLs, leaves PyInstaller data files empty, excludes Python CUDA package stacks, and uses PyInstaller's boot splash (`packaging/imgkey_splash.png`) so onefile extraction shows visible progress before the Qt UI can start.

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

`requirements-gpu-runtime-cu128.txt` is now a no-op compatibility note. There are no extra Python GPU packages for this flavor; build `native/imgkey_cuda/build/imgkey_cuda.dll` before running PyInstaller. Set `IMGKEY_CUDA_DLL` only when packaging a DLL from a non-default path.

## 3. D3D12 compute MVP DLL

Phase 5 adds the backend-neutral native `imgkey_gpu.dll` with backend id
`d3d12_compute`. It is not merged into either PyInstaller spec yet; build and
probe it locally while the packaging gate remains open.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File native/imgkey_gpu/build.ps1 -Clean
python -m gpu_runtime --probe --json
python smoke_test.py --gpu-parity
python smoke_test.py --gpu-benchmark
```

`native/imgkey_gpu/build.ps1` compiles HLSL with DXC/FXC at build time, embeds the
shader bytecode into the DLL, and writes all generated files under ignored
`native/imgkey_gpu/build/`. End-user machines must not need DXC, FXC, Windows SDK,
or a runtime shader compiler. The MVP D3D12 transition shader is capped at
`640*640` pixels per tile to avoid TDR; larger tiles fall back to CPU until the
full GPU tile pipeline phase adds better splitting/fusion.

## RTX 5060 Ti / Blackwell constraints

- CUDA Toolkit 12.6 can build the DLL with `sm_90` plus `compute_90` PTX forward-JIT, which has been verified on the local RTX 5060 Ti / compute 12.0 machine.
- A future CUDA Toolkit with `compute_120` support can be used by `native/imgkey_cuda/build.ps1`; the script selects it automatically when `nvcc --list-gpu-arch` exposes it.
- The target machine needs an NVIDIA display driver. A local CUDA Toolkit install is not required for the packaged EXE.
- Validate with `python -m gpu_runtime --probe --json` before packaging and with `ImgKey-GPU.exe --gpu-probe --json` after packaging.

## Native DLL dependency inspection

Inspect every DLL build before packaging:

```powershell
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"
cmd /d /s /c "call `"$vs`" -arch=amd64 -host_arch=amd64 >nul && dumpbin /dependents native\imgkey_cuda\build\imgkey_cuda.dll"
```

Current local static-runtime build dependents:

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

- `MSVCP140.dll`, `VCRUNTIME140.dll`, and `VCRUNTIME140_1.dll` are bundled by `ImgKey-GPU.spec`.
- `KERNEL32.dll` and `api-ms-win-crt-*` entries are Windows/UCRT platform DLLs.
- `cudart64_*.dll` is not listed for the static-runtime build and is not bundled. If a future dynamic-runtime build lists it, place the verified DLL beside `imgkey_cuda.dll` or set `IMGKEY_CUDA_RUNTIME_DLLS` before running PyInstaller.

## Clean-target testing expectations

Test generated EXEs on a clean Windows x64 target with an NVIDIA driver only: no Python, no pip packages, and no CUDA Toolkit on PATH.

1. `ImgKey.exe` opens and passes a manual import/export smoke path without torch files.
2. `ImgKey-GPU.exe --gpu-probe --json` reports compact CUDA DLL availability or a clear driver/runtime error, with extraction splash/progress visible during onefile startup.
3. Confirm `build/`, `dist/`, wheels, caches, and `.artifact/` outputs remain ignored and are not committed.

## Phase 4/5/7 native backend gate

`gpu_backend.py` now owns the backend-neutral Python protocol and wraps the
existing compact CUDA DLL as the `cuda_compat` backend. `native/imgkey_gpu/`
defines the C ABI contract and ships a D3D12 compute backend for identity,
transition repair, `screen_tile` local-plate inputs, and the fused full color tile
path. Vulkan remains optional/experimental behind the same backend registry gate.

Run the gate report with:

```powershell
python -m gpu_runtime --probe --json
```

The JSON includes:

- `backend_registry.backends` and `backend_registry.selected_backend`;
- `native_toolchain.components.msvc` for MSVC Build Tools;
- `native_toolchain.components.windows_sdk` for Windows SDK DirectX headers/libs;
- `native_toolchain.components.shader_compilers` for DXC/FXC build-time shader compilers;
- `native_toolchain.components.vulkan` for the Vulkan SDK header/import-lib/loader
  gate;
- `vulkan_runtime` and `backend_registry.backends[].runtime_probe` for a
  no-SDK runtime-load probe of the installed Vulkan loader/driver;
- `native_toolchain.components.dependency_audit` for `dumpbin` or `llvm-objdump`;
- `native_toolchain.packaging_decision` for the one-EXE merge gate.

Phase 7 gate result on the current local machine: `vulkan-1.dll` was present,
but Vulkan SDK headers and `vulkan-1.lib` were missing, so no Vulkan native tile
backend was built. This is a clean deferred state: D3D12 remains the primary GPU
backend, CPU remains the fallback, and packaged apps must not require the Vulkan
SDK, DXC, shader compilers, or validation layers. A later Vulkan implementation
must compile HLSL to SPIR-V at build time with DXC only after the SDK header/import
library gate is satisfied.

Current decision: one-EXE CPU/GPU merging is **deferred**. Keep `ImgKey.exe` as
the lightweight default build and `ImgKey-GPU.exe` as the optional compact CUDA
flavor until D3D12 packaging size measurements, dependency audit, sanitized-PATH
EXE fallback evidence, and explicit approval satisfy the release gates.
