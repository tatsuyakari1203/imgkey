# ImgKey native GPU backend

This folder defines the backend-neutral C ABI and the D3D12 compute backend.
The build produces `imgkey_gpu.dll` with precompiled embedded HLSL bytecode and
exposes backend id `d3d12_compute` through `gpu_backend.D3D12ComputeBackend`.

Build:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File native/imgkey_gpu/build.ps1 -Clean
```

The build requires MSVC Build Tools, Windows SDK D3D12/DXGI headers/libs, and DXC
or FXC at build time. Shader compiler outputs and DLL/linker artifacts are written
under ignored `native/imgkey_gpu/build/`; the packaged app must not rely on a
runtime shader compiler.

Native kernels:

- `imgkey_gpu_identity_rgba_v1`: byte-exact RGBA upload/dispatch/readback smoke.
- `imgkey_gpu_process_color_tile_v1`: RGB-only transition repair for constant
  screen color and `screen_tile`/local plate inputs. Alpha remains CPU-owned.
- `imgkey_gpu_process_color_tile_v2`: fused full color tile RGB cleanup for the
  D3D12 path; larger tiles are split by Python into TDR-bounded subdispatches.

Current native D3D12 calls are capped at `512*512` pixels to stay below Windows
TDR risk. The Python D3D12 session splits larger full-color tiles into persistent
buffer subdispatches; unsupported/unsafe calls fall back cleanly instead of
launching long kernels.

ABI rules:

- every public struct carries `struct_size` and `version`;
- tile buffers carry explicit dimensions, dtypes, byte size, row stride, and
  pixel stride;
- errors are returned as status/fallback codes with an owned/thread-local
  `imgkey_gpu_last_error()` string;
- native code must not throw exceptions across the C boundary;
- capability flags are explicit: `constant_screen`, `screen_tile`,
  `persistent_session`, `tile_batch`, `alpha_write`, `rgb_only`, and
  `full_color_tile`;
- runtime shader compilers and SDKs are build-time tools only, not packaged app
  dependencies unless a later gate explicitly approves that change.

The current CUDA path remains a compatibility backend behind Python's
`gpu_backend.CudaCompatBackend`. Vulkan is represented by
`gpu_backend.VulkanComputeBackend` as a Phase 7 gate/probe only when SDK
headers/import libraries are missing: it runtime-loads/probes the installed
Vulkan loader through Python, reports a clean deferred fallback, and does not
ship a fake tile implementation.
