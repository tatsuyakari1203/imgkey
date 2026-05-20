# ImgKey native GPU ABI scaffold

This folder defines the backend-neutral C ABI planned for the future D3D12 and
Vulkan native backends. Phase 4 only adds the contract and Python validation;
it does not ship a D3D12 or Vulkan shader implementation.

ABI rules:

- every public struct carries `struct_size` and `version`;
- tile buffers carry explicit dimensions, dtypes, byte size, row stride, and
  pixel stride;
- errors are returned as status/fallback codes with an owned/thread-local
  `imgkey_gpu_last_error()` string;
- native code must not throw exceptions across the C boundary;
- capability flags are explicit: `constant_screen`, `screen_tile`,
  `persistent_session`, `tile_batch`, `alpha_write`, and `rgb_only`;
- runtime shader compilers and SDKs are build-time tools only, not packaged app
  dependencies unless a later gate explicitly approves that change.

The current CUDA path remains a compatibility backend behind Python's
`gpu_backend.CudaCompatBackend` until the D3D12/Vulkan backends exist.
