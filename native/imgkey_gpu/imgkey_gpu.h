#pragma once

#include <stdint.h>

#ifdef _WIN32
#define IMGKEY_GPU_API __declspec(dllexport)
#define IMGKEY_GPU_CALL __cdecl
#else
#define IMGKEY_GPU_API
#define IMGKEY_GPU_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IMGKEY_GPU_ABI_VERSION 1u

typedef enum ImgKeyGpuStatus {
    IMGKEY_GPU_OK = 0,
    IMGKEY_GPU_INVALID_ARGUMENT = 1,
    IMGKEY_GPU_UNSUPPORTED_VERSION = 2,
    IMGKEY_GPU_UNSUPPORTED_CAPABILITY = 3,
    IMGKEY_GPU_BACKEND_UNAVAILABLE = 4,
    IMGKEY_GPU_EXECUTION_FAILED = 5,
    IMGKEY_GPU_FALLBACK = 6
} ImgKeyGpuStatus;

typedef enum ImgKeyGpuFallbackReason {
    IMGKEY_GPU_FALLBACK_NONE = 0,
    IMGKEY_GPU_FALLBACK_BAD_DTYPE = 1,
    IMGKEY_GPU_FALLBACK_BAD_SHAPE = 2,
    IMGKEY_GPU_FALLBACK_BAD_STRIDE = 3,
    IMGKEY_GPU_FALLBACK_BAD_VERSION = 4,
    IMGKEY_GPU_FALLBACK_NULL_POINTER = 5,
    IMGKEY_GPU_FALLBACK_UNSUPPORTED_CAPABILITY = 6,
    IMGKEY_GPU_FALLBACK_BACKEND_UNAVAILABLE = 7,
    IMGKEY_GPU_FALLBACK_EXECUTION_FAILED = 8
} ImgKeyGpuFallbackReason;

typedef enum ImgKeyGpuCapabilityFlags {
    IMGKEY_GPU_CAP_CONSTANT_SCREEN = 1u << 0,
    IMGKEY_GPU_CAP_SCREEN_TILE = 1u << 1,
    IMGKEY_GPU_CAP_PERSISTENT_SESSION = 1u << 2,
    IMGKEY_GPU_CAP_TILE_BATCH = 1u << 3,
    IMGKEY_GPU_CAP_ALPHA_WRITE = 1u << 4,
    IMGKEY_GPU_CAP_RGB_ONLY = 1u << 5
} ImgKeyGpuCapabilityFlags;

typedef enum ImgKeyGpuDType {
    IMGKEY_GPU_DTYPE_U8 = 1,
    IMGKEY_GPU_DTYPE_BOOL8 = 2
} ImgKeyGpuDType;

typedef struct ImgKeyGpuTileBufferV1 {
    uint32_t struct_size;
    uint32_t version;
    void* data;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t dtype;
    int64_t row_stride_bytes;
    int64_t pixel_stride_bytes;
    uint64_t byte_size;
} ImgKeyGpuTileBufferV1;

typedef struct ImgKeyGpuColorTileParamsV1 {
    uint32_t struct_size;
    uint32_t version;
    uint64_t required_capabilities;
    int32_t status;
    int32_t fallback_reason;
    uint8_t screen_r;
    uint8_t screen_g;
    uint8_t screen_b;
    uint8_t reserved0;
    float foreground_reference_pull;
    float key_vector_despill;
    float preserve_foreground_luma;
    float transition_spill_threshold;
    float transition_reconstruction_error;
    float clip_foreground;
    uint32_t transition_alpha_min;
    uint32_t transition_alpha_max;
} ImgKeyGpuColorTileParamsV1;

IMGKEY_GPU_API uint32_t IMGKEY_GPU_CALL imgkey_gpu_version(void);
IMGKEY_GPU_API const char* IMGKEY_GPU_CALL imgkey_gpu_last_error(void);
IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_probe_v1(void* out_probe_json, uint32_t out_probe_json_bytes);
IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_create_context_v1(const ImgKeyGpuColorTileParamsV1* params, void** out_context);
IMGKEY_GPU_API void IMGKEY_GPU_CALL imgkey_gpu_destroy_context_v1(void* context);
IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_identity_rgba_v1(
    void* context,
    const ImgKeyGpuTileBufferV1* rgba,
    ImgKeyGpuTileBufferV1* out_rgba
);
IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_process_color_tile_v1(
    void* context,
    ImgKeyGpuColorTileParamsV1* params,
    const ImgKeyGpuTileBufferV1* rgb,
    const ImgKeyGpuTileBufferV1* alpha,
    const ImgKeyGpuTileBufferV1* background_mask,
    const ImgKeyGpuTileBufferV1* edge_mask,
    const ImgKeyGpuTileBufferV1* probability,
    const ImgKeyGpuTileBufferV1* fringe_mask,
    const ImgKeyGpuTileBufferV1* screen_tile,
    const ImgKeyGpuTileBufferV1* foreground_ref_rgb,
    const ImgKeyGpuTileBufferV1* foreground_ref_valid,
    ImgKeyGpuTileBufferV1* out_rgb,
    ImgKeyGpuTileBufferV1* out_repair_mask
);

#ifdef __cplusplus
}
#endif
