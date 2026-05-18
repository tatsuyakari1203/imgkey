#pragma once

#ifdef _WIN32
#define IMGKEY_CUDA_API __declspec(dllexport)
#define IMGKEY_CUDA_CALL __cdecl
#else
#define IMGKEY_CUDA_API
#define IMGKEY_CUDA_CALL
#endif

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
