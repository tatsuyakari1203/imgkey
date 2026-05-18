#include "imgkey_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr int kAbiVersion = 1;
constexpr int kDllVersion = 10001;
constexpr int kMaxTilePixels = 16 * 1024 * 1024;
constexpr int kThreadsPerBlock = 256;

thread_local std::string g_last_error;

void set_last_error(const char* message) noexcept {
    try {
        g_last_error = message ? message : "";
    } catch (...) {
        // Keep the ABI noexcept even under allocation failure.
    }
}

void set_last_cuda_error(const char* prefix, cudaError_t error) noexcept {
    char buffer[512] = {};
    std::snprintf(buffer, sizeof(buffer), "%s: %s", prefix ? prefix : "CUDA error", cudaGetErrorString(error));
    set_last_error(buffer);
}

__device__ __forceinline__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ __forceinline__ float srgb_to_linear(float srgb) {
    float x = clamp01(srgb);
    return x <= 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float linear) {
    float x = clamp01(linear);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

__device__ __forceinline__ float luma(float r, float g, float b) {
    return clamp01(r) * 0.2126f + clamp01(g) * 0.7152f + clamp01(b) * 0.0722f;
}

__device__ __forceinline__ void match_luma(float r, float g, float b, float target_luma, float* out_r, float* out_g, float* out_b) {
    float src_luma = luma(r, g, b);
    float scale = src_luma > 1.0e-5f ? clamp01(target_luma) / fmaxf(src_luma, 1.0e-5f) : 1.0f;
    scale = fminf(fmaxf(scale, 0.0f), 4.0f);
    *out_r = clamp01(r * scale);
    *out_g = clamp01(g * scale);
    *out_b = clamp01(b * scale);
}

__device__ __forceinline__ unsigned char round_u8(float value) {
    return static_cast<unsigned char>(fminf(fmaxf(floorf(value + 0.5f), 0.0f), 255.0f));
}

__global__ void transition_repair_kernel(
    ImgKeyCudaTransitionParamsV1 params,
    const unsigned char* rgb,
    const unsigned char* alpha,
    const unsigned char* transition_mask,
    const unsigned char* foreground_ref_rgb,
    const unsigned char* foreground_ref_valid,
    unsigned char* out_rgb,
    unsigned char* out_repair_mask
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = params.width * params.height;
    if (index >= count) {
        return;
    }

    const int y = index / params.width;
    const int x = index - y * params.width;
    const unsigned char* rgb_px = rgb + y * params.rgb_stride_bytes + x * 3;
    const unsigned char* fg_px = foreground_ref_rgb + y * params.rgb_stride_bytes + x * 3;
    const unsigned char* alpha_px = alpha + y * params.alpha_stride_bytes + x;
    const unsigned char* transition_px = transition_mask + y * params.mask_stride_bytes + x;
    const unsigned char* fg_valid_px = foreground_ref_valid + y * params.mask_stride_bytes + x;
    unsigned char* out_px = out_rgb + y * params.out_stride_bytes + x * 3;
    unsigned char* repair_px = out_repair_mask + y * params.mask_stride_bytes + x;

    const unsigned char src_r_u8 = rgb_px[0];
    const unsigned char src_g_u8 = rgb_px[1];
    const unsigned char src_b_u8 = rgb_px[2];
    out_px[0] = src_r_u8;
    out_px[1] = src_g_u8;
    out_px[2] = src_b_u8;
    *repair_px = 0;

    const unsigned char alpha_u8 = *alpha_px;
    if (alpha_u8 == 0) {
        out_px[0] = 0;
        out_px[1] = 0;
        out_px[2] = 0;
        return;
    }
    const float repair_strength = static_cast<float>(*transition_px) / 255.0f;
    if (repair_strength <= (1.0f / 255.0f) || *fg_valid_px == 0) {
        return;
    }

    const float src_r = static_cast<float>(src_r_u8) / 255.0f;
    const float src_g = static_cast<float>(src_g_u8) / 255.0f;
    const float src_b = static_cast<float>(src_b_u8) / 255.0f;
    const float src_lr = srgb_to_linear(src_r);
    const float src_lg = srgb_to_linear(src_g);
    const float src_lb = srgb_to_linear(src_b);

    const float fg_lr = srgb_to_linear(static_cast<float>(fg_px[0]) / 255.0f);
    const float fg_lg = srgb_to_linear(static_cast<float>(fg_px[1]) / 255.0f);
    const float fg_lb = srgb_to_linear(static_cast<float>(fg_px[2]) / 255.0f);

    const float screen_lr = srgb_to_linear(static_cast<float>(params.screen_r) / 255.0f);
    const float screen_lg = srgb_to_linear(static_cast<float>(params.screen_g) / 255.0f);
    const float screen_lb = srgb_to_linear(static_cast<float>(params.screen_b) / 255.0f);
    const float screen_luma = luma(screen_lr, screen_lg, screen_lb);

    const float alpha_f = static_cast<float>(alpha_u8) / 255.0f;
    const float safe_alpha = fmaxf(alpha_f, 1.0f / 255.0f);
    float est_r = clamp01((src_lr - (1.0f - alpha_f) * screen_lr) / safe_alpha);
    float est_g = clamp01((src_lg - (1.0f - alpha_f) * screen_lg) / safe_alpha);
    float est_b = clamp01((src_lb - (1.0f - alpha_f) * screen_lb) / safe_alpha);

    float key_r = clamp01(screen_lr) - screen_luma;
    float key_g = clamp01(screen_lg) - screen_luma;
    float key_b = clamp01(screen_lb) - screen_luma;
    const float key_norm = sqrtf(key_r * key_r + key_g * key_g + key_b * key_b);
    if (key_norm >= 1.0e-5f) {
        key_r /= key_norm;
        key_g /= key_norm;
        key_b /= key_norm;
    } else {
        key_r = 0.0f;
        key_g = 0.0f;
        key_b = 0.0f;
    }

    const float est_luma = luma(est_r, est_g, est_b);
    const float ref_luma = luma(fg_lr, fg_lg, fg_lb);
    const float chroma_r = est_r - est_luma;
    const float chroma_g = est_g - est_luma;
    const float chroma_b = est_b - est_luma;
    const float vector_spill = fmaxf(chroma_r * key_r + chroma_g * key_g + chroma_b * key_b, 0.0f);

    const float despill_amount = clamp01(params.key_vector_despill);
    if (despill_amount > 0.0f) {
        const float amount = vector_spill * despill_amount * repair_strength;
        est_r = clamp01(est_r - key_r * amount);
        est_g = clamp01(est_g - key_g * amount);
        est_b = clamp01(est_b - key_b * amount);
    }

    const float pull_amount = clamp01(params.foreground_reference_pull);
    if (pull_amount > 0.0f) {
        const float pull = clamp01(repair_strength * pull_amount);
        float ref_match_r = 0.0f;
        float ref_match_g = 0.0f;
        float ref_match_b = 0.0f;
        match_luma(fg_lr, fg_lg, fg_lb, luma(est_r, est_g, est_b), &ref_match_r, &ref_match_g, &ref_match_b);
        est_r = est_r * (1.0f - pull) + ref_match_r * pull;
        est_g = est_g * (1.0f - pull) + ref_match_g * pull;
        est_b = est_b * (1.0f - pull) + ref_match_b * pull;
    }

    const float preserve_amount = clamp01(params.preserve_foreground_luma);
    if (preserve_amount > 0.0f) {
        const float preserve = clamp01(repair_strength * preserve_amount);
        float luma_match_r = 0.0f;
        float luma_match_g = 0.0f;
        float luma_match_b = 0.0f;
        match_luma(est_r, est_g, est_b, ref_luma, &luma_match_r, &luma_match_g, &luma_match_b);
        est_r = est_r * (1.0f - preserve) + luma_match_r * preserve;
        est_g = est_g * (1.0f - preserve) + luma_match_g * preserve;
        est_b = est_b * (1.0f - preserve) + luma_match_b * preserve;
    }

    const unsigned char repaired_r = round_u8(linear_to_srgb(est_r) * 255.0f);
    const unsigned char repaired_g = round_u8(linear_to_srgb(est_g) * 255.0f);
    const unsigned char repaired_b = round_u8(linear_to_srgb(est_b) * 255.0f);

    out_px[0] = repaired_r;
    out_px[1] = repaired_g;
    out_px[2] = repaired_b;

    const int delta_r = abs(static_cast<int>(repaired_r) - static_cast<int>(src_r_u8));
    const int delta_g = abs(static_cast<int>(repaired_g) - static_cast<int>(src_g_u8));
    const int delta_b = abs(static_cast<int>(repaired_b) - static_cast<int>(src_b_u8));
    const int delta_max = delta_r > delta_g ? (delta_r > delta_b ? delta_r : delta_b) : (delta_g > delta_b ? delta_g : delta_b);
    const float delta = static_cast<float>(delta_max) / 255.0f;
    *repair_px = round_u8(fmaxf(repair_strength, delta) * 255.0f);
}

ImgKeyCudaStatus validate_transition_args(
    const ImgKeyCudaTransitionParamsV1* params,
    const unsigned char* rgb,
    const unsigned char* alpha,
    const unsigned char* transition_mask,
    const unsigned char* foreground_ref_rgb,
    const unsigned char* foreground_ref_valid,
    unsigned char* out_rgb,
    unsigned char* out_repair_mask
) noexcept {
    if (params == nullptr) {
        set_last_error("params pointer is null");
        return IMGKEY_CUDA_INVALID_ARGUMENT;
    }
    if (params->struct_size != static_cast<int>(sizeof(ImgKeyCudaTransitionParamsV1))) {
        set_last_error("params struct_size does not match ImgKeyCudaTransitionParamsV1");
        return IMGKEY_CUDA_UNSUPPORTED_VERSION;
    }
    if (params->version != kAbiVersion) {
        set_last_error("params version is unsupported");
        return IMGKEY_CUDA_UNSUPPORTED_VERSION;
    }
    if (rgb == nullptr || alpha == nullptr || transition_mask == nullptr || foreground_ref_rgb == nullptr || foreground_ref_valid == nullptr || out_rgb == nullptr || out_repair_mask == nullptr) {
        set_last_error("one or more image pointers are null");
        return IMGKEY_CUDA_INVALID_ARGUMENT;
    }
    if (params->width <= 0 || params->height <= 0) {
        set_last_error("width and height must be positive");
        return IMGKEY_CUDA_INVALID_ARGUMENT;
    }
    const long long pixel_count = static_cast<long long>(params->width) * static_cast<long long>(params->height);
    if (pixel_count <= 0 || pixel_count > kMaxTilePixels) {
        set_last_error("tile dimensions exceed max tile size");
        return IMGKEY_CUDA_INVALID_ARGUMENT;
    }
    if (params->rgb_stride_bytes < params->width * 3 || params->out_stride_bytes < params->width * 3) {
        set_last_error("RGB/output strides must be positive and at least width * 3");
        return IMGKEY_CUDA_INVALID_ARGUMENT;
    }
    if (params->alpha_stride_bytes < params->width || params->mask_stride_bytes < params->width) {
        set_last_error("alpha/mask strides must be positive and at least width");
        return IMGKEY_CUDA_INVALID_ARGUMENT;
    }
    return IMGKEY_CUDA_OK;
}

}  // namespace

extern "C" IMGKEY_CUDA_API int IMGKEY_CUDA_CALL imgkey_cuda_version(void) {
    set_last_error("");
    return kDllVersion;
}

extern "C" IMGKEY_CUDA_API int IMGKEY_CUDA_CALL imgkey_cuda_device_count(void) {
    try {
        int count = 0;
        const cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            set_last_cuda_error("cudaGetDeviceCount failed", err);
            return 0;
        }
        set_last_error("");
        return count;
    } catch (...) {
        set_last_error("unexpected exception in imgkey_cuda_device_count");
        return 0;
    }
}

extern "C" IMGKEY_CUDA_API const char* IMGKEY_CUDA_CALL imgkey_cuda_last_error(void) {
    return g_last_error.c_str();
}

extern "C" IMGKEY_CUDA_API ImgKeyCudaStatus IMGKEY_CUDA_CALL imgkey_cuda_transition_repair_v1(
    const ImgKeyCudaTransitionParamsV1* params,
    const unsigned char* rgb,
    const unsigned char* alpha,
    const unsigned char* transition_mask,
    const unsigned char* foreground_ref_rgb,
    const unsigned char* foreground_ref_valid,
    unsigned char* out_rgb,
    unsigned char* out_repair_mask
) {
    try {
        ImgKeyCudaStatus valid = validate_transition_args(params, rgb, alpha, transition_mask, foreground_ref_rgb, foreground_ref_valid, out_rgb, out_repair_mask);
        if (valid != IMGKEY_CUDA_OK) {
            return valid;
        }

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            set_last_cuda_error("cudaGetDeviceCount failed", err);
            return IMGKEY_CUDA_NO_DEVICE;
        }
        if (device_count <= 0) {
            set_last_error("no CUDA device is available");
            return IMGKEY_CUDA_NO_DEVICE;
        }

        const size_t width = static_cast<size_t>(params->width);
        const size_t height = static_cast<size_t>(params->height);
        const size_t rgb_row_bytes = width * 3;
        const size_t mask_row_bytes = width;
        const size_t rgb_bytes = rgb_row_bytes * height;
        const size_t mask_bytes = mask_row_bytes * height;

        unsigned char* d_rgb = nullptr;
        unsigned char* d_alpha = nullptr;
        unsigned char* d_transition_mask = nullptr;
        unsigned char* d_foreground_ref_rgb = nullptr;
        unsigned char* d_foreground_ref_valid = nullptr;
        unsigned char* d_out_rgb = nullptr;
        unsigned char* d_out_repair_mask = nullptr;
        auto cleanup = [&]() noexcept {
            if (d_rgb) cudaFree(d_rgb);
            if (d_alpha) cudaFree(d_alpha);
            if (d_transition_mask) cudaFree(d_transition_mask);
            if (d_foreground_ref_rgb) cudaFree(d_foreground_ref_rgb);
            if (d_foreground_ref_valid) cudaFree(d_foreground_ref_valid);
            if (d_out_rgb) cudaFree(d_out_rgb);
            if (d_out_repair_mask) cudaFree(d_out_repair_mask);
        };

        err = cudaMalloc(&d_rgb, rgb_bytes);
        if (err == cudaSuccess) err = cudaMalloc(&d_alpha, mask_bytes);
        if (err == cudaSuccess) err = cudaMalloc(&d_transition_mask, mask_bytes);
        if (err == cudaSuccess) err = cudaMalloc(&d_foreground_ref_rgb, rgb_bytes);
        if (err == cudaSuccess) err = cudaMalloc(&d_foreground_ref_valid, mask_bytes);
        if (err == cudaSuccess) err = cudaMalloc(&d_out_rgb, rgb_bytes);
        if (err == cudaSuccess) err = cudaMalloc(&d_out_repair_mask, mask_bytes);
        if (err != cudaSuccess) {
            set_last_cuda_error("cudaMalloc failed", err);
            cleanup();
            return IMGKEY_CUDA_LAUNCH_FAILED;
        }

        err = cudaMemcpy2D(d_rgb, rgb_row_bytes, rgb, static_cast<size_t>(params->rgb_stride_bytes), rgb_row_bytes, height, cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy2D(d_alpha, mask_row_bytes, alpha, static_cast<size_t>(params->alpha_stride_bytes), mask_row_bytes, height, cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy2D(d_transition_mask, mask_row_bytes, transition_mask, static_cast<size_t>(params->mask_stride_bytes), mask_row_bytes, height, cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy2D(d_foreground_ref_rgb, rgb_row_bytes, foreground_ref_rgb, static_cast<size_t>(params->rgb_stride_bytes), rgb_row_bytes, height, cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy2D(d_foreground_ref_valid, mask_row_bytes, foreground_ref_valid, static_cast<size_t>(params->mask_stride_bytes), mask_row_bytes, height, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            set_last_cuda_error("cudaMemcpy host-to-device failed", err);
            cleanup();
            return IMGKEY_CUDA_LAUNCH_FAILED;
        }

        ImgKeyCudaTransitionParamsV1 device_params = *params;
        device_params.rgb_stride_bytes = static_cast<int>(rgb_row_bytes);
        device_params.alpha_stride_bytes = static_cast<int>(mask_row_bytes);
        device_params.mask_stride_bytes = static_cast<int>(mask_row_bytes);
        device_params.out_stride_bytes = static_cast<int>(rgb_row_bytes);

        const int pixel_count = params->width * params->height;
        const int block_count = (pixel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        transition_repair_kernel<<<block_count, kThreadsPerBlock>>>(device_params, d_rgb, d_alpha, d_transition_mask, d_foreground_ref_rgb, d_foreground_ref_valid, d_out_rgb, d_out_repair_mask);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            set_last_cuda_error("transition kernel launch failed", err);
            cleanup();
            return IMGKEY_CUDA_LAUNCH_FAILED;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            set_last_cuda_error("transition kernel synchronization failed", err);
            cleanup();
            return IMGKEY_CUDA_LAUNCH_FAILED;
        }
        err = cudaMemcpy2D(out_rgb, static_cast<size_t>(params->out_stride_bytes), d_out_rgb, rgb_row_bytes, rgb_row_bytes, height, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) err = cudaMemcpy2D(out_repair_mask, static_cast<size_t>(params->mask_stride_bytes), d_out_repair_mask, mask_row_bytes, mask_row_bytes, height, cudaMemcpyDeviceToHost);
        cleanup();
        if (err != cudaSuccess) {
            set_last_cuda_error("cudaMemcpy device-to-host failed", err);
            return IMGKEY_CUDA_LAUNCH_FAILED;
        }
        set_last_error("");
        return IMGKEY_CUDA_OK;
    } catch (...) {
        set_last_error("unexpected exception in imgkey_cuda_transition_repair_v1");
        return IMGKEY_CUDA_LAUNCH_FAILED;
    }
}
