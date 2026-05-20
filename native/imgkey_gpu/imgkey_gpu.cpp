#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "imgkey_gpu.h"
#include "imgkey_gpu_shaders.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint64_t kCapabilities =
    IMGKEY_GPU_CAP_CONSTANT_SCREEN |
    IMGKEY_GPU_CAP_SCREEN_TILE |
    IMGKEY_GPU_CAP_PERSISTENT_SESSION |
    IMGKEY_GPU_CAP_FULL_COLOR_TILE |
    IMGKEY_GPU_CAP_RGB_ONLY;
// Whole read-tiles may now be substantially larger than the Phase-5 MVP cap,
// but each shader dispatch is still chunked to avoid long single-dispatch TDR
// exposure on lower-end Windows GPUs.
constexpr uint32_t kMaxTilePixels = 3072u * 3072u;
constexpr uint32_t kMaxDispatchWidth = 512u;
constexpr uint32_t kMaxDispatchHeight = 512u;
constexpr uint32_t kMaxDispatchPixels = kMaxDispatchWidth * kMaxDispatchHeight;
constexpr uint32_t kMaxNativeCallPixels = kMaxDispatchPixels;
constexpr DWORD kFenceTimeoutMs = 120000u;

thread_local std::string g_last_error;

void set_error(const std::string& message) {
    g_last_error = message;
}

void clear_error() {
    g_last_error.clear();
}

std::string hresult_message(const char* label, HRESULT hr) {
    char buffer[256] = {};
    std::snprintf(buffer, sizeof(buffer), "%s failed (HRESULT 0x%08lx)", label, static_cast<unsigned long>(hr));
    return std::string(buffer);
}

std::string wide_to_utf8(const wchar_t* text) {
    if (!text || !*text) {
        return std::string();
    }
    int needed = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) {
        return std::string();
    }
    std::string out(static_cast<size_t>(needed - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), needed, nullptr, nullptr);
    return out;
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                char tmp[8] = {};
                std::snprintf(tmp, sizeof(tmp), "\\u%04x", static_cast<unsigned char>(ch));
                out += tmp;
            } else {
                out += ch;
            }
        }
    }
    return out;
}

bool write_json(void* out_probe_json, uint32_t out_probe_json_bytes, const std::string& json) {
    if (!out_probe_json || out_probe_json_bytes == 0) {
        set_error("probe output buffer is null or empty");
        return false;
    }
    if (json.size() + 1 > out_probe_json_bytes) {
        set_error("probe output buffer is too small");
        return false;
    }
    std::memcpy(out_probe_json, json.c_str(), json.size() + 1);
    return true;
}

ComPtr<IDXGIAdapter1> select_adapter(DXGI_ADAPTER_DESC1* out_desc, std::string* out_error) {
    UINT factory_flags = 0;
    ComPtr<IDXGIFactory6> factory6;
    HRESULT hr = CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&factory6));
    if (FAILED(hr)) {
        if (out_error) *out_error = hresult_message("CreateDXGIFactory2", hr);
        return nullptr;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT index = 0; SUCCEEDED(factory6->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))); ++index) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter.Reset();
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
            if (out_desc) *out_desc = desc;
            return adapter;
        }
        adapter.Reset();
    }

    ComPtr<IDXGIFactory1> factory1;
    hr = factory6.As(&factory1);
    if (FAILED(hr)) {
        if (out_error) *out_error = hresult_message("IDXGIFactory1 query", hr);
        return nullptr;
    }
    for (UINT index = 0; SUCCEEDED(factory1->EnumAdapters1(index, &adapter)); ++index) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter.Reset();
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
            if (out_desc) *out_desc = desc;
            return adapter;
        }
        adapter.Reset();
    }

    if (out_error) *out_error = "No hardware D3D12 adapter accepting feature level 11_0 was found";
    return nullptr;
}

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES props = {};
    props.Type = type;
    props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;
    return props;
}

D3D12_RESOURCE_DESC buffer_desc(uint64_t bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = std::max<uint64_t>(bytes, 4u);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

uint64_t align4(uint64_t value) {
    return (value + 3u) & ~uint64_t(3u);
}

D3D12_RESOURCE_BARRIER transition_barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

D3D12_RESOURCE_BARRIER uav_barrier(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.UAV.pResource = resource;
    return barrier;
}

bool validate_buffer(
    const ImgKeyGpuTileBufferV1* buffer,
    const char* name,
    uint32_t expected_channels,
    uint32_t width,
    uint32_t height,
    bool allow_bool,
    bool writable,
    ImgKeyGpuColorTileParamsV1* params) {
    auto fail = [&](ImgKeyGpuFallbackReason reason, const std::string& message) -> bool {
        if (params) {
            params->status = IMGKEY_GPU_INVALID_ARGUMENT;
            params->fallback_reason = reason;
        }
        set_error(message);
        return false;
    };

    if (!buffer) {
        return fail(IMGKEY_GPU_FALLBACK_NULL_POINTER, std::string(name) + " buffer pointer is null");
    }
    if (buffer->struct_size != sizeof(ImgKeyGpuTileBufferV1) || buffer->version != IMGKEY_GPU_ABI_VERSION) {
        if (params) {
            params->status = IMGKEY_GPU_UNSUPPORTED_VERSION;
            params->fallback_reason = IMGKEY_GPU_FALLBACK_BAD_VERSION;
        }
        set_error(std::string(name) + " buffer version/size is unsupported");
        return false;
    }
    if (!buffer->data) {
        return fail(IMGKEY_GPU_FALLBACK_NULL_POINTER, std::string(name) + " data pointer is null");
    }
    if (buffer->width != width || buffer->height != height || buffer->channels != expected_channels) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_SHAPE, std::string(name) + " shape does not match the color tile");
    }
    if (buffer->dtype != IMGKEY_GPU_DTYPE_U8 && !(allow_bool && buffer->dtype == IMGKEY_GPU_DTYPE_BOOL8)) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_DTYPE, std::string(name) + " dtype is unsupported");
    }
    if (buffer->row_stride_bytes <= 0 || buffer->pixel_stride_bytes <= 0) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " strides must be positive");
    }
    if (buffer->pixel_stride_bytes < static_cast<int64_t>(std::max<uint32_t>(1, expected_channels))) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " pixel stride is too small");
    }
    uint64_t min_row = static_cast<uint64_t>(width - 1u) * static_cast<uint64_t>(buffer->pixel_stride_bytes) + std::max<uint32_t>(1, expected_channels);
    if (static_cast<uint64_t>(buffer->row_stride_bytes) < min_row) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " row stride is too small");
    }
    uint64_t min_span = static_cast<uint64_t>(height - 1u) * static_cast<uint64_t>(buffer->row_stride_bytes) + min_row;
    if (buffer->byte_size < min_span) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " byte size is too small");
    }
    (void)writable;
    return true;
}

bool validate_buffer_v2(
    const ImgKeyGpuTileBufferV1* buffer,
    const char* name,
    uint32_t expected_channels,
    uint32_t width,
    uint32_t height,
    bool allow_bool,
    bool writable,
    ImgKeyGpuColorTileParamsV2* params) {
    auto fail = [&](ImgKeyGpuFallbackReason reason, const std::string& message) -> bool {
        if (params) {
            params->status = IMGKEY_GPU_INVALID_ARGUMENT;
            params->fallback_reason = reason;
        }
        set_error(message);
        return false;
    };

    if (!buffer) {
        return fail(IMGKEY_GPU_FALLBACK_NULL_POINTER, std::string(name) + " buffer pointer is null");
    }
    if (buffer->struct_size != sizeof(ImgKeyGpuTileBufferV1) || buffer->version != IMGKEY_GPU_ABI_VERSION) {
        if (params) {
            params->status = IMGKEY_GPU_UNSUPPORTED_VERSION;
            params->fallback_reason = IMGKEY_GPU_FALLBACK_BAD_VERSION;
        }
        set_error(std::string(name) + " buffer version/size is unsupported");
        return false;
    }
    if (!buffer->data) {
        return fail(IMGKEY_GPU_FALLBACK_NULL_POINTER, std::string(name) + " data pointer is null");
    }
    if (buffer->width != width || buffer->height != height || buffer->channels != expected_channels) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_SHAPE, std::string(name) + " shape does not match the color tile");
    }
    if (buffer->dtype != IMGKEY_GPU_DTYPE_U8 && !(allow_bool && buffer->dtype == IMGKEY_GPU_DTYPE_BOOL8)) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_DTYPE, std::string(name) + " dtype is unsupported");
    }
    if (buffer->row_stride_bytes <= 0 || buffer->pixel_stride_bytes <= 0) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " strides must be positive");
    }
    if (buffer->pixel_stride_bytes < static_cast<int64_t>(std::max<uint32_t>(1, expected_channels))) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " pixel stride is too small");
    }
    uint64_t min_row = static_cast<uint64_t>(width - 1u) * static_cast<uint64_t>(buffer->pixel_stride_bytes) + std::max<uint32_t>(1, expected_channels);
    if (static_cast<uint64_t>(buffer->row_stride_bytes) < min_row) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " row stride is too small");
    }
    uint64_t min_span = static_cast<uint64_t>(height - 1u) * static_cast<uint64_t>(buffer->row_stride_bytes) + min_row;
    if (buffer->byte_size < min_span) {
        return fail(IMGKEY_GPU_FALLBACK_BAD_STRIDE, std::string(name) + " byte size is too small");
    }
    (void)writable;
    return true;
}

bool validate_params(ImgKeyGpuColorTileParamsV1* params) {
    if (!params) {
        set_error("params pointer is null");
        return false;
    }
    if (params->struct_size != sizeof(ImgKeyGpuColorTileParamsV1) || params->version != IMGKEY_GPU_ABI_VERSION) {
        params->status = IMGKEY_GPU_UNSUPPORTED_VERSION;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_BAD_VERSION;
        set_error("params version/size is unsupported");
        return false;
    }
    uint64_t unsupported = params->required_capabilities & ~kCapabilities;
    if (unsupported != 0) {
        params->status = IMGKEY_GPU_UNSUPPORTED_CAPABILITY;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_UNSUPPORTED_CAPABILITY;
        set_error("requested capabilities are not supported by the D3D12 backend");
        return false;
    }
    return true;
}

bool validate_params_v2(ImgKeyGpuColorTileParamsV2* params) {
    if (!params) {
        set_error("params pointer is null");
        return false;
    }
    if (params->struct_size != sizeof(ImgKeyGpuColorTileParamsV2) || params->version != IMGKEY_GPU_ABI_VERSION) {
        params->status = IMGKEY_GPU_UNSUPPORTED_VERSION;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_BAD_VERSION;
        set_error("params v2 version/size is unsupported");
        return false;
    }
    uint64_t unsupported = params->required_capabilities & ~kCapabilities;
    if (unsupported != 0) {
        params->status = IMGKEY_GPU_UNSUPPORTED_CAPABILITY;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_UNSUPPORTED_CAPABILITY;
        set_error("requested capabilities are not supported by the D3D12 backend");
        return false;
    }
    return true;
}

struct UploadResource {
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> gpu;
    uint64_t size = 0;
};

struct ReadbackResource {
    ComPtr<ID3D12Resource> gpu;
    ComPtr<ID3D12Resource> readback;
    uint64_t size = 0;
};

struct PersistentInputResource {
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> gpu;
    uint64_t capacity = 0;
    void* mapped = nullptr;
    D3D12_RESOURCE_STATES gpu_state = D3D12_RESOURCE_STATE_COPY_DEST;

    ~PersistentInputResource() {
        if (upload && mapped) {
            upload->Unmap(0, nullptr);
            mapped = nullptr;
        }
    }
};

struct PersistentOutputResource {
    ComPtr<ID3D12Resource> gpu;
    ComPtr<ID3D12Resource> readback;
    uint64_t capacity = 0;
    D3D12_RESOURCE_STATES gpu_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
};

struct ColorPipelineResources {
    PersistentInputResource rgb;
    PersistentInputResource alpha;
    PersistentInputResource background;
    PersistentInputResource edge;
    PersistentInputResource probability;
    PersistentInputResource fringe;
    PersistentInputResource screen;
    PersistentInputResource foreground_rgb;
    PersistentInputResource foreground_valid;
    PersistentInputResource transition_rgb;
    PersistentInputResource transition_valid;
    PersistentOutputResource out_rgb;
    PersistentOutputResource out_mask;
    uint64_t allocations = 0;
    uint64_t reuses = 0;
};

struct ShaderConstants {
    uint32_t width;
    uint32_t height;
    uint32_t has_screen_tile;
    uint32_t screen_rgb;
    uint32_t rgb_row_stride;
    uint32_t alpha_row_stride;
    uint32_t mask_row_stride;
    uint32_t screen_row_stride;
    uint32_t foreground_row_stride;
    uint32_t transition_row_stride;
    uint32_t clip_foreground_limit;
    uint32_t transition_alpha_min;
    uint32_t transition_alpha_max;
    uint32_t dispatch_x0;
    uint32_t dispatch_y0;
    uint32_t dispatch_width;
    uint32_t dispatch_height;
    uint32_t transition_enabled;
    uint32_t transition_reference_enabled;
    uint32_t reserved0;
    float foreground_reference_pull;
    float key_vector_despill;
    float preserve_foreground_luma;
    float transition_spill_threshold;
    float transition_reconstruction_error;
    float despill;
    float decontaminate;
    float unmix_amount;
    float edge_color_repair;
    float inner_color_pull;
    float fringe_remove;
    float luminance_protect;
    float clamp_key_r;
    float clamp_key_g;
    float clamp_key_b;
    float reserved1;
};

struct D3D12Context {
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 adapter_desc = {};
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> identity_pso;
    ComPtr<ID3D12PipelineState> color_pso;
    ComPtr<ID3D12PipelineState> full_color_pso;
    ColorPipelineResources color_resources;
    PersistentInputResource identity_input;
    PersistentOutputResource identity_output;
    std::mutex mutex;

    ~D3D12Context() {
        if (fence_event) {
            CloseHandle(fence_event);
            fence_event = nullptr;
        }
    }
};

bool create_root_signature(D3D12Context& ctx) {
    D3D12_ROOT_PARAMETER params[14] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = sizeof(ShaderConstants) / sizeof(uint32_t);

    for (uint32_t i = 1; i <= 11; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[i].Descriptor.ShaderRegister = i - 1;
        params[i].Descriptor.RegisterSpace = 0;
    }
    for (uint32_t i = 12; i <= 13; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[i].Descriptor.ShaderRegister = i - 12;
        params[i].Descriptor.RegisterSpace = 0;
    }

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = static_cast<UINT>(sizeof(params) / sizeof(params[0]));
    desc.pParameters = params;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (FAILED(hr)) {
        std::string detail = hresult_message("D3D12SerializeRootSignature", hr);
        if (error) detail += ": " + std::string(static_cast<const char*>(error->GetBufferPointer()), error->GetBufferSize());
        set_error(detail);
        return false;
    }
    hr = ctx.device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&ctx.root_signature));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateRootSignature", hr));
        return false;
    }
    return true;
}

bool create_compute_pso(D3D12Context& ctx, const unsigned char* shader, size_t shader_size, ComPtr<ID3D12PipelineState>& out_pso, const char* label) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = ctx.root_signature.Get();
    desc.CS.pShaderBytecode = shader;
    desc.CS.BytecodeLength = shader_size;
    HRESULT hr = ctx.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&out_pso));
    if (FAILED(hr)) {
        set_error(hresult_message(label, hr));
        return false;
    }
    return true;
}

bool initialize_context(D3D12Context& ctx) {
    std::string adapter_error;
    ctx.adapter = select_adapter(&ctx.adapter_desc, &adapter_error);
    if (!ctx.adapter) {
        set_error(adapter_error.empty() ? "No D3D12 adapter found" : adapter_error);
        return false;
    }
    HRESULT hr = D3D12CreateDevice(ctx.adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx.device));
    if (FAILED(hr)) {
        set_error(hresult_message("D3D12CreateDevice", hr));
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.NodeMask = 0;
    hr = ctx.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&ctx.queue));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommandQueue", hr));
        return false;
    }
    hr = ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&ctx.allocator));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommandAllocator", hr));
        return false;
    }
    hr = ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, ctx.allocator.Get(), nullptr, IID_PPV_ARGS(&ctx.command_list));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommandList", hr));
        return false;
    }
    ctx.command_list->Close();
    hr = ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fence));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateFence", hr));
        return false;
    }
    ctx.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ctx.fence_event) {
        set_error("CreateEventW for D3D12 fence failed");
        return false;
    }
    if (!create_root_signature(ctx)) {
        return false;
    }
    if (!create_compute_pso(ctx, g_imgkey_identity_cs, sizeof(g_imgkey_identity_cs), ctx.identity_pso, "CreateComputePipelineState(identity)")) {
        return false;
    }
    if (!create_compute_pso(ctx, g_imgkey_color_tile_cs, sizeof(g_imgkey_color_tile_cs), ctx.color_pso, "CreateComputePipelineState(color_tile)")) {
        return false;
    }
    if (!create_compute_pso(ctx, g_imgkey_full_color_tile_cs, sizeof(g_imgkey_full_color_tile_cs), ctx.full_color_pso, "CreateComputePipelineState(full_color_tile)")) {
        return false;
    }
    return true;
}

bool wait_for_gpu(D3D12Context& ctx) {
    uint64_t value = ++ctx.fence_value;
    HRESULT hr = ctx.queue->Signal(ctx.fence.Get(), value);
    if (FAILED(hr)) {
        set_error(hresult_message("Queue::Signal", hr));
        return false;
    }
    if (ctx.fence->GetCompletedValue() < value) {
        hr = ctx.fence->SetEventOnCompletion(value, ctx.fence_event);
        if (FAILED(hr)) {
            set_error(hresult_message("Fence::SetEventOnCompletion", hr));
            return false;
        }
        DWORD wait_status = WaitForSingleObject(ctx.fence_event, kFenceTimeoutMs);
        if (wait_status != WAIT_OBJECT_0) {
            set_error(wait_status == WAIT_TIMEOUT ? "D3D12 fence wait timed out; CPU fallback is required to avoid TDR" : "D3D12 fence wait failed");
            return false;
        }
    }
    return true;
}

bool create_upload_resource(D3D12Context& ctx, const ImgKeyGpuTileBufferV1* buffer, UploadResource& out) {
    out.size = std::max<uint64_t>(align4(buffer->byte_size), 4u);
    D3D12_HEAP_PROPERTIES upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_HEAP_PROPERTIES default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC desc = buffer_desc(out.size);
    HRESULT hr = ctx.device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&out.upload));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(upload)", hr));
        return false;
    }
    hr = ctx.device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out.gpu));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(input default)", hr));
        return false;
    }
    void* mapped = nullptr;
    D3D12_RANGE range = {0, 0};
    hr = out.upload->Map(0, &range, &mapped);
    if (FAILED(hr) || !mapped) {
        set_error(hresult_message("Map(upload)", hr));
        return false;
    }
    std::memset(mapped, 0, static_cast<size_t>(out.size));
    std::memcpy(mapped, buffer->data, static_cast<size_t>(buffer->byte_size));
    out.upload->Unmap(0, nullptr);
    ctx.command_list->CopyBufferRegion(out.gpu.Get(), 0, out.upload.Get(), 0, out.size);
    D3D12_RESOURCE_BARRIER barrier = transition_barrier(out.gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ctx.command_list->ResourceBarrier(1, &barrier);
    return true;
}

bool create_zero_upload_resource(D3D12Context& ctx, UploadResource& out) {
    ImgKeyGpuTileBufferV1 dummy = {};
    uint8_t zero[4] = {};
    dummy.struct_size = sizeof(dummy);
    dummy.version = IMGKEY_GPU_ABI_VERSION;
    dummy.data = zero;
    dummy.width = 1;
    dummy.height = 1;
    dummy.channels = 1;
    dummy.dtype = IMGKEY_GPU_DTYPE_U8;
    dummy.row_stride_bytes = 4;
    dummy.pixel_stride_bytes = 1;
    dummy.byte_size = 4;
    return create_upload_resource(ctx, &dummy, out);
}

bool create_readback_resource(D3D12Context& ctx, uint64_t bytes, ReadbackResource& out) {
    out.size = std::max<uint64_t>(bytes, 4u);
    D3D12_HEAP_PROPERTIES default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_HEAP_PROPERTIES readback_heap = heap_properties(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC gpu_desc = buffer_desc(out.size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hr = ctx.device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &gpu_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&out.gpu));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(output default)", hr));
        return false;
    }
    D3D12_RESOURCE_DESC readback_desc = buffer_desc(out.size);
    hr = ctx.device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out.readback));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(readback)", hr));
        return false;
    }
    return true;
}

bool ensure_persistent_input(D3D12Context& ctx, PersistentInputResource& resource, uint64_t bytes, ColorPipelineResources* stats = nullptr) {
    uint64_t required = std::max<uint64_t>(align4(bytes), 4u);
    if (resource.upload && resource.gpu && resource.capacity >= required && resource.mapped) {
        if (stats) ++stats->reuses;
        return true;
    }
    if (resource.upload && resource.mapped) {
        resource.upload->Unmap(0, nullptr);
        resource.mapped = nullptr;
    }
    resource.upload.Reset();
    resource.gpu.Reset();
    resource.capacity = required;
    resource.gpu_state = D3D12_RESOURCE_STATE_COPY_DEST;

    D3D12_HEAP_PROPERTIES upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_HEAP_PROPERTIES default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC desc = buffer_desc(required);
    HRESULT hr = ctx.device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&resource.upload));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(persistent upload)", hr));
        return false;
    }
    hr = ctx.device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&resource.gpu));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(persistent input default)", hr));
        return false;
    }
    D3D12_RANGE range = {0, 0};
    hr = resource.upload->Map(0, &range, &resource.mapped);
    if (FAILED(hr) || !resource.mapped) {
        set_error(hresult_message("Map(persistent upload)", hr));
        return false;
    }
    if (stats) ++stats->allocations;
    return true;
}

bool upload_persistent_input(D3D12Context& ctx, PersistentInputResource& resource, const void* data, uint64_t bytes, ColorPipelineResources* stats = nullptr) {
    if (!ensure_persistent_input(ctx, resource, bytes, stats)) {
        return false;
    }
    std::memset(resource.mapped, 0, static_cast<size_t>(resource.capacity));
    if (data && bytes > 0) {
        std::memcpy(resource.mapped, data, static_cast<size_t>(bytes));
    }
    if (resource.gpu_state != D3D12_RESOURCE_STATE_COPY_DEST) {
        D3D12_RESOURCE_BARRIER barrier = transition_barrier(resource.gpu.Get(), resource.gpu_state, D3D12_RESOURCE_STATE_COPY_DEST);
        ctx.command_list->ResourceBarrier(1, &barrier);
        resource.gpu_state = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    ctx.command_list->CopyBufferRegion(resource.gpu.Get(), 0, resource.upload.Get(), 0, resource.capacity);
    D3D12_RESOURCE_BARRIER barrier = transition_barrier(resource.gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ctx.command_list->ResourceBarrier(1, &barrier);
    resource.gpu_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    return true;
}

bool upload_persistent_buffer(D3D12Context& ctx, PersistentInputResource& resource, const ImgKeyGpuTileBufferV1* buffer, ColorPipelineResources* stats = nullptr) {
    return upload_persistent_input(ctx, resource, buffer ? buffer->data : nullptr, buffer ? buffer->byte_size : 4u, stats);
}

bool upload_zero_persistent_buffer(D3D12Context& ctx, PersistentInputResource& resource, ColorPipelineResources* stats = nullptr) {
    uint8_t zero[4] = {};
    return upload_persistent_input(ctx, resource, zero, sizeof(zero), stats);
}

bool ensure_persistent_output(D3D12Context& ctx, PersistentOutputResource& resource, uint64_t bytes, ColorPipelineResources* stats = nullptr) {
    uint64_t required = std::max<uint64_t>(align4(bytes), 4u);
    if (resource.gpu && resource.readback && resource.capacity >= required) {
        if (stats) ++stats->reuses;
        return true;
    }
    resource.gpu.Reset();
    resource.readback.Reset();
    resource.capacity = required;
    resource.gpu_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_HEAP_PROPERTIES readback_heap = heap_properties(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC gpu_desc = buffer_desc(required, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hr = ctx.device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &gpu_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&resource.gpu));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(persistent output default)", hr));
        return false;
    }
    D3D12_RESOURCE_DESC readback_desc = buffer_desc(required);
    hr = ctx.device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&resource.readback));
    if (FAILED(hr)) {
        set_error(hresult_message("CreateCommittedResource(persistent readback)", hr));
        return false;
    }
    if (stats) ++stats->allocations;
    return true;
}

void transition_output_for_dispatch(D3D12Context& ctx, PersistentOutputResource& resource) {
    if (resource.gpu_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        D3D12_RESOURCE_BARRIER barrier = transition_barrier(resource.gpu.Get(), resource.gpu_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ctx.command_list->ResourceBarrier(1, &barrier);
        resource.gpu_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
}

void transition_output_for_readback(D3D12Context& ctx, PersistentOutputResource& resource) {
    D3D12_RESOURCE_BARRIER barriers[2] = {
        uav_barrier(resource.gpu.Get()),
        transition_barrier(resource.gpu.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    ctx.command_list->ResourceBarrier(2, barriers);
    resource.gpu_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
}

void dispatch_tile_chunks(D3D12Context& ctx, ShaderConstants constants) {
    for (uint32_t y0 = 0; y0 < constants.height; y0 += kMaxDispatchHeight) {
        uint32_t chunk_h = std::min<uint32_t>(kMaxDispatchHeight, constants.height - y0);
        for (uint32_t x0 = 0; x0 < constants.width; x0 += kMaxDispatchWidth) {
            uint32_t chunk_w = std::min<uint32_t>(kMaxDispatchWidth, constants.width - x0);
            constants.dispatch_x0 = x0;
            constants.dispatch_y0 = y0;
            constants.dispatch_width = chunk_w;
            constants.dispatch_height = chunk_h;
            ctx.command_list->SetComputeRoot32BitConstants(0, sizeof(constants) / sizeof(uint32_t), &constants, 0);
            ctx.command_list->Dispatch((chunk_w + 15u) / 16u, (chunk_h + 15u) / 16u, 1);
        }
    }
}

bool copy_rgb4_readback_to_buffer(const uint8_t* packed, const ImgKeyGpuTileBufferV1* out_rgb, uint32_t width, uint32_t height) {
    auto* dst_base = static_cast<uint8_t*>(out_rgb->data);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* dst_row = dst_base + static_cast<size_t>(y) * static_cast<size_t>(out_rgb->row_stride_bytes);
        const uint8_t* src_row = packed + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u;
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t* dst = dst_row + static_cast<size_t>(x) * static_cast<size_t>(out_rgb->pixel_stride_bytes);
            const uint8_t* src = src_row + static_cast<size_t>(x) * 4u;
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
    }
    return true;
}

bool copy_mask4_readback_to_buffer(const uint8_t* packed, const ImgKeyGpuTileBufferV1* out_mask, uint32_t width, uint32_t height) {
    auto* dst_base = static_cast<uint8_t*>(out_mask->data);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* dst_row = dst_base + static_cast<size_t>(y) * static_cast<size_t>(out_mask->row_stride_bytes);
        const uint8_t* src_row = packed + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u;
        for (uint32_t x = 0; x < width; ++x) {
            dst_row[static_cast<size_t>(x) * static_cast<size_t>(out_mask->pixel_stride_bytes)] = src_row[static_cast<size_t>(x) * 4u];
        }
    }
    return true;
}

ImgKeyGpuStatus execute_identity(D3D12Context& ctx, const ImgKeyGpuTileBufferV1* rgba, ImgKeyGpuTileBufferV1* out_rgba) {
    if (!rgba || !out_rgba) {
        set_error("identity rgba input/output buffers are required");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }
    if (rgba->struct_size != sizeof(*rgba) || out_rgba->struct_size != sizeof(*out_rgba) || rgba->version != IMGKEY_GPU_ABI_VERSION || out_rgba->version != IMGKEY_GPU_ABI_VERSION) {
        set_error("identity buffer version/size is unsupported");
        return IMGKEY_GPU_UNSUPPORTED_VERSION;
    }
    if (!rgba->data || !out_rgba->data || rgba->channels != 4 || out_rgba->channels != 4 || rgba->width != out_rgba->width || rgba->height != out_rgba->height) {
        set_error("identity buffer shape/pointers are invalid");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }
    if (rgba->dtype != IMGKEY_GPU_DTYPE_U8 || out_rgba->dtype != IMGKEY_GPU_DTYPE_U8) {
        set_error("identity buffers must be uint8 RGBA");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }
    uint32_t width = rgba->width;
    uint32_t height = rgba->height;
    if (width == 0 || height == 0 || static_cast<uint64_t>(width) * height > kMaxNativeCallPixels) {
        set_error("identity dimensions are invalid or exceed max tile pixels");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> guard(ctx.mutex);
    HRESULT hr = ctx.allocator->Reset();
    if (FAILED(hr)) { set_error(hresult_message("CommandAllocator::Reset", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    hr = ctx.command_list->Reset(ctx.allocator.Get(), ctx.identity_pso.Get());
    if (FAILED(hr)) { set_error(hresult_message("CommandList::Reset", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }

    UploadResource input;
    ReadbackResource output;
    if (!create_upload_resource(ctx, rgba, input) || !create_readback_resource(ctx, static_cast<uint64_t>(width) * height * 4u, output)) {
        return IMGKEY_GPU_EXECUTION_FAILED;
    }

    ShaderConstants constants = {};
    constants.width = width;
    constants.height = height;
    constants.rgb_row_stride = static_cast<uint32_t>(rgba->row_stride_bytes);
    constants.dispatch_width = width;
    constants.dispatch_height = height;

    ctx.command_list->SetComputeRootSignature(ctx.root_signature.Get());
    ctx.command_list->SetPipelineState(ctx.identity_pso.Get());
    ctx.command_list->SetComputeRoot32BitConstants(0, sizeof(constants) / sizeof(uint32_t), &constants, 0);
    ctx.command_list->SetComputeRootShaderResourceView(1, input.gpu->GetGPUVirtualAddress());
    ctx.command_list->SetComputeRootUnorderedAccessView(12, output.gpu->GetGPUVirtualAddress());
    ctx.command_list->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1);
    D3D12_RESOURCE_BARRIER barriers[2] = {
        uav_barrier(output.gpu.Get()),
        transition_barrier(output.gpu.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    ctx.command_list->ResourceBarrier(2, barriers);
    ctx.command_list->CopyBufferRegion(output.readback.Get(), 0, output.gpu.Get(), 0, output.size);
    hr = ctx.command_list->Close();
    if (FAILED(hr)) { set_error(hresult_message("CommandList::Close", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    ID3D12CommandList* lists[] = {ctx.command_list.Get()};
    ctx.queue->ExecuteCommandLists(1, lists);
    if (!wait_for_gpu(ctx)) { return IMGKEY_GPU_EXECUTION_FAILED; }

    void* mapped = nullptr;
    D3D12_RANGE range = {0, static_cast<SIZE_T>(static_cast<uint64_t>(width) * height * 4u)};
    hr = output.readback->Map(0, &range, &mapped);
    if (FAILED(hr) || !mapped) { set_error(hresult_message("Map(identity readback)", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    const uint8_t* packed = static_cast<const uint8_t*>(mapped);
    auto* dst_base = static_cast<uint8_t*>(out_rgba->data);
    for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(dst_base + static_cast<size_t>(y) * static_cast<size_t>(out_rgba->row_stride_bytes), packed + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u, static_cast<size_t>(width) * 4u);
    }
    D3D12_RANGE empty = {0, 0};
    output.readback->Unmap(0, &empty);
    clear_error();
    return IMGKEY_GPU_OK;
}

} // namespace

extern "C" IMGKEY_GPU_API uint32_t IMGKEY_GPU_CALL imgkey_gpu_version(void) {
    return IMGKEY_GPU_ABI_VERSION;
}

extern "C" IMGKEY_GPU_API const char* IMGKEY_GPU_CALL imgkey_gpu_last_error(void) {
    return g_last_error.c_str();
}

extern "C" IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_probe_v1(void* out_probe_json, uint32_t out_probe_json_bytes) {
    clear_error();
    DXGI_ADAPTER_DESC1 desc = {};
    std::string error;
    ComPtr<IDXGIAdapter1> adapter = select_adapter(&desc, &error);
    if (!adapter) {
        std::ostringstream json;
        json << "{\"id\":\"d3d12_compute\",\"name\":\"D3D12 compute backend\",\"available\":false,\"status\":\"unavailable\",\"reason\":\"d3d12_unavailable\",\"message\":\""
             << json_escape(error.empty() ? "No compatible D3D12 adapter was found. CPU fallback will be used." : error)
             << "\",\"capability_mask\":" << kCapabilities
              << ",\"capabilities\":[\"constant_screen\",\"screen_tile\",\"persistent_session\",\"rgb_only\",\"full_color_tile\"],\"device\":null,\"device_index\":null,\"device_count\":0,\"version\":" << IMGKEY_GPU_ABI_VERSION
              << ",\"max_tile_pixels\":" << kMaxTilePixels
              << ",\"max_dispatch_pixels\":" << kMaxDispatchPixels
              << ",\"max_native_call_pixels\":" << kMaxNativeCallPixels << "}";
        return write_json(out_probe_json, out_probe_json_bytes, json.str()) ? IMGKEY_GPU_OK : IMGKEY_GPU_INVALID_ARGUMENT;
    }

    std::string device = wide_to_utf8(desc.Description);
    std::ostringstream json;
    json << "{\"id\":\"d3d12_compute\",\"name\":\"D3D12 compute backend\",\"available\":true,\"status\":\"available\",\"reason\":null,\"message\":\"D3D12 compute backend available.\","
         << "\"capability_mask\":" << kCapabilities
         << ",\"capabilities\":[\"constant_screen\",\"screen_tile\",\"persistent_session\",\"rgb_only\",\"full_color_tile\"]"
         << ",\"device\":\"" << json_escape(device) << "\""
         << ",\"device_index\":0,\"device_count\":1,\"version\":" << IMGKEY_GPU_ABI_VERSION
         << ",\"vendor_id\":" << desc.VendorId
         << ",\"device_id\":" << desc.DeviceId
         << ",\"dedicated_video_memory\":" << static_cast<unsigned long long>(desc.DedicatedVideoMemory)
          << ",\"max_tile_pixels\":" << kMaxTilePixels
          << ",\"max_dispatch_pixels\":" << kMaxDispatchPixels
          << ",\"max_native_call_pixels\":" << kMaxNativeCallPixels
          << ",\"persistent_buffers\":true"
          << ",\"shader_model\":\"cs_6_0_or_cs_5_0_precompiled\"}";
    return write_json(out_probe_json, out_probe_json_bytes, json.str()) ? IMGKEY_GPU_OK : IMGKEY_GPU_INVALID_ARGUMENT;
}

extern "C" IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_create_context_v1(const ImgKeyGpuColorTileParamsV1* params, void** out_context) {
    clear_error();
    if (!out_context) {
        set_error("out_context pointer is null");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }
    *out_context = nullptr;
    if (params && (params->struct_size != sizeof(ImgKeyGpuColorTileParamsV1) || params->version != IMGKEY_GPU_ABI_VERSION)) {
        set_error("create_context params version/size is unsupported");
        return IMGKEY_GPU_UNSUPPORTED_VERSION;
    }
    std::unique_ptr<D3D12Context> ctx = std::make_unique<D3D12Context>();
    if (!initialize_context(*ctx)) {
        return IMGKEY_GPU_BACKEND_UNAVAILABLE;
    }
    *out_context = ctx.release();
    return IMGKEY_GPU_OK;
}

extern "C" IMGKEY_GPU_API void IMGKEY_GPU_CALL imgkey_gpu_destroy_context_v1(void* context) {
    auto* ctx = static_cast<D3D12Context*>(context);
    delete ctx;
}

extern "C" IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_identity_rgba_v1(
    void* context,
    const ImgKeyGpuTileBufferV1* rgba,
    ImgKeyGpuTileBufferV1* out_rgba) {
    clear_error();
    auto* ctx = static_cast<D3D12Context*>(context);
    if (!ctx) {
        set_error("D3D12 context is null");
        return IMGKEY_GPU_BACKEND_UNAVAILABLE;
    }
    return execute_identity(*ctx, rgba, out_rgba);
}

extern "C" IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_process_color_tile_v1(
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
    ImgKeyGpuTileBufferV1* out_repair_mask) {
    clear_error();
    auto* ctx = static_cast<D3D12Context*>(context);
    if (!ctx) {
        set_error("D3D12 context is null");
        if (params) { params->status = IMGKEY_GPU_BACKEND_UNAVAILABLE; params->fallback_reason = IMGKEY_GPU_FALLBACK_BACKEND_UNAVAILABLE; }
        return IMGKEY_GPU_BACKEND_UNAVAILABLE;
    }
    if (!validate_params(params)) {
        return params && params->status == IMGKEY_GPU_UNSUPPORTED_VERSION ? IMGKEY_GPU_UNSUPPORTED_VERSION : static_cast<ImgKeyGpuStatus>(params ? params->status : IMGKEY_GPU_INVALID_ARGUMENT);
    }
    uint32_t width = rgb ? rgb->width : 0;
    uint32_t height = rgb ? rgb->height : 0;
    if (width == 0 || height == 0 || static_cast<uint64_t>(width) * height > kMaxNativeCallPixels) {
        params->status = IMGKEY_GPU_INVALID_ARGUMENT;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_BAD_SHAPE;
        set_error("tile dimensions are invalid or exceed max native call pixels");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }
    bool needs_screen_tile = (params->required_capabilities & IMGKEY_GPU_CAP_SCREEN_TILE) != 0;
    if (!validate_buffer(rgb, "rgb", 3, width, height, false, false, params) ||
        !validate_buffer(alpha, "alpha", 1, width, height, false, false, params) ||
        !validate_buffer(background_mask, "background_mask", 1, width, height, true, false, params) ||
        !validate_buffer(edge_mask, "edge_mask", 1, width, height, true, false, params) ||
        !validate_buffer(probability, "probability", 1, width, height, false, false, params) ||
        !validate_buffer(fringe_mask, "fringe_mask", 1, width, height, false, false, params) ||
        !validate_buffer(foreground_ref_rgb, "foreground_ref_rgb", 3, width, height, false, false, params) ||
        !validate_buffer(foreground_ref_valid, "foreground_ref_valid", 1, width, height, true, false, params) ||
        !validate_buffer(out_rgb, "out_rgb", 3, width, height, false, true, params) ||
        !validate_buffer(out_repair_mask, "out_repair_mask", 1, width, height, false, true, params)) {
        return static_cast<ImgKeyGpuStatus>(params->status);
    }
    if (needs_screen_tile && !validate_buffer(screen_tile, "screen_tile", 3, width, height, false, false, params)) {
        return static_cast<ImgKeyGpuStatus>(params->status);
    }

    std::lock_guard<std::mutex> guard(ctx->mutex);
    HRESULT hr = ctx->allocator->Reset();
    if (FAILED(hr)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("CommandAllocator::Reset", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    hr = ctx->command_list->Reset(ctx->allocator.Get(), ctx->color_pso.Get());
    if (FAILED(hr)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("CommandList::Reset", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }

    ColorPipelineResources& resources = ctx->color_resources;
    uint64_t output_bytes = static_cast<uint64_t>(width) * height * 4u;
    if (!upload_persistent_buffer(*ctx, resources.rgb, rgb, &resources) ||
        !upload_persistent_buffer(*ctx, resources.alpha, alpha, &resources) ||
        !upload_persistent_buffer(*ctx, resources.background, background_mask, &resources) ||
        !upload_persistent_buffer(*ctx, resources.edge, edge_mask, &resources) ||
        !upload_persistent_buffer(*ctx, resources.probability, probability, &resources) ||
        !upload_persistent_buffer(*ctx, resources.fringe, fringe_mask, &resources) ||
        !(needs_screen_tile ? upload_persistent_buffer(*ctx, resources.screen, screen_tile, &resources) : upload_zero_persistent_buffer(*ctx, resources.screen, &resources)) ||
        !upload_persistent_buffer(*ctx, resources.foreground_rgb, foreground_ref_rgb, &resources) ||
        !upload_persistent_buffer(*ctx, resources.foreground_valid, foreground_ref_valid, &resources) ||
        !upload_persistent_buffer(*ctx, resources.transition_rgb, foreground_ref_rgb, &resources) ||
        !upload_persistent_buffer(*ctx, resources.transition_valid, foreground_ref_valid, &resources) ||
        !ensure_persistent_output(*ctx, resources.out_rgb, output_bytes, &resources) ||
        !ensure_persistent_output(*ctx, resources.out_mask, output_bytes, &resources)) {
        params->status = IMGKEY_GPU_EXECUTION_FAILED;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED;
        return IMGKEY_GPU_EXECUTION_FAILED;
    }
    transition_output_for_dispatch(*ctx, resources.out_rgb);
    transition_output_for_dispatch(*ctx, resources.out_mask);

    uint32_t screen_rgb = static_cast<uint32_t>(params->screen_r) | (static_cast<uint32_t>(params->screen_g) << 8u) | (static_cast<uint32_t>(params->screen_b) << 16u);
    uint32_t clip_limit = static_cast<uint32_t>(std::max<int>(64, std::min<int>(255, static_cast<int>(std::lround(std::clamp(params->clip_foreground, 0.0f, 1.0f) * 255.0f)) + 32)));
    ShaderConstants constants = {};
    constants.width = width;
    constants.height = height;
    constants.has_screen_tile = needs_screen_tile ? 1u : 0u;
    constants.screen_rgb = screen_rgb;
    constants.rgb_row_stride = static_cast<uint32_t>(rgb->row_stride_bytes);
    constants.alpha_row_stride = static_cast<uint32_t>(alpha->row_stride_bytes);
    constants.mask_row_stride = static_cast<uint32_t>(background_mask->row_stride_bytes);
    constants.screen_row_stride = needs_screen_tile ? static_cast<uint32_t>(screen_tile->row_stride_bytes) : 4u;
    constants.foreground_row_stride = static_cast<uint32_t>(foreground_ref_rgb->row_stride_bytes);
    constants.transition_row_stride = static_cast<uint32_t>(foreground_ref_rgb->row_stride_bytes);
    constants.clip_foreground_limit = clip_limit;
    constants.transition_alpha_min = std::min<uint32_t>(255u, params->transition_alpha_min);
    constants.transition_alpha_max = std::min<uint32_t>(255u, params->transition_alpha_max);
    constants.dispatch_width = width;
    constants.dispatch_height = height;
    constants.transition_enabled = 1u;
    constants.transition_reference_enabled = 1u;
    constants.foreground_reference_pull = std::clamp(params->foreground_reference_pull, 0.0f, 1.0f);
    constants.key_vector_despill = std::clamp(params->key_vector_despill, 0.0f, 1.0f);
    constants.preserve_foreground_luma = std::clamp(params->preserve_foreground_luma, 0.0f, 1.0f);
    constants.transition_spill_threshold = params->transition_spill_threshold;
    constants.transition_reconstruction_error = std::max(params->transition_reconstruction_error, 0.0f);

    ctx->command_list->SetComputeRootSignature(ctx->root_signature.Get());
    ctx->command_list->SetPipelineState(ctx->color_pso.Get());
    ctx->command_list->SetComputeRootShaderResourceView(1, resources.rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(2, resources.alpha.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(3, resources.background.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(4, resources.edge.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(5, resources.probability.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(6, resources.fringe.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(7, resources.screen.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(8, resources.foreground_rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(9, resources.foreground_valid.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(10, resources.transition_rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(11, resources.transition_valid.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootUnorderedAccessView(12, resources.out_rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootUnorderedAccessView(13, resources.out_mask.gpu->GetGPUVirtualAddress());
    dispatch_tile_chunks(*ctx, constants);

    transition_output_for_readback(*ctx, resources.out_rgb);
    transition_output_for_readback(*ctx, resources.out_mask);
    ctx->command_list->CopyBufferRegion(resources.out_rgb.readback.Get(), 0, resources.out_rgb.gpu.Get(), 0, output_bytes);
    ctx->command_list->CopyBufferRegion(resources.out_mask.readback.Get(), 0, resources.out_mask.gpu.Get(), 0, output_bytes);
    hr = ctx->command_list->Close();
    if (FAILED(hr)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("CommandList::Close", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    ID3D12CommandList* lists[] = {ctx->command_list.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    if (!wait_for_gpu(*ctx)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; return IMGKEY_GPU_EXECUTION_FAILED; }

    void* mapped_rgb = nullptr;
    void* mapped_mask = nullptr;
    D3D12_RANGE read_range = {0, static_cast<SIZE_T>(output_bytes)};
    hr = resources.out_rgb.readback->Map(0, &read_range, &mapped_rgb);
    if (FAILED(hr) || !mapped_rgb) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("Map(rgb readback)", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    hr = resources.out_mask.readback->Map(0, &read_range, &mapped_mask);
    if (FAILED(hr) || !mapped_mask) {
        D3D12_RANGE empty = {0, 0};
        resources.out_rgb.readback->Unmap(0, &empty);
        params->status = IMGKEY_GPU_EXECUTION_FAILED;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED;
        set_error(hresult_message("Map(mask readback)", hr));
        return IMGKEY_GPU_EXECUTION_FAILED;
    }
    copy_rgb4_readback_to_buffer(static_cast<const uint8_t*>(mapped_rgb), out_rgb, width, height);
    copy_mask4_readback_to_buffer(static_cast<const uint8_t*>(mapped_mask), out_repair_mask, width, height);
    D3D12_RANGE empty = {0, 0};
    resources.out_mask.readback->Unmap(0, &empty);
    resources.out_rgb.readback->Unmap(0, &empty);

    params->status = IMGKEY_GPU_OK;
    params->fallback_reason = IMGKEY_GPU_FALLBACK_NONE;
    clear_error();
    return IMGKEY_GPU_OK;
}

extern "C" IMGKEY_GPU_API ImgKeyGpuStatus IMGKEY_GPU_CALL imgkey_gpu_process_color_tile_v2(
    void* context,
    ImgKeyGpuColorTileParamsV2* params,
    const ImgKeyGpuTileBufferV1* rgb,
    const ImgKeyGpuTileBufferV1* alpha,
    const ImgKeyGpuTileBufferV1* background_mask,
    const ImgKeyGpuTileBufferV1* edge_mask,
    const ImgKeyGpuTileBufferV1* probability,
    const ImgKeyGpuTileBufferV1* fringe_mask,
    const ImgKeyGpuTileBufferV1* screen_tile,
    const ImgKeyGpuTileBufferV1* nearest_inner_rgb,
    const ImgKeyGpuTileBufferV1* nearest_inner_valid,
    const ImgKeyGpuTileBufferV1* transition_ref_rgb,
    const ImgKeyGpuTileBufferV1* transition_ref_valid,
    ImgKeyGpuTileBufferV1* out_rgb,
    ImgKeyGpuTileBufferV1* out_repair_mask) {
    clear_error();
    auto* ctx = static_cast<D3D12Context*>(context);
    if (!ctx) {
        set_error("D3D12 context is null");
        if (params) { params->status = IMGKEY_GPU_BACKEND_UNAVAILABLE; params->fallback_reason = IMGKEY_GPU_FALLBACK_BACKEND_UNAVAILABLE; }
        return IMGKEY_GPU_BACKEND_UNAVAILABLE;
    }
    if (!validate_params_v2(params)) {
        return params && params->status == IMGKEY_GPU_UNSUPPORTED_VERSION ? IMGKEY_GPU_UNSUPPORTED_VERSION : static_cast<ImgKeyGpuStatus>(params ? params->status : IMGKEY_GPU_INVALID_ARGUMENT);
    }
    if ((params->required_capabilities & IMGKEY_GPU_CAP_FULL_COLOR_TILE) == 0) {
        params->status = IMGKEY_GPU_UNSUPPORTED_CAPABILITY;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_UNSUPPORTED_CAPABILITY;
        set_error("full_color_tile capability is required for color tile v2");
        return IMGKEY_GPU_UNSUPPORTED_CAPABILITY;
    }
    uint32_t width = rgb ? rgb->width : 0;
    uint32_t height = rgb ? rgb->height : 0;
    if (width == 0 || height == 0 || static_cast<uint64_t>(width) * height > kMaxNativeCallPixels) {
        params->status = IMGKEY_GPU_INVALID_ARGUMENT;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_BAD_SHAPE;
        set_error("tile dimensions are invalid or exceed max native call pixels");
        return IMGKEY_GPU_INVALID_ARGUMENT;
    }
    bool needs_screen_tile = (params->required_capabilities & IMGKEY_GPU_CAP_SCREEN_TILE) != 0;
    if (!validate_buffer_v2(rgb, "rgb", 3, width, height, false, false, params) ||
        !validate_buffer_v2(alpha, "alpha", 1, width, height, false, false, params) ||
        !validate_buffer_v2(background_mask, "background_mask", 1, width, height, true, false, params) ||
        !validate_buffer_v2(edge_mask, "edge_mask", 1, width, height, true, false, params) ||
        !validate_buffer_v2(probability, "probability", 1, width, height, false, false, params) ||
        !validate_buffer_v2(fringe_mask, "fringe_mask", 1, width, height, false, false, params) ||
        !validate_buffer_v2(nearest_inner_rgb, "nearest_inner_rgb", 3, width, height, false, false, params) ||
        !validate_buffer_v2(nearest_inner_valid, "nearest_inner_valid", 1, width, height, true, false, params) ||
        !validate_buffer_v2(transition_ref_rgb, "transition_ref_rgb", 3, width, height, false, false, params) ||
        !validate_buffer_v2(transition_ref_valid, "transition_ref_valid", 1, width, height, true, false, params) ||
        !validate_buffer_v2(out_rgb, "out_rgb", 3, width, height, false, true, params) ||
        !validate_buffer_v2(out_repair_mask, "out_repair_mask", 1, width, height, false, true, params)) {
        return static_cast<ImgKeyGpuStatus>(params->status);
    }
    if (needs_screen_tile && !validate_buffer_v2(screen_tile, "screen_tile", 3, width, height, false, false, params)) {
        return static_cast<ImgKeyGpuStatus>(params->status);
    }

    std::lock_guard<std::mutex> guard(ctx->mutex);
    HRESULT hr = ctx->allocator->Reset();
    if (FAILED(hr)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("CommandAllocator::Reset", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    hr = ctx->command_list->Reset(ctx->allocator.Get(), ctx->full_color_pso.Get());
    if (FAILED(hr)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("CommandList::Reset", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }

    ColorPipelineResources& resources = ctx->color_resources;
    uint64_t output_bytes = static_cast<uint64_t>(width) * height * 4u;
    if (!upload_persistent_buffer(*ctx, resources.rgb, rgb, &resources) ||
        !upload_persistent_buffer(*ctx, resources.alpha, alpha, &resources) ||
        !upload_persistent_buffer(*ctx, resources.background, background_mask, &resources) ||
        !upload_persistent_buffer(*ctx, resources.edge, edge_mask, &resources) ||
        !upload_persistent_buffer(*ctx, resources.probability, probability, &resources) ||
        !upload_persistent_buffer(*ctx, resources.fringe, fringe_mask, &resources) ||
        !(needs_screen_tile ? upload_persistent_buffer(*ctx, resources.screen, screen_tile, &resources) : upload_zero_persistent_buffer(*ctx, resources.screen, &resources)) ||
        !upload_persistent_buffer(*ctx, resources.foreground_rgb, nearest_inner_rgb, &resources) ||
        !upload_persistent_buffer(*ctx, resources.foreground_valid, nearest_inner_valid, &resources) ||
        !upload_persistent_buffer(*ctx, resources.transition_rgb, transition_ref_rgb, &resources) ||
        !upload_persistent_buffer(*ctx, resources.transition_valid, transition_ref_valid, &resources) ||
        !ensure_persistent_output(*ctx, resources.out_rgb, output_bytes, &resources) ||
        !ensure_persistent_output(*ctx, resources.out_mask, output_bytes, &resources)) {
        params->status = IMGKEY_GPU_EXECUTION_FAILED;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED;
        return IMGKEY_GPU_EXECUTION_FAILED;
    }
    transition_output_for_dispatch(*ctx, resources.out_rgb);
    transition_output_for_dispatch(*ctx, resources.out_mask);

    uint32_t screen_rgb = static_cast<uint32_t>(params->screen_r) | (static_cast<uint32_t>(params->screen_g) << 8u) | (static_cast<uint32_t>(params->screen_b) << 16u);
    uint32_t clip_limit = static_cast<uint32_t>(std::max<int>(64, std::min<int>(255, static_cast<int>(std::lround(std::clamp(params->clip_foreground, 0.0f, 1.0f) * 255.0f)) + 32)));
    ShaderConstants constants = {};
    constants.width = width;
    constants.height = height;
    constants.has_screen_tile = needs_screen_tile ? 1u : 0u;
    constants.screen_rgb = screen_rgb;
    constants.rgb_row_stride = static_cast<uint32_t>(rgb->row_stride_bytes);
    constants.alpha_row_stride = static_cast<uint32_t>(alpha->row_stride_bytes);
    constants.mask_row_stride = static_cast<uint32_t>(background_mask->row_stride_bytes);
    constants.screen_row_stride = needs_screen_tile ? static_cast<uint32_t>(screen_tile->row_stride_bytes) : 4u;
    constants.foreground_row_stride = static_cast<uint32_t>(nearest_inner_rgb->row_stride_bytes);
    constants.transition_row_stride = static_cast<uint32_t>(transition_ref_rgb->row_stride_bytes);
    constants.clip_foreground_limit = clip_limit;
    constants.transition_alpha_min = std::min<uint32_t>(255u, params->transition_alpha_min);
    constants.transition_alpha_max = std::min<uint32_t>(255u, params->transition_alpha_max);
    constants.dispatch_width = width;
    constants.dispatch_height = height;
    constants.transition_enabled = params->transition_enabled ? 1u : 0u;
    constants.transition_reference_enabled = params->transition_reference_enabled ? 1u : 0u;
    constants.foreground_reference_pull = std::clamp(params->foreground_reference_pull, 0.0f, 1.0f);
    constants.key_vector_despill = std::clamp(params->key_vector_despill, 0.0f, 1.0f);
    constants.preserve_foreground_luma = std::clamp(params->preserve_foreground_luma, 0.0f, 1.0f);
    constants.transition_spill_threshold = params->transition_spill_threshold;
    constants.transition_reconstruction_error = std::max(params->transition_reconstruction_error, 0.0f);
    constants.despill = std::clamp(params->despill, 0.0f, 1.0f);
    constants.decontaminate = std::clamp(params->decontaminate, 0.0f, 1.0f);
    constants.unmix_amount = std::clamp(params->unmix_amount, 0.0f, 1.0f);
    constants.edge_color_repair = std::clamp(params->edge_color_repair, 0.0f, 1.0f);
    constants.inner_color_pull = std::clamp(params->inner_color_pull, 0.0f, 1.0f);
    constants.fringe_remove = std::clamp(params->fringe_remove, 0.0f, 1.0f);
    constants.luminance_protect = std::clamp(params->luminance_protect, 0.0f, 1.0f);
    constants.clamp_key_r = std::clamp(params->clamp_key_r, 0.0f, 1.0f);
    constants.clamp_key_g = std::clamp(params->clamp_key_g, 0.0f, 1.0f);
    constants.clamp_key_b = std::clamp(params->clamp_key_b, 0.0f, 1.0f);

    ctx->command_list->SetComputeRootSignature(ctx->root_signature.Get());
    ctx->command_list->SetPipelineState(ctx->full_color_pso.Get());
    ctx->command_list->SetComputeRootShaderResourceView(1, resources.rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(2, resources.alpha.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(3, resources.background.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(4, resources.edge.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(5, resources.probability.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(6, resources.fringe.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(7, resources.screen.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(8, resources.foreground_rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(9, resources.foreground_valid.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(10, resources.transition_rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootShaderResourceView(11, resources.transition_valid.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootUnorderedAccessView(12, resources.out_rgb.gpu->GetGPUVirtualAddress());
    ctx->command_list->SetComputeRootUnorderedAccessView(13, resources.out_mask.gpu->GetGPUVirtualAddress());
    dispatch_tile_chunks(*ctx, constants);

    transition_output_for_readback(*ctx, resources.out_rgb);
    transition_output_for_readback(*ctx, resources.out_mask);
    ctx->command_list->CopyBufferRegion(resources.out_rgb.readback.Get(), 0, resources.out_rgb.gpu.Get(), 0, output_bytes);
    ctx->command_list->CopyBufferRegion(resources.out_mask.readback.Get(), 0, resources.out_mask.gpu.Get(), 0, output_bytes);
    hr = ctx->command_list->Close();
    if (FAILED(hr)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("CommandList::Close", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    ID3D12CommandList* lists[] = {ctx->command_list.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    if (!wait_for_gpu(*ctx)) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; return IMGKEY_GPU_EXECUTION_FAILED; }

    void* mapped_rgb = nullptr;
    void* mapped_mask = nullptr;
    D3D12_RANGE read_range = {0, static_cast<SIZE_T>(output_bytes)};
    hr = resources.out_rgb.readback->Map(0, &read_range, &mapped_rgb);
    if (FAILED(hr) || !mapped_rgb) { params->status = IMGKEY_GPU_EXECUTION_FAILED; params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED; set_error(hresult_message("Map(rgb readback)", hr)); return IMGKEY_GPU_EXECUTION_FAILED; }
    hr = resources.out_mask.readback->Map(0, &read_range, &mapped_mask);
    if (FAILED(hr) || !mapped_mask) {
        D3D12_RANGE empty = {0, 0};
        resources.out_rgb.readback->Unmap(0, &empty);
        params->status = IMGKEY_GPU_EXECUTION_FAILED;
        params->fallback_reason = IMGKEY_GPU_FALLBACK_EXECUTION_FAILED;
        set_error(hresult_message("Map(mask readback)", hr));
        return IMGKEY_GPU_EXECUTION_FAILED;
    }
    copy_rgb4_readback_to_buffer(static_cast<const uint8_t*>(mapped_rgb), out_rgb, width, height);
    copy_mask4_readback_to_buffer(static_cast<const uint8_t*>(mapped_mask), out_repair_mask, width, height);
    D3D12_RANGE empty = {0, 0};
    resources.out_mask.readback->Unmap(0, &empty);
    resources.out_rgb.readback->Unmap(0, &empty);

    params->status = IMGKEY_GPU_OK;
    params->fallback_reason = IMGKEY_GPU_FALLBACK_NONE;
    clear_error();
    return IMGKEY_GPU_OK;
}
