// ImgKey native GPU compute kernels.
// Compiled at build time by native/imgkey_gpu/build.ps1 and embedded into imgkey_gpu.dll.

cbuffer ImgKeyParams : register(b0)
{
    uint g_width;
    uint g_height;
    uint g_has_screen_tile;
    uint g_screen_rgb;
    uint g_rgb_row_stride;
    uint g_alpha_row_stride;
    uint g_mask_row_stride;
    uint g_screen_row_stride;
    uint g_foreground_row_stride;
    uint g_transition_row_stride;
    uint g_clip_foreground_limit;
    uint g_transition_alpha_min;
    uint g_transition_alpha_max;
    uint g_dispatch_x0;
    uint g_dispatch_y0;
    uint g_dispatch_width;
    uint g_dispatch_height;
    uint g_transition_enabled;
    uint g_transition_reference_enabled;
    uint g_reserved0;
    float g_foreground_reference_pull;
    float g_key_vector_despill;
    float g_preserve_foreground_luma;
    float g_transition_spill_threshold;
    float g_transition_reconstruction_error;
    float g_despill;
    float g_decontaminate;
    float g_unmix_amount;
    float g_edge_color_repair;
    float g_inner_color_pull;
    float g_fringe_remove;
    float g_luminance_protect;
    float g_clamp_key_r;
    float g_clamp_key_g;
    float g_clamp_key_b;
    float g_reserved1;
};

ByteAddressBuffer g_rgb : register(t0);
ByteAddressBuffer g_alpha : register(t1);
ByteAddressBuffer g_background_mask : register(t2);
ByteAddressBuffer g_edge_mask : register(t3);
ByteAddressBuffer g_probability : register(t4);
ByteAddressBuffer g_fringe_mask : register(t5);
ByteAddressBuffer g_screen_tile : register(t6);
ByteAddressBuffer g_foreground_ref_rgb : register(t7);
ByteAddressBuffer g_foreground_ref_valid : register(t8);
ByteAddressBuffer g_transition_ref_rgb : register(t9);
ByteAddressBuffer g_transition_ref_valid : register(t10);

RWByteAddressBuffer g_out_rgb4 : register(u0);
RWByteAddressBuffer g_out_mask4 : register(u1);

uint load_u8(ByteAddressBuffer buffer, uint byte_offset)
{
    uint aligned = byte_offset & 0xfffffffcu;
    uint shift = (byte_offset & 3u) * 8u;
    return (buffer.Load(aligned) >> shift) & 0xffu;
}

float3 load_rgb_u8(ByteAddressBuffer buffer, uint row_stride, uint x, uint y)
{
    uint offset = y * row_stride + x * 3u;
    return float3(
        (float)load_u8(buffer, offset + 0u),
        (float)load_u8(buffer, offset + 1u),
        (float)load_u8(buffer, offset + 2u));
}

uint pack_rgb(float3 rgb)
{
    uint r = (uint)clamp(round(rgb.x), 0.0, 255.0);
    uint g = (uint)clamp(round(rgb.y), 0.0, 255.0);
    uint b = (uint)clamp(round(rgb.z), 0.0, 255.0);
    return r | (g << 8u) | (b << 16u);
}

uint dominant_channel(float3 value)
{
    return (value.x >= value.y && value.x >= value.z) ? 0u : ((value.y >= value.z) ? 1u : 2u);
}

float channel_value(float3 value, uint channel)
{
    return channel == 0u ? value.x : (channel == 1u ? value.y : value.z);
}

float other_max(float3 value, uint channel)
{
    if (channel == 0u)
    {
        return max(value.y, value.z);
    }
    if (channel == 1u)
    {
        return max(value.x, value.z);
    }
    return max(value.x, value.y);
}

float3 set_channel_value(float3 value, uint channel, float replacement)
{
    if (channel == 0u)
    {
        value.x = replacement;
    }
    else if (channel == 1u)
    {
        value.y = replacement;
    }
    else
    {
        value.z = replacement;
    }
    return value;
}

float srgb_to_linear_scalar(float value_u8)
{
    float v = clamp(value_u8 / 255.0, 0.0, 1.0);
    return (v <= 0.04045) ? (v / 12.92) : pow((v + 0.055) / 1.055, 2.4);
}

float3 srgb_to_linear(float3 value_u8)
{
    return float3(
        srgb_to_linear_scalar(value_u8.x),
        srgb_to_linear_scalar(value_u8.y),
        srgb_to_linear_scalar(value_u8.z));
}

float linear_to_srgb_scalar(float linear_value)
{
    float v = clamp(linear_value, 0.0, 1.0);
    return (v <= 0.0031308) ? (v * 12.92) : (1.055 * pow(v, 1.0 / 2.4) - 0.055);
}

float3 linear_to_srgb_u8(float3 linear_value)
{
    return clamp(round(float3(
        linear_to_srgb_scalar(linear_value.x),
        linear_to_srgb_scalar(linear_value.y),
        linear_to_srgb_scalar(linear_value.z)) * 255.0), 0.0, 255.0);
}

float linear_luma(float3 rgb)
{
    float3 v = clamp(rgb, 0.0, 1.0);
    return dot(v, float3(0.2126, 0.7152, 0.0722));
}

float3 match_luma_linear(float3 rgb, float target_luma)
{
    float src_luma = linear_luma(rgb);
    float scale = (src_luma > 1e-5) ? (clamp(target_luma, 0.0, 1.0) / max(src_luma, 1e-5)) : 1.0;
    scale = clamp(scale, 0.0, 4.0);
    return clamp(clamp(rgb, 0.0, 1.0) * scale, 0.0, 1.0);
}

float smoothstep_py(float edge0, float edge1, float x)
{
    if (edge1 <= edge0)
    {
        return x >= edge1 ? 1.0 : 0.0;
    }
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float compute_key_spill_strength(float3 rgb_u8, float3 screen_u8)
{
    float3 pix = clamp(rgb_u8 / 255.0, 0.0, 1.0);
    float3 key = clamp(screen_u8 / 255.0, 1e-4, 1.0);
    uint key_channel = dominant_channel(key);
    float key_value = channel_value(key, key_channel);
    float key_dom = key_value - other_max(key, key_channel);
    if (key_dom > 0.12)
    {
        float pix_key = channel_value(pix, key_channel);
        float pix_other_max = other_max(pix, key_channel);
        return clamp(max(pix_key - pix_other_max, 0.0) / max(pix_key, 1.0 / 255.0), 0.0, 1.0);
    }

    float key_luma = dot(key, float3(0.2126, 0.7152, 0.0722));
    float3 key_vec = key - key_luma.xxx;
    float norm_v = length(key_vec);
    if (norm_v < 1e-4)
    {
        return 0.0;
    }
    key_vec /= norm_v;
    float pix_luma = dot(pix, float3(0.2126, 0.7152, 0.0722));
    float3 residual = pix - pix_luma.xxx;
    return clamp(max(dot(residual, key_vec), 0.0), 0.0, 1.0);
}

void store_outputs(uint pixel_index, float3 rgb_u8, uint repair_mask)
{
    g_out_rgb4.Store(pixel_index * 4u, pack_rgb(rgb_u8));
    g_out_mask4.Store(pixel_index * 4u, repair_mask & 0xffu);
}

bool dispatch_pixel(uint3 dispatch_id, out uint x, out uint y)
{
    if (dispatch_id.x >= g_dispatch_width || dispatch_id.y >= g_dispatch_height)
    {
        return false;
    }
    x = dispatch_id.x + g_dispatch_x0;
    y = dispatch_id.y + g_dispatch_y0;
    return x < g_width && y < g_height;
}

float3 screen_color_u8_constant()
{
    return float3(
        (float)(g_screen_rgb & 0xffu),
        (float)((g_screen_rgb >> 8u) & 0xffu),
        (float)((g_screen_rgb >> 16u) & 0xffu));
}

float3 apply_vlahos_clamp_pixel(float3 rgb, float3 screen_linear, float clamp_mask)
{
    if (clamp_mask <= 0.0)
    {
        return clamp(rgb, 0.0, 1.0);
    }
    float3 out_rgb = clamp(rgb, 0.0, 1.0);
    float weight = clamp(clamp_mask * 1.50, 0.0, 1.0);
    float3 key = clamp(float3(g_clamp_key_r, g_clamp_key_g, g_clamp_key_b), 0.0, 1.0);
    uint key_channel = dominant_channel(key);
    float key_dom = channel_value(key, key_channel) - other_max(key, key_channel);
    if (key_dom > 0.12)
    {
        float current = channel_value(out_rgb, key_channel);
        float excess = max(current - other_max(out_rgb, key_channel), 0.0);
        out_rgb = set_channel_value(out_rgb, key_channel, current - excess * weight);
    }
    else
    {
        float key_luma = linear_luma(screen_linear);
        float3 key_vec = clamp(screen_linear, 0.0, 1.0) - key_luma.xxx;
        float norm_v = length(key_vec);
        if (norm_v >= 1e-4)
        {
            key_vec /= max(norm_v, 1e-4);
            float out_luma = linear_luma(out_rgb);
            float3 residual = out_rgb - out_luma.xxx;
            float excess = max(dot(residual, key_vec), 0.0);
            out_rgb -= key_vec * (excess * weight) * 0.70;
        }
    }
    return clamp(out_rgb, 0.0, 1.0);
}

float3 protect_luminance_pixel(float3 rgb, float3 original_rgb, float repair_mask)
{
    float protect = clamp(g_luminance_protect, 0.0, 1.0);
    if (protect <= 0.0 || repair_mask <= 0.0)
    {
        return clamp(rgb, 0.0, 1.0);
    }
    float src_luma = linear_luma(original_rgb);
    float out_luma = linear_luma(rgb);
    float scale = (out_luma > 1e-4) ? (src_luma / max(out_luma, 1e-4)) : 1.0;
    scale = clamp(scale, 0.70, 1.45);
    float amount = clamp(repair_mask, 0.0, 1.0) * protect;
    float3 protected_rgb = clamp(rgb * scale, 0.0, 1.0);
    return clamp(rgb * (1.0 - amount) + protected_rgb * amount, 0.0, 1.0);
}

void compute_transition_repair_pixel(
    uint x,
    uint y,
    float3 original_rgb_u8,
    uint alpha_u8,
    bool background,
    bool edge,
    uint probability_u8,
    uint fringe_u8,
    bool use_transition_ref,
    out float3 out_u8,
    out uint out_mask)
{
    out_u8 = original_rgb_u8;
    out_mask = 0u;
    if (alpha_u8 == 0u || g_transition_reference_enabled == 0u)
    {
        if (alpha_u8 == 0u)
        {
            out_u8 = float3(0.0, 0.0, 0.0);
        }
        return;
    }

    uint mask_offset = y * g_mask_row_stride + x;
    bool foreground_valid = use_transition_ref ? (load_u8(g_transition_ref_valid, mask_offset) != 0u) : (load_u8(g_foreground_ref_valid, mask_offset) != 0u);
    if (!foreground_valid)
    {
        return;
    }

    float3 screen_color_u8 = screen_color_u8_constant();
    float spill_strength = compute_key_spill_strength(original_rgb_u8, screen_color_u8);
    uint alpha_min_v = min(g_transition_alpha_min, g_transition_alpha_max);
    uint alpha_max_v = max(g_transition_alpha_min, g_transition_alpha_max);
    bool foreground_core = (alpha_u8 >= 250u) && (!background) && (probability_u8 <= g_clip_foreground_limit) && (fringe_u8 <= 24u);
    bool semi = (alpha_u8 >= alpha_min_v) && (alpha_u8 <= alpha_max_v);
    bool protected_semi = semi && (alpha_u8 < 240u);
    bool live = (alpha_u8 > 0u) && (!background);
    bool live_edge = edge && live;
    bool live_fringe = (fringe_u8 > 0u) && live;
    bool protected_core_fringe = (fringe_u8 > 24u) && live;
    bool live_spill = (spill_strength > g_transition_spill_threshold) && live;
    bool eligible = semi || live_edge || live_fringe || live_spill;
    bool near_opaque_core = (alpha_u8 >= 240u) && (!background) && (fringe_u8 <= 24u);
    bool protected_core = (foreground_core || near_opaque_core) && (alpha_u8 >= 240u);
    bool core_allowed = (!protected_core) || protected_semi || protected_core_fringe;
    eligible = live && eligible && core_allowed && foreground_valid;
    if (!eligible)
    {
        return;
    }

    float3 foreground_rgb_u8 = use_transition_ref ? load_rgb_u8(g_transition_ref_rgb, g_transition_row_stride, x, y) : load_rgb_u8(g_foreground_ref_rgb, g_foreground_row_stride, x, y);
    float3 screen_u8 = (g_has_screen_tile != 0u) ? load_rgb_u8(g_screen_tile, g_screen_row_stride, x, y) : screen_color_u8;
    float3 source_linear = srgb_to_linear(original_rgb_u8);
    float3 foreground_linear = srgb_to_linear(foreground_rgb_u8);
    float3 screen_linear = srgb_to_linear(screen_u8);
    float alpha_f = (float)alpha_u8 / 255.0;
    float safe_alpha = max(alpha_f, 1.0 / 255.0);
    float3 foreground_est = (source_linear - (1.0 - alpha_f) * screen_linear) / safe_alpha;
    foreground_est = clamp(foreground_est, 0.0, 1.0);

    float3 recon = alpha_f * foreground_est + (1.0 - alpha_f) * screen_linear;
    float recon_error = length(source_linear - recon);
    float reconstruction_limit = max(g_transition_reconstruction_error * 1.25, 1e-4);
    if (recon_error > reconstruction_limit)
    {
        return;
    }

    float key_luma = linear_luma(screen_linear);
    float3 key_vec = clamp(screen_linear, 0.0, 1.0) - key_luma.xxx;
    float key_norm = length(key_vec);
    bool key_vec_valid = key_norm >= 1e-5;
    key_vec = key_vec_valid ? (key_vec / max(key_norm, 1e-5)) : float3(0.0, 0.0, 0.0);
    float foreground_luma = linear_luma(foreground_est);
    float reference_luma = linear_luma(foreground_linear);
    float3 foreground_chroma = foreground_est - foreground_luma.xxx;
    float vector_spill = max(dot(foreground_chroma, key_vec), 0.0);
    vector_spill = key_vec_valid ? vector_spill : 0.0;

    float edge_strength = clamp(alpha_f * (1.0 - alpha_f) * 4.0, 0.0, 1.0);
    edge_strength = max(edge_strength, edge ? 0.45 : 0.0);
    float fringe_signal = (float)fringe_u8 / 255.0;
    float near_screen = ((float)probability_u8 / 255.0) * clamp(1.0 - alpha_f, 0.0, 1.0);
    float spill_gate = max(max(clamp(spill_strength, 0.0, 1.0), smoothstep_py(0.005, 0.18, vector_spill)), max(near_screen, fringe_signal * 0.75));
    float transition_strength = max(max(edge_strength, fringe_signal), near_screen);
    float repair_strength = clamp(transition_strength * max(spill_gate, 0.35), 0.0, 1.0);
    if (repair_strength <= 0.0)
    {
        return;
    }

    float3 cleaned = foreground_est;
    float despill_amount = clamp(g_key_vector_despill, 0.0, 1.0);
    if (despill_amount > 0.0)
    {
        cleaned -= key_vec * (vector_spill * despill_amount * repair_strength);
        cleaned = clamp(cleaned, 0.0, 1.0);
    }

    float pull_amount = clamp(g_foreground_reference_pull, 0.0, 1.0);
    if (pull_amount > 0.0)
    {
        float pull = clamp(repair_strength * pull_amount, 0.0, 1.0);
        if (pull > 0.0)
        {
            float3 reference_luma_matched = match_luma_linear(foreground_linear, linear_luma(cleaned));
            cleaned = cleaned * (1.0 - pull) + reference_luma_matched * pull;
        }
    }

    float luma_preserve = clamp(g_preserve_foreground_luma, 0.0, 1.0);
    if (luma_preserve > 0.0)
    {
        float preserve = clamp(repair_strength * luma_preserve, 0.0, 1.0);
        if (preserve > 0.0)
        {
            float3 luma_matched = match_luma_linear(cleaned, reference_luma);
            cleaned = cleaned * (1.0 - preserve) + luma_matched * preserve;
        }
    }

    cleaned = clamp(cleaned, 0.0, 1.0);
    float3 repaired_u8 = linear_to_srgb_u8(cleaned);
    if (repair_strength > (1.0 / 255.0))
    {
        out_u8 = repaired_u8;
    }

    float3 delta_rgb = abs(out_u8 - original_rgb_u8) / 255.0;
    float delta = max(max(delta_rgb.x, delta_rgb.y), delta_rgb.z);
    float repair_mask_f = max(repair_strength, delta);
    out_mask = (uint)clamp(round(saturate(repair_mask_f) * 255.0), 0.0, 255.0);
}

[numthreads(16, 16, 1)]
void ImgKeyIdentityCS(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint x;
    uint y;
    if (!dispatch_pixel(dispatch_id, x, y))
    {
        return;
    }
    uint src = y * g_rgb_row_stride + x * 4u;
    uint dst = (y * g_width + x) * 4u;
    uint rgba =
        load_u8(g_rgb, src + 0u) |
        (load_u8(g_rgb, src + 1u) << 8u) |
        (load_u8(g_rgb, src + 2u) << 16u) |
        (load_u8(g_rgb, src + 3u) << 24u);
    g_out_rgb4.Store(dst, rgba);
}

[numthreads(16, 16, 1)]
void ImgKeyColorTileCS(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint x;
    uint y;
    if (!dispatch_pixel(dispatch_id, x, y))
    {
        return;
    }

    uint pixel_index = y * g_width + x;
    uint mask_offset = y * g_mask_row_stride + x;
    float3 original_rgb_u8 = load_rgb_u8(g_rgb, g_rgb_row_stride, x, y);
    uint alpha_u8 = load_u8(g_alpha, y * g_alpha_row_stride + x);
    bool background = load_u8(g_background_mask, mask_offset) != 0u;
    bool edge = load_u8(g_edge_mask, mask_offset) != 0u;
    uint probability_u8 = load_u8(g_probability, mask_offset);
    uint fringe_u8 = load_u8(g_fringe_mask, mask_offset);
    float3 out_u8;
    uint out_mask;
    compute_transition_repair_pixel(x, y, original_rgb_u8, alpha_u8, background, edge, probability_u8, fringe_u8, false, out_u8, out_mask);
    store_outputs(pixel_index, out_u8, out_mask);
}

[numthreads(16, 16, 1)]
void ImgKeyFullColorTileCS(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint x;
    uint y;
    if (!dispatch_pixel(dispatch_id, x, y))
    {
        return;
    }

    uint pixel_index = y * g_width + x;
    uint mask_offset = y * g_mask_row_stride + x;
    float3 original_rgb_u8 = load_rgb_u8(g_rgb, g_rgb_row_stride, x, y);
    uint alpha_u8 = load_u8(g_alpha, y * g_alpha_row_stride + x);
    if (alpha_u8 == 0u)
    {
        store_outputs(pixel_index, float3(0.0, 0.0, 0.0), 0u);
        return;
    }

    bool background = load_u8(g_background_mask, mask_offset) != 0u;
    bool edge = load_u8(g_edge_mask, mask_offset) != 0u;
    uint probability_u8 = load_u8(g_probability, mask_offset);
    uint fringe_u8 = load_u8(g_fringe_mask, mask_offset);
    float alpha_f = (float)alpha_u8 / 255.0;
    float3 screen_color_u8 = screen_color_u8_constant();
    float3 screen_u8 = (g_has_screen_tile != 0u) ? load_rgb_u8(g_screen_tile, g_screen_row_stride, x, y) : screen_color_u8;
    float3 rgb_linear = srgb_to_linear(original_rgb_u8);
    float3 screen_linear = srgb_to_linear(screen_u8);
    float3 out_linear = rgb_linear;

    float edge_strength = clamp(alpha_f * (1.0 - alpha_f) * 4.0, 0.0, 1.0);
    edge_strength = max(edge_strength, edge ? 0.35 : 0.0);
    bool live = alpha_f > 0.001;
    bool protected_core = false;
    if (g_transition_enabled != 0u)
    {
        protected_core = (alpha_u8 >= 250u) && (!background) && (probability_u8 <= g_clip_foreground_limit) && (fringe_u8 <= 24u);
    }

    float despill_amount = clamp(g_despill, 0.0, 1.0);
    float near_screen_base = (float)probability_u8 / 255.0;
    float legacy_spill = despill_amount <= 0.0 ? 0.0 : clamp(max(edge_strength, near_screen_base * clamp(1.0 - alpha_f, 0.0, 1.0)) * despill_amount, 0.0, 1.0);
    if (!live)
    {
        legacy_spill = 0.0;
    }
    float fringe_signal = live ? ((float)fringe_u8 / 255.0) : 0.0;
    float edge_repair = clamp(g_edge_color_repair, 0.0, 1.0);
    float fringe_remove = clamp(g_fringe_remove, 0.0, 1.0);
    float decontaminate = 0.25 + 0.75 * clamp(g_decontaminate, 0.0, 1.0);

    float unmix_amount = clamp(g_unmix_amount, 0.0, 1.0) * edge_repair * decontaminate;
    if (unmix_amount > 0.0 && fringe_signal > 0.0)
    {
        float safe_alpha = max(alpha_f, 0.06);
        float3 unmixed = (rgb_linear - (1.0 - alpha_f) * screen_linear) / safe_alpha;
        unmixed = clamp(unmixed, 0.0, 1.0);
        float blend = fringe_signal * unmix_amount;
        out_linear = out_linear * (1.0 - blend) + unmixed * blend;
    }

    float clamp_signal = max(fringe_signal * despill_amount, legacy_spill * 0.40) * fringe_remove;
    out_linear = apply_vlahos_clamp_pixel(out_linear, screen_linear, clamp_signal);

    float inner_pull_amount = clamp(g_inner_color_pull, 0.0, 1.0) * edge_repair * decontaminate;
    bool nearest_valid = load_u8(g_foreground_ref_valid, mask_offset) != 0u;
    if (inner_pull_amount > 0.0 && nearest_valid)
    {
        float pull = fringe_signal * inner_pull_amount;
        if (pull > 0.0)
        {
            float3 nearest_linear = srgb_to_linear(load_rgb_u8(g_foreground_ref_rgb, g_foreground_row_stride, x, y));
            out_linear = out_linear * (1.0 - pull) + nearest_linear * pull;
        }
    }

    float spill_mask = max(legacy_spill, fringe_signal * max(despill_amount, edge_repair * decontaminate));
    out_linear = protect_luminance_pixel(out_linear, rgb_linear, spill_mask);
    if (!live)
    {
        out_linear = float3(0.0, 0.0, 0.0);
    }

    float3 rgb_out_u8 = original_rgb_u8;
    bool changed = live && (spill_mask > 0.0);
    if (protected_core)
    {
        changed = false;
    }
    if (changed)
    {
        rgb_out_u8 = linear_to_srgb_u8(out_linear);
    }

    if (g_transition_enabled != 0u)
    {
        float3 transition_u8;
        uint transition_mask;
        compute_transition_repair_pixel(x, y, original_rgb_u8, alpha_u8, background, edge, probability_u8, fringe_u8, true, transition_u8, transition_mask);
        if (live && transition_mask > 0u)
        {
            rgb_out_u8 = transition_u8;
            spill_mask = max(spill_mask, (float)transition_mask / 255.0);
        }
    }

    if (!live)
    {
        rgb_out_u8 = float3(0.0, 0.0, 0.0);
    }
    uint final_mask = (uint)clamp(round(saturate(spill_mask) * 255.0), 0.0, 255.0);
    store_outputs(pixel_index, rgb_out_u8, final_mask);
}
