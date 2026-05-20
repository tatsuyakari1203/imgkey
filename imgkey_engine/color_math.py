from __future__ import annotations

import numpy as np


_LINEAR_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _linear_luma_from_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    luma = _srgb_to_linear_f32(arr[:, :, 0].astype(np.float32) / 255.0) * 0.2126
    luma += _srgb_to_linear_f32(arr[:, :, 1].astype(np.float32) / 255.0) * 0.7152
    luma += _srgb_to_linear_f32(arr[:, :, 2].astype(np.float32) / 255.0) * 0.0722
    return np.clip(luma, 0.0, 1.0).astype(np.float32, copy=False)


def _compute_key_spill_strength(rgb: np.ndarray, screen_color: tuple[int, int, int]) -> np.ndarray:
    pix = rgb.astype(np.float32) / 255.0
    key = np.asarray(screen_color, dtype=np.float32) / 255.0
    key = np.clip(key, 1e-4, 1.0)
    key_channel = int(np.argmax(key))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
    if key_dom > 0.12:
        key_values = pix[:, :, key_channel]
        other_max = np.maximum(pix[:, :, other[0]], pix[:, :, other[1]])
        return np.clip(np.maximum(key_values - other_max, 0.0) / np.maximum(key_values, 1.0 / 255.0), 0.0, 1.0)

    # Custom-key fallback: subtract perceived luminance, then project the color
    # residual onto the screen-color residual vector. This detects magenta/cyan
    # halos without treating neutral bright edges as spill.
    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    key_luma = float(key @ luma_weights)
    key_vec = key - key_luma
    norm = float(np.linalg.norm(key_vec))
    if norm < 1e-4:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    key_vec /= norm
    pix_luma = np.sum(pix * luma_weights.reshape(1, 1, 3), axis=2)
    residual = pix - pix_luma[:, :, None]
    projection = np.sum(residual * key_vec.reshape(1, 1, 3), axis=2)
    return np.clip(np.maximum(projection, 0.0), 0.0, 1.0).astype(np.float32)


def _linear_luma(rgb_linear: np.ndarray) -> np.ndarray:
    return np.sum(np.clip(rgb_linear, 0.0, 1.0) * _LINEAR_LUMA_WEIGHTS.reshape(1, 1, 3), axis=2).astype(np.float32)


def _match_luma_linear(rgb_linear: np.ndarray, target_luma: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb_linear, dtype=np.float32), 0.0, 1.0)
    src_luma = _linear_luma(rgb)
    target = np.clip(np.asarray(target_luma, dtype=np.float32), 0.0, 1.0)
    scale = np.divide(target, np.maximum(src_luma, 1e-5), out=np.ones_like(target), where=src_luma > 1e-5)
    scale = np.clip(scale, 0.0, 4.0)
    return np.clip(rgb * scale[:, :, None], 0.0, 1.0)


def _screen_chroma_unit_vectors(screen_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key_luma = _linear_luma(screen_linear)
    key_vec = np.clip(screen_linear, 0.0, 1.0) - key_luma[:, :, None]
    norm = np.linalg.norm(key_vec, axis=2).astype(np.float32)
    valid = norm >= 1e-5
    unit = np.divide(key_vec, np.maximum(norm[:, :, None], 1e-5), out=np.zeros_like(key_vec), where=valid[:, :, None])
    return unit.astype(np.float32, copy=False), valid


def _srgb_to_linear_f32(srgb: np.ndarray) -> np.ndarray:
    srgb_f = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(srgb_f <= 0.04045, srgb_f / 12.92, np.power((srgb_f + 0.055) / 1.055, 2.4)).astype(np.float32)


def _linear_to_srgb_f32(linear: np.ndarray) -> np.ndarray:
    linear_f = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(
        linear_f <= 0.0031308,
        linear_f * 12.92,
        1.055 * np.power(linear_f, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _srgb_u8_to_linear_f32(srgb: np.ndarray) -> np.ndarray:
    return _srgb_to_linear_f32(np.asarray(srgb, dtype=np.float32) / 255.0)


def _linear_f32_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(_linear_to_srgb_f32(linear) * 255.0), 0, 255).astype(np.uint8)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))
