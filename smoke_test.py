from __future__ import annotations

import ast
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import keyer as keyer_module
from keyer import (
    KeyResult,
    KeySettings,
    _MAX_INNER_LABEL_PIXELS,
    _build_foreground_core_mask,
    _build_nearest_inner_label_map,
    _build_nearest_inner_reference_map,
    _build_tile_local_nearest_inner_reference,
    _build_tile_local_nearest_inner_rgb,
    _build_transition_repair_mask,
    _compute_key_spill_strength,
    _estimate_screen_tile,
    _foreground_reference_for_slice,
    _guided_filter_gray,
    _linear_f32_to_srgb_u8,
    _process_color_tile,
    _repair_transition_unmix,
    _srgb_u8_to_linear_f32,
    _tile_local_nearest_inner_radius,
    checkerboard_composite,
    process_chroma_key,
    process_key_image,
)


ARTIFACT_DIR = Path(".artifact") / "smoke-fixtures"
EDGE_ARTIFACT_DIR = Path(".artifact") / "edge-repair-verification"
ALGORITHM_BASELINE_DIR = Path(".artifact") / "algorithm-upgrade-baseline"
TRANSITION_UNMIX_DIAGNOSTIC_DIR = Path(".artifact") / "transition-unmix-diagnostics"
GPU_BENCHMARK_DIR = Path(".artifact") / "gpu-benchmarks"
GEOMETRIC_BENCHMARK_DIR = Path(".artifact") / "geometric-benchmark"
PERF_BASELINE_DIR = Path(".artifact") / "perf"
HEAVY_OPTIONAL_MODULES = frozenset(
    {
        "accelerate",
        "einops",
        "hugging" + "face_hub",
        "kornia",
        "numba",
        "onnxruntime",
        "onnxruntime_gpu",
        "pymatting",
        "safe" + "tensors",
        "scipy",
        "skimage",
        "timm",
        "torch",
        "torchvision",
        "trans" + "formers",
    }
)


@dataclass(frozen=True)
class DiagnosticFixture:
    name: str
    rgb: np.ndarray
    settings: KeySettings
    notes: str
    known_background_mask: np.ndarray | None = None
    foreground_core_mask: np.ndarray | None = None
    soft_edge_mask: np.ndarray | None = None
    expected_alpha: np.ndarray | None = None
    expected_foreground_rgb: np.ndarray | tuple[int, int, int] | None = None
    original_alpha: np.ndarray | None = None
    keep_mask: np.ndarray | None = None
    remove_mask: np.ndarray | None = None
    alpha_hint: np.ndarray | None = None
    diagnostic_only: bool = True


@dataclass(frozen=True)
class GeometricBenchmarkAsset:
    name: str
    alpha: np.ndarray
    foreground_rgb_template: np.ndarray
    key_color_region: np.ndarray
    feature_masks: dict[str, np.ndarray]
    primary_feature_order: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class GeometricBenchmarkCase:
    name: str
    background_name: str
    key_color: tuple[int, int, int]
    background_rgb: np.ndarray
    source_rgb: np.ndarray
    expected_alpha: np.ndarray
    expected_foreground_rgb: np.ndarray
    expected_rgba: np.ndarray
    feature_masks: dict[str, np.ndarray]
    settings: KeySettings
    notes: str


@dataclass(frozen=True)
class GeometricTuningProfile:
    name: str
    label: str
    description: str
    settings: KeySettings


@contextmanager
def _temporary_inner_label_cap(cap: int):
    previous = keyer_module._MAX_INNER_LABEL_PIXELS
    keyer_module._MAX_INNER_LABEL_PIXELS = int(cap)
    try:
        yield
    finally:
        keyer_module._MAX_INNER_LABEL_PIXELS = previous


def _disc_alpha(h: int, w: int, radius: float, feather: float = 0.0) -> np.ndarray:
    yy, xx = np.indices((h, w), dtype=np.float32)
    dist = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2)
    if feather <= 0:
        return (dist <= radius).astype(np.float32)
    return np.clip((radius + feather - dist) / max(feather * 2.0, 1e-4), 0.0, 1.0)


def _composite_rgb(background: np.ndarray, foreground: tuple[int, int, int], alpha: np.ndarray) -> np.ndarray:
    fg = np.asarray(foreground, dtype=np.float32).reshape(1, 1, 3)
    bg = background.astype(np.float32)
    rgb = bg * (1.0 - alpha[:, :, None]) + fg * alpha[:, :, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _composite_rgb_array(background: np.ndarray, foreground_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bg = background.astype(np.float32)
    fg = foreground_rgb.astype(np.float32)
    rgb = bg * (1.0 - alpha[:, :, None]) + fg * alpha[:, :, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _composite_rgb_linear(background: np.ndarray, foreground_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bg = _srgb_u8_to_linear_f32(background)
    fg = _srgb_u8_to_linear_f32(foreground_rgb)
    rgb = bg * (1.0 - alpha[:, :, None]) + fg * alpha[:, :, None]
    return _linear_f32_to_srgb_u8(rgb)


def _masks_from_alpha(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_background = alpha <= 0.002
    foreground_core = alpha >= 0.995
    soft_edge = (alpha > 0.002) & (alpha < 0.995)
    return known_background, foreground_core, soft_edge


def _draw_antialiased_alpha(
    shape: tuple[int, int],
    draw_callback,
    scale: int = 4,
) -> np.ndarray:
    h, w = shape
    mask = Image.new("L", (w * scale, h * scale), 0)
    draw_callback(ImageDraw.Draw(mask), scale)
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return np.asarray(mask.resize((w, h), resampling), dtype=np.float32) / 255.0


def _graphic_transition_settings(key_color: tuple[int, int, int]) -> KeySettings:
    return KeySettings(
        key_color=key_color,
        auto_border_sample=False,
        local_screen_model=False,
        sample_size=10,
        tolerance=0.45,
        softness=0.01,
        clip_background=0.97,
        clip_foreground=0.00,
        matte_gamma=2.20,
        core_strength=0.38,
        edge_refine_radius=12,
        erode_expand=-8,
        despill=0.70,
        decontaminate=0.50,
        luminance_restore=0.76,
        luminance_protect=0.76,
        fringe_remove=0.75,
        edge_color_repair=0.65,
        inner_color_pull=0.45,
        fringe_band_radius=3,
        screen_cleanup_strength=1.00,
        screen_cleanup_similarity=8,
    )


def _expected_foreground_array(fixture: DiagnosticFixture) -> np.ndarray | None:
    expected = fixture.expected_foreground_rgb
    if expected is None:
        return None
    if isinstance(expected, tuple):
        return np.broadcast_to(np.asarray(expected, dtype=np.uint8).reshape(1, 1, 3), fixture.rgb.shape).copy()
    if expected.shape != fixture.rgb.shape:
        raise AssertionError(f"{fixture.name}: expected foreground RGB shape mismatch")
    return expected.astype(np.uint8, copy=False)


def _expected_rgba_for_fixture(fixture: DiagnosticFixture) -> np.ndarray | None:
    if fixture.expected_alpha is None:
        return None
    fg = _expected_foreground_array(fixture)
    if fg is None:
        return None
    alpha = np.clip(fixture.expected_alpha, 0.0, 1.0)
    if fixture.original_alpha is not None:
        alpha = np.minimum(alpha, np.clip(fixture.original_alpha.astype(np.float32), 0.0, 1.0))
    alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
    rgba = np.zeros((*alpha_u8.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = fg
    rgba[:, :, 3] = alpha_u8
    rgba[alpha_u8 == 0, :3] = 0
    return rgba


def _process_fixture_result(
    fixture: DiagnosticFixture,
    settings: KeySettings | None = None,
    *,
    include_debug: bool = True,
) -> KeyResult:
    return process_key_image(
        fixture.rgb,
        settings or fixture.settings,
        fixture.original_alpha,
        keep_mask=fixture.keep_mask,
        remove_mask=fixture.remove_mask,
        alpha_hint=fixture.alpha_hint,
        include_debug=include_debug,
    )


def _fixture_masks(fixture: DiagnosticFixture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = fixture.rgb.shape[:2]
    if fixture.known_background_mask is not None:
        known_background = fixture.known_background_mask.astype(bool, copy=False)
    elif fixture.expected_alpha is not None:
        known_background = fixture.expected_alpha <= 0.002
    else:
        known_background = np.zeros(shape, dtype=bool)

    if fixture.foreground_core_mask is not None:
        foreground_core = fixture.foreground_core_mask.astype(bool, copy=False)
    elif fixture.expected_alpha is not None:
        foreground_core = fixture.expected_alpha >= 0.995
    else:
        foreground_core = np.zeros(shape, dtype=bool)

    if fixture.soft_edge_mask is not None:
        soft_edge = fixture.soft_edge_mask.astype(bool, copy=False)
    elif fixture.expected_alpha is not None:
        soft_edge = (fixture.expected_alpha > 0.002) & (fixture.expected_alpha < 0.995)
    else:
        soft_edge = np.zeros(shape, dtype=bool)
    return known_background, foreground_core, soft_edge


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _edge_fringe_fixture(
    key_color: tuple[int, int, int],
    foreground: tuple[int, int, int] = (220, 185, 120),
) -> tuple[np.ndarray, np.ndarray, KeySettings]:
    h, w = 280, 360
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = key_color
    alpha = _disc_alpha(h, w, 78, feather=8.0)
    rgb = _composite_rgb(background, foreground, alpha)
    settings = KeySettings(
        key_color=key_color,
        auto_border_sample=False,
        local_screen_model=False,
        edge_refine_radius=6,
        clip_background=0.78,
        clip_foreground=0.14,
        matte_gamma=1.0,
        fringe_band_radius=3,
    )
    return rgb, alpha, settings


def _dominant_excess(pixel: np.ndarray, key_color: tuple[int, int, int]) -> int:
    key_channel = int(np.argmax(np.asarray(key_color)))
    other = [c for c in range(3) if c != key_channel]
    px = pixel.astype(int)
    return int(px[key_channel] - max(px[other[0]], px[other[1]]))


def _luma(pixel: np.ndarray) -> float:
    return float(pixel.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))


def _find_fringe_sample(rgb: np.ndarray, result: KeyResult, key_color: tuple[int, int, int]) -> tuple[int, int]:
    if result.fringe_mask is None:
        raise AssertionError("result should include mandatory fringe_mask")
    key_channel = int(np.argmax(np.asarray(key_color)))
    other = [c for c in range(3) if c != key_channel]
    source_excess = rgb[:, :, key_channel].astype(np.int16) - np.maximum(
        rgb[:, :, other[0]].astype(np.int16),
        rgb[:, :, other[1]].astype(np.int16),
    )
    candidates = (result.alpha > 35) & (result.alpha < 225) & (result.fringe_mask > 70) & (source_excess > 18)
    ys, xs = np.nonzero(candidates)
    if ys.size == 0:
        raise AssertionError("expected at least one contaminated soft-edge fringe sample")
    best = int(np.argmax(source_excess[ys, xs]))
    return int(ys[best]), int(xs[best])


def green_flat_fixture() -> DiagnosticFixture:
    h, w = 720, 960
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (0, 220, 40)
    rgb[_disc_alpha(h, w, 180).astype(bool)] = (215, 160, 120)
    return DiagnosticFixture(
        name="green_flat",
        rgb=rgb,
        settings=KeySettings(key_color=(0, 220, 50)),
        notes="Current enforcing baseline: flat green screen with opaque foreground.",
    )


def blue_flat_fixture() -> DiagnosticFixture:
    h, w = 600, 840
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (28, 80, 232)
    rgb[_disc_alpha(h, w, 150).astype(bool)] = (230, 190, 120)
    return DiagnosticFixture(
        name="blue_flat",
        rgb=rgb,
        settings=KeySettings(key_color=(30, 80, 235)),
        notes="Diagnostic only: blue-screen equivalent of the baseline fixture.",
    )


def custom_flat_fixture() -> DiagnosticFixture:
    h, w = 600, 840
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (185, 42, 170)
    rgb[_disc_alpha(h, w, 150).astype(bool)] = (70, 190, 125)
    return DiagnosticFixture(
        name="custom_flat",
        rgb=rgb,
        settings=KeySettings(key_color=(185, 42, 170)),
        notes="Diagnostic only: custom magenta key color path.",
    )


def uneven_gradient_fixture() -> DiagnosticFixture:
    h, w = 720, 960
    x_grad = np.linspace(185, 242, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-16, 18, h, dtype=np.float32).reshape(h, 1)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(4 + y_grad * 0.15, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(x_grad + y_grad, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(35 + y_grad * 0.25, 0, 255).astype(np.uint8)
    rgb[_disc_alpha(h, w, 175).astype(bool)] = (218, 158, 120)
    return DiagnosticFixture(
        name="green_uneven_gradient",
        rgb=rgb,
        settings=KeySettings(key_color=(0, 220, 50)),
        notes="Future v2 diagnostic: uneven screen should key without eating foreground.",
    )


def same_color_island_fixture() -> DiagnosticFixture:
    h, w = 720, 960
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (0, 220, 45)
    foreground = _disc_alpha(h, w, 210).astype(bool)
    rgb[foreground] = (210, 154, 118)
    yy, xx = np.indices((h, w))
    island = (xx - w // 2) ** 2 + (yy - (h // 2 - 20)) ** 2 < 58**2
    rgb[island] = (4, 206, 52)
    return DiagnosticFixture(
        name="green_same_color_island",
        rgb=rgb,
        settings=KeySettings(key_color=(0, 220, 50)),
        notes="Future v2 diagnostic: border-disconnected foreground island near key color should be preserved.",
    )


def antialiased_edge_fixture() -> DiagnosticFixture:
    h, w = 720, 960
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = (0, 220, 42)
    alpha = _disc_alpha(h, w, 185, feather=8.0)
    rgb = _composite_rgb(background, (214, 160, 122), alpha)
    return DiagnosticFixture(
        name="green_antialiased_edge",
        rgb=rgb,
        settings=KeySettings(key_color=(0, 220, 50)),
        notes="Future v2 diagnostic: semi-transparent/anti-aliased edge should retain soft alpha.",
    )


def guided_alpha_refine_fixture() -> DiagnosticFixture:
    h, w = 360, 500
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = (30, 80, 235)
    alpha = _disc_alpha(h, w, 108, feather=18.0)
    foreground = (224, 170, 118)
    rgb = _composite_rgb(background, foreground, alpha)

    yy, xx = np.indices((h, w))
    soft_edge = (alpha > 0.02) & (alpha < 0.98)
    blue_dither = np.where(((xx // 3 + yy // 2) & 1) == 0, 30, -30)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(np.int16) + blue_dither.astype(np.int16) * soft_edge, 0, 255).astype(np.uint8)
    known_background, foreground_core, _ = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="guided_alpha_refine",
        rgb=rgb,
        settings=KeySettings(
            key_color=(30, 80, 235),
            auto_border_sample=False,
            local_screen_model=False,
            edge_refine_radius=7,
            edge_softness=0.18,
            fringe_band_radius=3,
        ),
        notes="Phase 3 fixture: blue-channel edge dither should be smoothed by guided alpha refinement.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def blue_gradient_screen_fixture() -> DiagnosticFixture:
    h, w = 540, 760
    x_grad = np.linspace(-20, 20, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-14, 16, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(30 + x_grad * 0.45 + y_grad * 0.18, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(80 + x_grad * 0.35 - y_grad * 0.20, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(235 + x_grad + y_grad * 0.55, 0, 255).astype(np.uint8)
    alpha = _disc_alpha(h, w, 150, feather=12.0)
    foreground = (224, 170, 118)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="blue_gradient_screen",
        rgb=_composite_rgb(background, foreground, alpha),
        settings=KeySettings(
            key_color=(30, 80, 235),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
        ),
        notes="v5 baseline diagnostic: blue screen with horizontal/vertical illumination gradient.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def green_gradient_screen_fixture() -> DiagnosticFixture:
    h, w = 540, 760
    x_grad = np.linspace(-24, 22, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-18, 18, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(4 + x_grad * 0.12 + y_grad * 0.10, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(220 + x_grad + y_grad * 0.65, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(45 + x_grad * 0.16 - y_grad * 0.25, 0, 255).astype(np.uint8)
    alpha = _disc_alpha(h, w, 150, feather=12.0)
    foreground = (218, 162, 124)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="green_gradient_screen",
        rgb=_composite_rgb(background, foreground, alpha),
        settings=KeySettings(
            key_color=(0, 220, 50),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
        ),
        notes="v5 baseline diagnostic: green screen with shadow/highlight gradient.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def same_key_foreground_core_fixture() -> DiagnosticFixture:
    h, w = 520, 700
    key_color = (0, 220, 50)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = key_color
    alpha = _disc_alpha(h, w, 155, feather=5.0)
    foreground = np.zeros((h, w, 3), dtype=np.uint8)
    foreground[:, :] = (214, 156, 120)
    yy, xx = np.indices((h, w))
    key_like_core = ((xx - w // 2) ** 2) / float(62**2) + ((yy - (h // 2 - 8)) ** 2) / float(42**2) <= 1.0
    foreground[key_like_core] = (3, 210, 55)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    foreground_core = foreground_core | key_like_core
    return DiagnosticFixture(
        name="same_key_foreground_core",
        rgb=_composite_rgb_array(background, foreground, alpha),
        settings=KeySettings(key_color=key_color, aggressive_interior_removal=False, fringe_band_radius=3),
        notes="v5 baseline diagnostic: disconnected foreground core intentionally matches the key color.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def hair_lines_fixture() -> DiagnosticFixture:
    h, w = 420, 620
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = (0, 220, 45)
    scale = 3
    mask_image = Image.new("L", (w * scale, h * scale), 0)
    draw = ImageDraw.Draw(mask_image)
    root_x = w // 2
    root_y = h // 2 + 95
    for index in range(19):
        offset = index - 9
        x0 = (root_x + offset * 8) * scale
        y0 = root_y * scale
        x1 = (root_x + offset * 20) * scale
        y1 = (h // 2 - 125 - abs(offset) * 2) * scale
        x_mid = (root_x + offset * 13 + (-1) ** index * 20) * scale
        y_mid = (h // 2 - 12 - abs(offset) * 5) * scale
        draw.line([(x0, y0), (x_mid, y_mid), (x1, y1)], fill=245, width=max(2, scale * 2))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    alpha = np.asarray(mask_image.resize((w, h), resampling), dtype=np.float32) / 255.0
    foreground = (62, 42, 30)
    rgb = _composite_rgb(background, foreground, alpha)
    known_background = alpha <= 0.002
    foreground_core = alpha >= 0.88
    soft_edge = (alpha > 0.02) & (alpha < 0.88)
    return DiagnosticFixture(
        name="hair_lines",
        rgb=rgb,
        settings=KeySettings(
            key_color=(0, 220, 50),
            auto_border_sample=True,
            edge_refine_radius=6,
            fringe_band_radius=2,
        ),
        notes="v5 baseline diagnostic: sub-pixel anti-aliased hair-like lines over a green screen.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def semi_transparent_glass_fixture() -> DiagnosticFixture:
    h, w = 500, 700
    x_grad = np.linspace(-18, 16, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-10, 14, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(28 + x_grad * 0.25, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(78 - y_grad * 0.10, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(235 + x_grad * 0.80 + y_grad * 0.45, 0, 255).astype(np.uint8)
    yy, xx = np.indices((h, w), dtype=np.float32)
    ellipse = ((xx - w / 2.0) ** 2) / float(165**2) + ((yy - h / 2.0) ** 2) / float(118**2)
    glass = np.clip((1.16 - ellipse) / 0.24, 0.0, 1.0) * 0.48
    highlight = (np.abs((yy - h / 2.0) - 0.32 * (xx - w / 2.0)) < 4.0) & (ellipse < 0.78)
    alpha = np.maximum(glass, highlight.astype(np.float32) * 0.82)
    foreground = (226, 242, 255)
    rgb = _composite_rgb(background, foreground, alpha)
    known_background = alpha <= 0.002
    foreground_core = alpha >= 0.80
    soft_edge = (alpha > 0.02) & (alpha < 0.80)
    return DiagnosticFixture(
        name="semi_transparent_glass",
        rgb=rgb,
        settings=KeySettings(
            key_color=(30, 80, 235),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
        ),
        notes="v5 baseline diagnostic: semi-transparent glass body and bright highlight over blue screen.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def white_gray_black_composite_fixture() -> DiagnosticFixture:
    h, w = 380, 540
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = (0, 220, 48)
    alpha = _disc_alpha(h, w, 112, feather=18.0)
    yy, xx = np.indices((h, w), dtype=np.float32)
    stem = (np.abs(xx - w / 2.0) < 16.0) & (yy > h / 2.0 - 96) & (yy < h / 2.0 + 112)
    alpha = np.maximum(alpha, stem.astype(np.float32) * 0.92)
    foreground = (235, 182, 92)
    rgb = _composite_rgb(background, foreground, alpha)
    known_background = alpha <= 0.002
    foreground_core = alpha >= 0.90
    soft_edge = (alpha > 0.02) & (alpha < 0.90)
    return DiagnosticFixture(
        name="white_gray_black_composite",
        rgb=rgb,
        settings=KeySettings(
            key_color=(0, 220, 50),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
        ),
        notes="v5 baseline diagnostic: fixture used to compare black/white/gray/checkerboard composites.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def _transition_slash_fixture(key_color: tuple[int, int, int], name: str) -> DiagnosticFixture:
    h, w = 220, 320
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = key_color

    def draw_slash(draw: ImageDraw.ImageDraw, scale: int) -> None:
        draw.line(
            [(-28 * scale, (h + 18) * scale), ((w + 28) * scale, -18 * scale)],
            fill=255,
            width=24 * scale,
        )

    alpha = _draw_antialiased_alpha((h, w), draw_slash, scale=4)
    foreground = np.zeros_like(background)
    foreground[:, :] = (228, 18, 16)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name=name,
        rgb=_composite_rgb_linear(background, foreground, alpha),
        settings=_graphic_transition_settings(key_color),
        notes="v7 transition baseline: red anti-aliased slash physically composited over a key plate.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def red_slash_green_transition_fixture() -> DiagnosticFixture:
    return _transition_slash_fixture((0, 220, 50), "transition_red_slash_green")


def red_slash_blue_transition_fixture() -> DiagnosticFixture:
    return _transition_slash_fixture((30, 80, 235), "transition_red_slash_blue")


def white_black_barcode_transition_fixture() -> DiagnosticFixture:
    h, w = 170, 300
    key_color = (30, 80, 235)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = key_color
    alpha = np.zeros((h, w), dtype=np.float32)
    foreground = np.zeros_like(background)
    bar_xs = [42, 45, 51, 59, 72, 75, 90, 104, 107, 123, 139, 143, 160, 178, 181, 199, 217, 221, 236, 251]
    for index, x in enumerate(bar_xs):
        color = (255, 255, 255) if index % 2 == 0 else (0, 0, 0)
        alpha[32:138, x : x + 1] = 1.0
        foreground[32:138, x : x + 1] = color
    # A few horizontal one-pixel strokes make the fixture text-like without font dependencies.
    for index, y in enumerate((52, 76, 101, 126)):
        color = (255, 255, 255) if index % 2 == 0 else (0, 0, 0)
        alpha[y : y + 1, 58:244] = 1.0
        foreground[y : y + 1, 58:244] = color
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="transition_white_black_1px_lines",
        rgb=_composite_rgb_linear(background, foreground, alpha),
        settings=_graphic_transition_settings(key_color),
        notes="v7 transition baseline: white/black one-pixel barcode/text strokes over blue key.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def black_tape_edge_transition_fixture() -> DiagnosticFixture:
    h, w = 190, 320
    key_color = (0, 220, 50)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = key_color

    def draw_tape(draw: ImageDraw.ImageDraw, scale: int) -> None:
        points = [
            (30 * scale, 76 * scale),
            (286 * scale, 48 * scale),
            (296 * scale, 98 * scale),
            (39 * scale, 128 * scale),
        ]
        draw.polygon(points, fill=255)

    alpha = _draw_antialiased_alpha((h, w), draw_tape, scale=4)
    foreground = np.zeros_like(background)
    foreground[:, :] = (0, 0, 0)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="transition_black_tape_edge",
        rgb=_composite_rgb_linear(background, foreground, alpha),
        settings=_graphic_transition_settings(key_color),
        notes="v7 transition baseline: black tape edge with anti-aliased key-color transition pixels.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def source_alpha_cap_transition_fixture() -> DiagnosticFixture:
    h, w = 180, 260
    key_color = (30, 80, 235)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = key_color
    alpha = _disc_alpha(h, w, 62, feather=9.0)
    foreground = np.zeros_like(background)
    foreground[:, :] = (230, 22, 18)
    original_alpha = np.ones((h, w), dtype=np.float32)
    original_alpha[:, 120:170] = 0.50
    original_alpha[28:62, 34:92] = 0.0
    rgb = _composite_rgb_linear(background, foreground, alpha)
    rgb[original_alpha <= 0.0] = (255, 32, 16)
    expected_alpha = np.minimum(alpha, original_alpha)
    known_background, foreground_core, soft_edge = _masks_from_alpha(expected_alpha)
    return DiagnosticFixture(
        name="transition_source_alpha_cap",
        rgb=rgb,
        settings=_graphic_transition_settings(key_color),
        notes="v7 transition baseline: source alpha caps recovered alpha and transparent RGB must be zeroed.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=expected_alpha,
        expected_foreground_rgb=foreground,
        original_alpha=original_alpha,
    )


def manual_keep_transition_fixture() -> DiagnosticFixture:
    base = red_slash_green_transition_fixture()
    h, w = base.rgb.shape[:2]
    yy, xx = np.indices((h, w))
    band = (xx >= 88) & (xx <= 226) & (yy >= 34) & (yy <= 176)
    keep = band & (base.soft_edge_mask.astype(bool) | base.foreground_core_mask.astype(bool))
    return replace(
        base,
        name="transition_manual_keep_transition_core",
        notes="v7 regression: manual keep mask crosses transition and core pixels and must remain authoritative.",
        keep_mask=keep.astype(np.uint8) * 255,
    )


def manual_remove_transition_fixture() -> DiagnosticFixture:
    base = red_slash_green_transition_fixture()
    h, w = base.rgb.shape[:2]
    yy, xx = np.indices((h, w))
    band = (xx >= 88) & (xx <= 236) & (yy >= 30) & (yy <= 184)
    remove = band & (base.soft_edge_mask.astype(bool) | base.known_background_mask.astype(bool))
    return replace(
        base,
        name="transition_manual_remove_transition_background",
        notes="v7 regression: manual remove mask crosses transition/background pixels and must force alpha/RGB zero.",
        remove_mask=remove.astype(np.uint8) * 255,
    )


def transition_unmix_baseline_fixtures() -> list[DiagnosticFixture]:
    return [
        red_slash_green_transition_fixture(),
        red_slash_blue_transition_fixture(),
        white_black_barcode_transition_fixture(),
        black_tape_edge_transition_fixture(),
        source_alpha_cap_transition_fixture(),
    ]


def transition_unmix_manual_mask_fixtures() -> list[DiagnosticFixture]:
    return [manual_keep_transition_fixture(), manual_remove_transition_fixture()]


def transition_unmix_diagnostic_fixtures() -> list[DiagnosticFixture]:
    return transition_unmix_baseline_fixtures() + transition_unmix_manual_mask_fixtures()


def large_tile_gradient_runtime_fixture() -> DiagnosticFixture:
    h, w = 1152, 1536
    x_grad = np.linspace(-24, 26, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-18, 20, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(4 + y_grad * 0.10, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(220 + x_grad + y_grad * 0.55, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(46 + x_grad * 0.15 - y_grad * 0.20, 0, 255).astype(np.uint8)
    alpha = _disc_alpha(h, w, 330, feather=9.0)
    foreground = (220, 164, 128)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="large_tile_gradient_runtime",
        rgb=_composite_rgb(background, foreground, alpha),
        settings=KeySettings(
            key_color=(0, 220, 50),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
            tile_size=384,
            tile_overlap=56,
        ),
        notes="v5 baseline diagnostic: large gradient screen fixture for tiled-vs-full comparisons.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def tile_local_diagonal_gradient_fixture() -> DiagnosticFixture:
    h, w = 540, 760
    x_grad = np.linspace(-70, 70, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-45, 45, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(3 + x_grad * 0.06 + y_grad * 0.05, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(220 + x_grad + y_grad * 0.45, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(45 + x_grad * 0.10 - y_grad * 0.12, 0, 255).astype(np.uint8)
    alpha = _disc_alpha(h, w, 150, feather=14.0)
    foreground = np.zeros_like(background)
    foreground[:, :] = (218, 162, 124)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="tile_local_diagonal_gradient",
        rgb=_composite_rgb_linear(background, foreground, alpha),
        settings=KeySettings(
            key_color=(0, 220, 50),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
            clip_background=0.65,
            brightness_tolerance=0.55,
            tolerance=0.28,
            tile_size=173,
            tile_overlap=8,
        ),
        notes="Phase 4 fixture: diagonal green-screen illumination gradient for tile-local screen estimation.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def tile_local_shadow_gradient_fixture() -> DiagnosticFixture:
    h, w = 500, 720
    x_grad = np.linspace(-35, 35, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-80, 65, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(5 + x_grad * 0.05 + y_grad * 0.04, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(215 + x_grad * 0.20 + y_grad * 0.90, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(48 + x_grad * 0.08 - y_grad * 0.10, 0, 255).astype(np.uint8)
    alpha = _disc_alpha(h, w, 135, feather=16.0)
    foreground = np.zeros_like(background)
    foreground[:, :] = (222, 166, 126)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="tile_local_shadow_gradient",
        rgb=_composite_rgb_linear(background, foreground, alpha),
        settings=KeySettings(
            key_color=(0, 220, 50),
            auto_border_sample=True,
            edge_refine_radius=8,
            fringe_band_radius=3,
            clip_background=0.65,
            brightness_tolerance=0.55,
            tolerance=0.28,
            tile_size=173,
            tile_overlap=8,
        ),
        notes="Phase 4 fixture: green screen with vertical shadow gradient for tile-local screen estimation.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def phase4_tile_local_screen_fixtures() -> list[DiagnosticFixture]:
    return [tile_local_diagonal_gradient_fixture(), tile_local_shadow_gradient_fixture()]


def tile_local_nearest_inner_fixture() -> DiagnosticFixture:
    h, w = 720, 980
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = (30, 80, 235)
    alpha = _disc_alpha(h, w, 205, feather=18.0)
    foreground = (232, 178, 92)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="tile_local_nearest_inner",
        rgb=_composite_rgb_linear(background, np.broadcast_to(np.asarray(foreground, dtype=np.uint8), background.shape), alpha),
        settings=KeySettings(
            key_color=(30, 80, 235),
            auto_border_sample=False,
            local_screen_model=False,
            edge_refine_radius=8,
            clip_background=0.78,
            clip_foreground=0.14,
            fringe_band_radius=4,
            use_tiling=True,
            tile_size=173,
            tile_overlap=5,
            despill=0.35,
            decontaminate=1.0,
            unmix_amount=0.35,
            fringe_remove=0.45,
            edge_color_repair=1.0,
            inner_color_pull=1.0,
            luminance_restore=0.0,
            luminance_protect=0.0,
        ),
        notes="Phase 6 fixture: cap-forced tile-local nearest-inner repair on a large soft blue edge.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
        diagnostic_only=False,
    )


def large_synthetic_fixture() -> DiagnosticFixture:
    # Runtime-only synthetic case; do not commit generated images.
    h, w = 1536, 2048
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (0, 218, 45)
    alpha = _disc_alpha(h, w, 430, feather=5.0)
    rgb = _composite_rgb(rgb, (220, 164, 128), alpha)
    return DiagnosticFixture(
        name="green_large_runtime",
        rgb=rgb,
        settings=KeySettings(key_color=(0, 220, 50)),
        notes="Runtime-only large-image diagnostic; generated under .artifact when requested.",
    )


def diagnostic_fixtures(include_large: bool = False) -> list[DiagnosticFixture]:
    fixtures = [
        green_flat_fixture(),
        blue_flat_fixture(),
        custom_flat_fixture(),
        uneven_gradient_fixture(),
        same_color_island_fixture(),
        antialiased_edge_fixture(),
        blue_gradient_screen_fixture(),
        green_gradient_screen_fixture(),
        same_key_foreground_core_fixture(),
        hair_lines_fixture(),
        semi_transparent_glass_fixture(),
        white_gray_black_composite_fixture(),
    ]
    if include_large:
        fixtures.append(large_synthetic_fixture())
        fixtures.append(large_tile_gradient_runtime_fixture())
    return fixtures


def algorithm_upgrade_fixtures(include_large: bool = True) -> list[DiagnosticFixture]:
    fixtures = [
        blue_gradient_screen_fixture(),
        green_gradient_screen_fixture(),
        same_key_foreground_core_fixture(),
        hair_lines_fixture(),
        semi_transparent_glass_fixture(),
        white_gray_black_composite_fixture(),
    ]
    if include_large:
        fixtures.append(large_tile_gradient_runtime_fixture())
    return fixtures


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, mode="RGBA").save(path)


def write_diagnostic_outputs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing non-enforcing diagnostics to {ARTIFACT_DIR}")
    for fixture in diagnostic_fixtures(include_large=True):
        rgba = process_chroma_key(fixture.rgb, fixture.settings)
        _save_rgb(ARTIFACT_DIR / f"{fixture.name}_source.png", fixture.rgb)
        _save_rgba(ARTIFACT_DIR / f"{fixture.name}_result.png", rgba)
        Image.fromarray(rgba[:, :, 3]).save(ARTIFACT_DIR / f"{fixture.name}_alpha.png")
        mean_alpha = float(np.mean(rgba[:, :, 3]))
        print(f"diagnostic {fixture.name}: mean alpha={mean_alpha:.2f}; {fixture.notes}")


def _solid_composite(rgba: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)


def edge_key_residual(rgba: np.ndarray, key_color: tuple[int, int, int], edge_mask: np.ndarray) -> dict[str, int | float]:
    mask = edge_mask.astype(bool, copy=False) & (rgba[:, :, 3] > 0)
    if not np.any(mask):
        return {"count": 0, "mean_positive_excess": 0.0, "p95_positive_excess": 0.0, "max_positive_excess": 0}
    key_channel = int(np.argmax(np.asarray(key_color)))
    other = [c for c in range(3) if c != key_channel]
    rgb = rgba[:, :, :3].astype(np.int16)
    excess = rgb[:, :, key_channel] - np.maximum(rgb[:, :, other[0]], rgb[:, :, other[1]])
    positive = np.maximum(excess[mask], 0)
    return {
        "count": int(positive.size),
        "mean_positive_excess": float(np.mean(positive)),
        "p95_positive_excess": float(np.percentile(positive, 95)),
        "max_positive_excess": int(np.max(positive)),
    }


def rgb_key_residual(rgb: np.ndarray, key_color: tuple[int, int, int], edge_mask: np.ndarray) -> dict[str, int | float]:
    mask = edge_mask.astype(bool, copy=False)
    if not np.any(mask):
        return {"count": 0, "mean_positive_excess": 0.0, "p95_positive_excess": 0.0, "max_positive_excess": 0}
    key_channel = int(np.argmax(np.asarray(key_color)))
    other = [c for c in range(3) if c != key_channel]
    rgb_i = rgb.astype(np.int16)
    excess = rgb_i[:, :, key_channel] - np.maximum(rgb_i[:, :, other[0]], rgb_i[:, :, other[1]])
    positive = np.maximum(excess[mask], 0)
    return {
        "count": int(positive.size),
        "mean_positive_excess": float(np.mean(positive)),
        "p95_positive_excess": float(np.percentile(positive, 95)),
        "max_positive_excess": int(np.max(positive)),
    }


def opaque_foreground_max_delta(
    source_rgb: np.ndarray,
    rgba: np.ndarray,
    foreground_core_mask: np.ndarray,
    expected_foreground_rgb: np.ndarray | tuple[int, int, int] | None = None,
    min_alpha: int = 240,
) -> dict[str, int | float]:
    mask = foreground_core_mask.astype(bool, copy=False) & (rgba[:, :, 3] >= min_alpha)
    if not np.any(mask):
        return {"count": 0, "max_delta": 0, "mean_delta": 0.0}
    if expected_foreground_rgb is None:
        expected = source_rgb
    elif isinstance(expected_foreground_rgb, tuple):
        expected = np.broadcast_to(np.asarray(expected_foreground_rgb, dtype=np.uint8).reshape(1, 1, 3), source_rgb.shape)
    else:
        expected = expected_foreground_rgb
    delta = np.abs(rgba[mask, :3].astype(np.int16) - expected[mask].astype(np.int16))
    return {"count": int(delta.shape[0]), "max_delta": int(np.max(delta)), "mean_delta": float(np.mean(delta))}


def _mask_image(mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError("shape is required for an empty diagnostic mask")
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if arr.size and float(np.nanmax(arr)) <= 1.0 else 1.0
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0) * scale
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_mask(path: Path, mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> None:
    Image.fromarray(_mask_image(mask, shape), mode="L").save(path)


def alpha_soft_band_count(alpha: np.ndarray, low: int = 0, high: int = 255) -> int:
    return int(np.count_nonzero((alpha > low) & (alpha < high)))


def alpha_edge_roughness(alpha: np.ndarray, mask: np.ndarray) -> float:
    alpha_f = alpha.astype(np.float32)
    mask_b = mask.astype(bool, copy=False)
    dx_mask = mask_b[:, 1:] & mask_b[:, :-1]
    dy_mask = mask_b[1:, :] & mask_b[:-1, :]
    values: list[np.ndarray] = []
    if np.any(dx_mask):
        values.append(np.abs(alpha_f[:, 1:] - alpha_f[:, :-1])[dx_mask])
    if np.any(dy_mask):
        values.append(np.abs(alpha_f[1:, :] - alpha_f[:-1, :])[dy_mask])
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def transparent_rgb_zero(rgba: np.ndarray) -> dict[str, int | bool]:
    transparent = rgba[:, :, 3] == 0
    if not np.any(transparent):
        return {"ok": True, "transparent_pixel_count": 0, "max_rgb_when_transparent": 0}
    max_rgb = int(rgba[transparent, :3].max())
    return {
        "ok": max_rgb == 0,
        "transparent_pixel_count": int(np.count_nonzero(transparent)),
        "max_rgb_when_transparent": max_rgb,
    }


def tiled_vs_full_max_diff(
    rgb: np.ndarray,
    settings: KeySettings,
    tile_size: int,
    tile_overlap: int,
) -> dict[str, int]:
    tiled_settings = replace(settings, use_tiling=True, tile_size=tile_size, tile_overlap=tile_overlap)
    full_settings = replace(tiled_settings, use_tiling=False)
    tiled = process_key_image(rgb, tiled_settings)
    full = process_key_image(rgb, full_settings)
    diff = np.abs(tiled.rgba.astype(np.int16) - full.rgba.astype(np.int16))
    return {
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "max_rgba_diff": int(diff.max()),
        "max_alpha_diff": int(diff[:, :, 3].max()),
    }


def composite_black_white_gray_error(
    rgba: np.ndarray,
    expected_rgba: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, int | float]]:
    if mask is None:
        mask = np.ones(rgba.shape[:2], dtype=bool)
    else:
        mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return {
            name: {"count": 0, "max_abs_error": 0, "mean_abs_error": 0.0}
            for name in ("black", "white", "gray", "checker")
        }
    metrics: dict[str, dict[str, int | float]] = {}
    for name, color in (("black", (0, 0, 0)), ("white", (255, 255, 255)), ("gray", (128, 128, 128))):
        actual = _solid_composite(rgba, color)
        expected = _solid_composite(expected_rgba, color)
        delta = np.abs(actual[mask].astype(np.int16) - expected[mask].astype(np.int16))
        metrics[name] = {
            "count": int(delta.shape[0]),
            "max_abs_error": int(delta.max()),
            "mean_abs_error": float(np.mean(delta)),
        }
    actual_checker = checkerboard_composite(rgba)
    expected_checker = checkerboard_composite(expected_rgba)
    checker_delta = np.abs(actual_checker[mask].astype(np.int16) - expected_checker[mask].astype(np.int16))
    metrics["checker"] = {
        "count": int(checker_delta.shape[0]),
        "max_abs_error": int(checker_delta.max()),
        "mean_abs_error": float(np.mean(checker_delta)),
    }
    return metrics


def background_alpha_leak(alpha_u8: np.ndarray, known_background: np.ndarray) -> dict[str, int | float]:
    mask = known_background.astype(bool, copy=False)
    if not np.any(mask):
        return {"count": 0, "max_alpha": 0, "mean_alpha": 0.0, "leaking_pixels": 0}
    values = alpha_u8[mask]
    return {
        "count": int(values.size),
        "max_alpha": int(values.max()),
        "mean_alpha": float(np.mean(values)),
        "leaking_pixels": int(np.count_nonzero(values > 0)),
    }


def alpha_detail_recall(expected_alpha: np.ndarray | None, actual_alpha_u8: np.ndarray) -> dict[str, int | float]:
    if expected_alpha is None:
        return {"count": 0, "visible_recall": 1.0, "mean_alpha_ratio": 1.0}
    expected_u8 = np.rint(np.clip(expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    detail = expected_u8 > 0
    if not np.any(detail):
        return {"count": 0, "visible_recall": 1.0, "mean_alpha_ratio": 1.0}
    actual = actual_alpha_u8[detail].astype(np.float32)
    expected = np.maximum(expected_u8[detail].astype(np.float32), 1.0)
    return {
        "count": int(actual.size),
        "visible_recall": float(np.mean(actual > 0)),
        "mean_alpha_ratio": float(np.mean(np.minimum(actual / expected, 1.0))),
    }


def foreground_core_rgb_delta(fixture: DiagnosticFixture, result: KeyResult) -> dict[str, int | float]:
    if fixture.expected_alpha is not None:
        mask = (fixture.expected_alpha >= 0.999) & (result.alpha >= 250)
    else:
        _, foreground_core, _ = _fixture_masks(fixture)
        mask = foreground_core & (result.alpha >= 250)
    if not np.any(mask):
        return {"count": 0, "max_delta": 0, "mean_delta": 0.0}
    expected = _expected_foreground_array(fixture)
    if expected is None:
        expected = fixture.rgb
    delta = np.abs(result.rgba[mask, :3].astype(np.int16) - expected[mask].astype(np.int16))
    return {"count": int(delta.shape[0]), "max_delta": int(np.max(delta)), "mean_delta": float(np.mean(delta))}


def _baseline_metrics_for_fixture(fixture: DiagnosticFixture, result: KeyResult) -> dict[str, Any]:
    known_background, foreground_core, soft_edge = _fixture_masks(fixture)
    expected_rgba = _expected_rgba_for_fixture(fixture)
    edge_mask = soft_edge
    if not np.any(edge_mask) and result.fringe_mask is not None:
        edge_mask = result.fringe_mask > 0

    metrics: dict[str, Any] = {
        "diagnostic_only": bool(fixture.diagnostic_only),
        "shape": list(fixture.rgb.shape),
        "mask_counts": {
            "known_background": int(np.count_nonzero(known_background)),
            "foreground_core": int(np.count_nonzero(foreground_core)),
            "soft_edge": int(np.count_nonzero(soft_edge)),
        },
        "edge_key_residual": edge_key_residual(result.rgba, fixture.settings.key_color, edge_mask),
        "opaque_foreground_max_delta": opaque_foreground_max_delta(
            fixture.rgb,
            result.rgba,
            foreground_core,
            _expected_foreground_array(fixture),
        ),
        "alpha_soft_band_count": alpha_soft_band_count(result.alpha),
        "transparent_rgb_zero": transparent_rgb_zero(result.rgba),
        "tiled_vs_full_max_diff": None,
        "composite_black_white_gray_error": None,
    }

    if expected_rgba is not None:
        composite_mask = ~known_background | soft_edge
        metrics["composite_black_white_gray_error"] = composite_black_white_gray_error(
            result.rgba,
            expected_rgba,
            composite_mask,
        )

    if fixture.name == "large_tile_gradient_runtime":
        metrics["tiled_vs_full_max_diff"] = {
            "tile_257": tiled_vs_full_max_diff(fixture.rgb, fixture.settings, tile_size=257, tile_overlap=48),
            "tile_384": tiled_vs_full_max_diff(fixture.rgb, fixture.settings, tile_size=384, tile_overlap=56),
        }
    return metrics


def _transition_unmix_baseline_metrics_for_fixture(fixture: DiagnosticFixture, result: KeyResult) -> dict[str, Any]:
    known_background, foreground_core, soft_edge = _fixture_masks(fixture)
    expected_rgba = _expected_rgba_for_fixture(fixture)
    transition_mask = soft_edge
    if not np.any(transition_mask) and result.fringe_mask is not None:
        transition_mask = result.fringe_mask > 0
    if not np.any(transition_mask) and fixture.expected_alpha is not None:
        transition_mask = (fixture.expected_alpha > 0.0) & ~foreground_core

    transition_key = edge_key_residual(result.rgba, fixture.settings.key_color, transition_mask)
    detail_recall = alpha_detail_recall(fixture.expected_alpha, result.alpha)
    core_delta = foreground_core_rgb_delta(fixture, result)
    transparent = transparent_rgb_zero(result.rgba)
    metrics = _baseline_metrics_for_fixture(fixture, result)
    metrics.update(
        {
            "hard_edge_core_rgb_delta": opaque_foreground_max_delta(
                fixture.rgb,
                result.rgba,
                foreground_core,
                _expected_foreground_array(fixture),
            ),
            "foreground_core_rgb_delta": core_delta,
            "transition_key_residual": transition_key,
            "key_residual_on_transition": transition_key,
            "alpha_detail_recall": detail_recall,
            "background_alpha_leak": background_alpha_leak(result.alpha, known_background),
            "transparent_rgb_residual_max": int(transparent["max_rgb_when_transparent"]),
            "composite_residuals": None,
        }
    )
    if expected_rgba is not None:
        composite_mask = foreground_core | soft_edge
        if not np.any(composite_mask):
            composite_mask = ~known_background
        metrics["composite_residuals"] = composite_black_white_gray_error(result.rgba, expected_rgba, composite_mask)
    return metrics


def _write_composite_baseline_previews(fixture: DiagnosticFixture, result: KeyResult) -> None:
    expected_rgba = _expected_rgba_for_fixture(fixture)
    if expected_rgba is None:
        return
    _save_rgb(ALGORITHM_BASELINE_DIR / f"{fixture.name}_source.png", fixture.rgb)
    _save_rgba(ALGORITHM_BASELINE_DIR / f"{fixture.name}_result.png", result.rgba)
    Image.fromarray(result.alpha, mode="L").save(ALGORITHM_BASELINE_DIR / f"{fixture.name}_alpha.png")
    backgrounds: dict[str, tuple[int, int, int] | None] = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "checkerboard": None,
    }
    for name, color in backgrounds.items():
        actual = checkerboard_composite(result.rgba) if color is None else _solid_composite(result.rgba, color)
        expected = checkerboard_composite(expected_rgba) if color is None else _solid_composite(expected_rgba, color)
        _save_rgb(ALGORITHM_BASELINE_DIR / f"{fixture.name}_actual_on_{name}.png", actual)
        _save_rgb(ALGORITHM_BASELINE_DIR / f"{fixture.name}_expected_on_{name}.png", expected)
        _save_rgb(ALGORITHM_BASELINE_DIR / f"{fixture.name}_compare_on_{name}.png", np.concatenate([expected, actual], axis=1))


def write_algorithm_upgrade_baseline() -> None:
    ALGORITHM_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = algorithm_upgrade_fixtures(include_large=True) + transition_unmix_baseline_fixtures()
    all_metrics: dict[str, Any] = {
        "schema_version": 2,
        "generated_by": "python smoke_test.py --write-algorithm-baseline",
        "baseline_note": "Classical diagnostic baseline; v7 transition-unmix fixtures are non-strict until later phases.",
        "fixtures": {},
    }
    summary_lines = [
        "# ImgKey classical algorithm baseline",
        "",
        "Generated by `python smoke_test.py --write-algorithm-baseline`.",
        "Transition-unmix fixtures are diagnostic-only in Phase 1; later phases compare against these hashes/metrics.",
        "",
        "| Fixture | Edge residual max | Core max delta | Soft band px | Transparent RGB zero | Tile/full max |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]

    for fixture in fixtures:
        print(f"baseline fixture {fixture.name}: {fixture.notes}")
        result = process_key_image(fixture.rgb, fixture.settings, fixture.original_alpha)
        known_background, foreground_core, soft_edge = _fixture_masks(fixture)
        expected_alpha_u8 = (
            np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
            if fixture.expected_alpha is not None
            else np.zeros(fixture.rgb.shape[:2], dtype=np.uint8)
        )
        if fixture.name.startswith("transition_"):
            metrics = _transition_unmix_baseline_metrics_for_fixture(fixture, result)
        else:
            metrics = _baseline_metrics_for_fixture(fixture, result)
        hashes = {
            "source_rgb_sha256": _array_sha256(fixture.rgb),
            "rgba_sha256": _array_sha256(result.rgba),
            "alpha_sha256": _array_sha256(result.alpha),
            "known_background_mask_sha256": _array_sha256(known_background.astype(np.uint8)),
            "foreground_core_mask_sha256": _array_sha256(foreground_core.astype(np.uint8)),
            "soft_edge_mask_sha256": _array_sha256(soft_edge.astype(np.uint8)),
        }
        fixture_record = {
            "notes": fixture.notes,
            "settings": asdict(fixture.settings),
            "metrics": metrics,
            "hashes": hashes,
            "artifact": f"{fixture.name}.npz",
        }
        all_metrics["fixtures"][fixture.name] = fixture_record

        np.savez_compressed(
            ALGORITHM_BASELINE_DIR / f"{fixture.name}.npz",
            source_rgb=fixture.rgb,
            rgba=result.rgba,
            alpha=result.alpha,
            known_background_mask=known_background.astype(np.uint8),
            foreground_core_mask=foreground_core.astype(np.uint8),
            soft_edge_mask=soft_edge.astype(np.uint8),
            expected_alpha=expected_alpha_u8,
            metrics_json=np.asarray(json.dumps(_json_ready(metrics), sort_keys=True)),
            hashes_json=np.asarray(json.dumps(hashes, sort_keys=True)),
        )

        if fixture.name == "white_gray_black_composite":
            _write_composite_baseline_previews(fixture, result)

        tile_metrics = metrics.get("tiled_vs_full_max_diff") or {}
        tile_max = 0
        if isinstance(tile_metrics, dict):
            tile_max = max((int(v.get("max_rgba_diff", 0)) for v in tile_metrics.values() if isinstance(v, dict)), default=0)
        transparent = metrics["transparent_rgb_zero"]
        summary_lines.append(
            f"| {fixture.name} | {metrics['edge_key_residual']['max_positive_excess']} | "
            f"{metrics['opaque_foreground_max_delta']['max_delta']} | {metrics['alpha_soft_band_count']} | "
            f"{transparent['ok']} (max {transparent['max_rgb_when_transparent']}) | {tile_max} |"
        )

    metrics_path = ALGORITHM_BASELINE_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(all_metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = ALGORITHM_BASELINE_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote algorithm baseline metrics to {metrics_path}")
    print(f"wrote algorithm baseline summary to {summary_path}")


def ensure_algorithm_upgrade_baseline() -> None:
    metrics_path = ALGORITHM_BASELINE_DIR / "metrics.json"
    if metrics_path.exists():
        return
    print(f"missing algorithm baseline at {metrics_path}; generating it for self-contained smoke checks")
    write_algorithm_upgrade_baseline()


def _transition_detail_region(fixture: DiagnosticFixture, result: KeyResult) -> np.ndarray:
    _, foreground_core, soft_edge = _fixture_masks(fixture)
    detail_region = soft_edge.copy()
    if fixture.expected_alpha is not None:
        detail_region |= (fixture.expected_alpha > 0.0) & ~foreground_core
    if not np.any(detail_region):
        detail_region = result.alpha > 0
    return detail_region


def _transition_unmix_comparison_metrics(
    fixture: DiagnosticFixture,
    baseline: KeyResult,
    result: KeyResult,
) -> dict[str, Any]:
    before = _transition_unmix_baseline_metrics_for_fixture(fixture, baseline)
    after = _transition_unmix_baseline_metrics_for_fixture(fixture, result)
    alpha_delta = result.alpha.astype(np.int16) - baseline.alpha.astype(np.int16)
    detail_region = _transition_detail_region(fixture, result)
    return {
        "fixture": fixture.name,
        "key_residual_on_transition_before": before["key_residual_on_transition"],
        "key_residual_on_transition_after": after["key_residual_on_transition"],
        "alpha_detail_recall_before": before["alpha_detail_recall"],
        "alpha_detail_recall_after": after["alpha_detail_recall"],
        "foreground_core_rgb_delta": after["foreground_core_rgb_delta"],
        "transparent_rgb_residual_max": after["transparent_rgb_residual_max"],
        "composite_residuals_before": before["composite_residuals"],
        "composite_residuals_after": after["composite_residuals"],
        "alpha_recovered_pixel_count": int(np.count_nonzero(alpha_delta[detail_region] > 0)),
        "alpha_delta_min_on_detail": int(alpha_delta[detail_region].min()) if np.any(detail_region) else 0,
        "alpha_delta_max_on_detail": int(alpha_delta[detail_region].max()) if np.any(detail_region) else 0,
    }


def _foreground_reference_validity_for_diagnostics(fixture: DiagnosticFixture, result: KeyResult) -> np.ndarray | None:
    if result.background_mask is None or result.screen_probability is None or result.fringe_mask is None:
        return None
    labels, label_to_flat, distance = _build_nearest_inner_reference_map(
        result.alpha,
        result.background_mask > 0,
        result.screen_probability,
        result.fringe_mask,
        fixture.settings,
    )
    if labels is None or label_to_flat is None or distance is None:
        return None
    radius = int(np.clip(int(fixture.settings.foreground_reference_radius), 0, np.iinfo(np.uint16).max - 1))
    return (labels > 0) & (distance <= radius)


def write_transition_unmix_diagnostics() -> None:
    TRANSITION_UNMIX_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing transition-unmix diagnostics to {TRANSITION_UNMIX_DIAGNOSTIC_DIR}")
    all_metrics: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "python smoke_test.py --write-transition-unmix-diagnostics",
        "fixtures": {},
    }
    for fixture in transition_unmix_diagnostic_fixtures():
        baseline = _process_fixture_result(fixture, replace(fixture.settings, transition_unmix=False))
        result = _process_fixture_result(fixture)
        metrics = _transition_unmix_comparison_metrics(fixture, baseline, result)
        all_metrics["fixtures"][fixture.name] = metrics

        expected_rgba = _expected_rgba_for_fixture(fixture)
        transition_mask = _transition_detail_region(fixture, result)
        reference_valid = _foreground_reference_validity_for_diagnostics(fixture, result)
        alpha_delta = np.maximum(result.alpha.astype(np.int16) - baseline.alpha.astype(np.int16), 0).astype(np.uint8)

        _save_rgb(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_source.png", fixture.rgb)
        _save_rgb(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_baseline_rgb.png", baseline.rgba[:, :, :3])
        _save_mask(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_baseline_alpha.png", baseline.alpha)
        _save_mask(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_transition_mask.png", transition_mask)
        _save_mask(
            TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_foreground_reference_valid.png",
            reference_valid,
            fixture.rgb.shape[:2],
        )
        _save_mask(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_alpha_recovered.png", alpha_delta)
        _save_rgb(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_repaired_rgb.png", result.rgba[:, :, :3])
        _save_rgba(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_result.png", result.rgba)
        if fixture.keep_mask is not None:
            _save_mask(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_keep_mask.png", fixture.keep_mask)
        if fixture.remove_mask is not None:
            _save_mask(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_remove_mask.png", fixture.remove_mask)
        if fixture.original_alpha is not None:
            _save_mask(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_source_alpha.png", fixture.original_alpha)

        for background_name, color in (
            ("black", (0, 0, 0)),
            ("white", (255, 255, 255)),
            ("gray", (128, 128, 128)),
            ("checker", None),
        ):
            actual = checkerboard_composite(result.rgba) if color is None else _solid_composite(result.rgba, color)
            _save_rgb(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_composite_{background_name}.png", actual)
            if expected_rgba is not None:
                expected = checkerboard_composite(expected_rgba) if color is None else _solid_composite(expected_rgba, color)
                _save_rgb(TRANSITION_UNMIX_DIAGNOSTIC_DIR / f"{fixture.name}_expected_composite_{background_name}.png", expected)

        print(
            f"transition diagnostic {fixture.name}: "
            f"residual_mean={metrics['key_residual_on_transition_before']['mean_positive_excess']:.2f}"
            f"->{metrics['key_residual_on_transition_after']['mean_positive_excess']:.2f}; "
            f"recovered={metrics['alpha_recovered_pixel_count']}"
        )

    metrics_path = TRANSITION_UNMIX_DIAGNOSTIC_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(all_metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote transition-unmix metrics to {metrics_path}")


def _load_algorithm_baseline_metrics() -> dict[str, Any]:
    metrics_path = ALGORITHM_BASELINE_DIR / "metrics.json"
    if not metrics_path.exists():
        raise AssertionError(
            f"missing Phase 1 algorithm baseline metrics at {metrics_path}; "
            "run `python smoke_test.py --write-algorithm-baseline` before Phase 2+ checks"
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _load_algorithm_baseline_artifact(fixture_name: str) -> dict[str, np.ndarray]:
    artifact_path = ALGORITHM_BASELINE_DIR / f"{fixture_name}.npz"
    if not artifact_path.exists():
        raise AssertionError(
            f"missing Phase 1 algorithm baseline artifact at {artifact_path}; "
            "run `python smoke_test.py --write-algorithm-baseline` before Phase 2+ checks"
        )
    with np.load(artifact_path) as artifact:
        return {
            "source_rgb": artifact["source_rgb"].copy(),
            "rgba": artifact["rgba"].copy(),
            "alpha": artifact["alpha"].copy(),
            "known_background_mask": artifact["known_background_mask"].copy(),
            "foreground_core_mask": artifact["foreground_core_mask"].copy(),
            "soft_edge_mask": artifact["soft_edge_mask"].copy(),
        }


def _assert_phase2_linear_helper_round_trip() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1, 1)
    ramp_rgb = np.repeat(ramp, 3, axis=2)
    round_trip = _linear_f32_to_srgb_u8(_srgb_u8_to_linear_f32(ramp_rgb))
    max_delta = int(np.abs(round_trip.astype(np.int16) - ramp_rgb.astype(np.int16)).max())
    assert max_delta <= 1, f"sRGB/linear helpers should round-trip uint8 ramp within 1 level, max={max_delta}"


def run_phase2_linear_color_tests() -> None:
    _assert_phase2_linear_helper_round_trip()
    baseline = _load_algorithm_baseline_metrics()
    fixture_records = baseline.get("fixtures", {})
    summaries: list[str] = []

    for fixture in algorithm_upgrade_fixtures(include_large=True):
        if fixture.name not in fixture_records:
            raise AssertionError(f"{fixture.name}: missing record in Phase 1 baseline metrics")
        record = fixture_records[fixture.name]
        artifact = _load_algorithm_baseline_artifact(fixture.name)
        source_hash = record.get("hashes", {}).get("source_rgb_sha256")
        assert source_hash == _array_sha256(fixture.rgb), f"{fixture.name}: source fixture changed since Phase 1 baseline"

        result = process_key_image(fixture.rgb, fixture.settings)
        alpha_delta = result.alpha.astype(np.int16) - artifact["alpha"].astype(np.int16)
        alpha_diff = int(np.abs(alpha_delta).max())
        alpha_min_delta = int(alpha_delta.min())
        assert alpha_min_delta >= 0, (
            f"{fixture.name}: transition alpha recovery must not erode alpha vs v4 baseline, min_delta={alpha_min_delta}"
        )

        transparent = transparent_rgb_zero(result.rgba)
        assert transparent["ok"], (
            f"{fixture.name}: transparent RGB must stay zero, max={transparent['max_rgb_when_transparent']}"
        )

        soft_edge = artifact["soft_edge_mask"].astype(bool)
        edge_mask = soft_edge if np.any(soft_edge) else (result.fringe_mask > 0 if result.fringe_mask is not None else soft_edge)
        baseline_visible_edge = edge_mask & (artifact["alpha"] > 0)
        current_residual = edge_key_residual(result.rgba, fixture.settings.key_color, baseline_visible_edge)
        baseline_residual = record["metrics"]["edge_key_residual"]
        assert current_residual["max_positive_excess"] <= baseline_residual["max_positive_excess"], (
            f"{fixture.name}: fringe key-channel max excess regressed vs v4 baseline, "
            f"{current_residual['max_positive_excess']} > {baseline_residual['max_positive_excess']}"
        )
        assert current_residual["p95_positive_excess"] <= baseline_residual["p95_positive_excess"], (
            f"{fixture.name}: fringe key-channel p95 excess regressed vs v4 baseline, "
            f"{current_residual['p95_positive_excess']} > {baseline_residual['p95_positive_excess']}"
        )

        unchanged_mask = (result.alpha > 0) & (result.despill_mask == 0)
        unchanged_delta = (
            int(np.abs(result.rgba[unchanged_mask, :3].astype(np.int16) - fixture.rgb[unchanged_mask].astype(np.int16)).max())
            if np.any(unchanged_mask)
            else 0
        )
        assert unchanged_delta == 0, f"{fixture.name}: live pixels outside color-repair masks drifted by {unchanged_delta}"

        core_mask = artifact["foreground_core_mask"].astype(bool) & (result.alpha >= 250)
        if np.any(core_mask):
            core_delta = int(
                np.abs(result.rgba[core_mask, :3].astype(np.int16) - artifact["rgba"][core_mask, :3].astype(np.int16)).max()
            )
        else:
            core_delta = 0
        assert core_delta <= 5, f"{fixture.name}: opaque core RGB drift vs v4 baseline should be <=5, got {core_delta}"

        if fixture.name == "large_tile_gradient_runtime":
            for tile_size, tile_overlap in ((257, 48), (384, 56)):
                tile_diff = tiled_vs_full_max_diff(fixture.rgb, fixture.settings, tile_size=tile_size, tile_overlap=tile_overlap)
                assert tile_diff["max_alpha_diff"] == 0, (
                    f"{fixture.name}: tiled/reference alpha diff must be 0 for tile {tile_size}, got {tile_diff['max_alpha_diff']}"
                )
                assert tile_diff["max_rgba_diff"] <= 1, (
                    f"{fixture.name}: tiled/reference max RGBA diff must be <=1 for tile {tile_size}, got {tile_diff['max_rgba_diff']}"
                )

        summaries.append(
            f"{fixture.name}: alpha_diff={alpha_diff} alpha_min_delta={alpha_min_delta} "
            f"fringe_max={current_residual['max_positive_excess']}"
            f"<=v4:{baseline_residual['max_positive_excess']} fringe_p95={current_residual['p95_positive_excess']}"
            f"<=v4:{baseline_residual['p95_positive_excess']} core_drift={core_delta} unchanged_drift={unchanged_delta}"
        )

    print("Phase 2 transition alpha/color checks vs v4 baseline:")
    for line in summaries:
        print(f"  {line}")


def _assert_guided_filter_helper() -> None:
    guide = np.tile(np.linspace(0.0, 1.0, 33, dtype=np.float32), (21, 1))
    src = guide.copy()
    filtered = _guided_filter_gray(guide, src, radius=3, eps=1e-3)
    assert filtered.shape == src.shape, "guided filter should preserve 2D shape"
    assert filtered.dtype == np.float32, "guided filter should return float32"
    assert float(filtered.min()) >= 0.0 and float(filtered.max()) <= 1.0, "guided filter output should stay clamped"
    max_delta = float(np.max(np.abs(filtered - src)))
    assert max_delta <= 0.035, f"guided filter should preserve a self-guided ramp closely, max_delta={max_delta:.4f}"


def run_phase3_guided_alpha_tests() -> None:
    _assert_guided_filter_helper()
    fixture = guided_alpha_refine_fixture()
    known_background, foreground_core, soft_edge = _fixture_masks(fixture)

    default_off = process_key_image(fixture.rgb, fixture.settings)
    explicit_off = process_key_image(
        fixture.rgb,
        replace(fixture.settings, guided_alpha_refine=0.0, guided_radius=5, guided_eps=1e-3),
    )
    assert np.array_equal(default_off.alpha, explicit_off.alpha), "guided default/off alpha output must be exact"
    assert np.array_equal(default_off.rgba, explicit_off.rgba), "guided default/off RGBA output must be exact"

    skipped = process_key_image(
        fixture.rgb,
        replace(fixture.settings, guided_alpha_refine=1.0, guided_radius=5, guided_max_pixels=1),
    )
    assert np.array_equal(default_off.alpha, skipped.alpha), "guided cap fallback should skip deterministically unchanged"
    assert np.array_equal(default_off.rgba, skipped.rgba), "guided cap fallback should leave RGBA unchanged"

    guided_settings = replace(
        fixture.settings,
        guided_alpha_refine=0.85,
        guided_radius=5,
        guided_eps=1e-3,
        guided_max_pixels=500_000,
    )
    guided = process_key_image(fixture.rgb, guided_settings)
    assert int(guided.alpha[known_background].max()) == 0, "guided refinement must keep known background alpha 0"
    assert int(guided.alpha[foreground_core].min()) == 255, "guided refinement must keep known foreground/core alpha 255"

    off_roughness = alpha_edge_roughness(default_off.alpha, soft_edge)
    guided_roughness = alpha_edge_roughness(guided.alpha, soft_edge)
    off_soft = alpha_soft_band_count(default_off.alpha)
    guided_soft = alpha_soft_band_count(guided.alpha)
    edge_changed = int(np.count_nonzero(default_off.alpha[soft_edge] != guided.alpha[soft_edge]))
    assert edge_changed > 0, "guided refinement should change the soft edge on the guided fixture"
    assert guided_roughness <= off_roughness * 0.92, (
        f"guided refinement should reduce edge alpha roughness, {off_roughness:.3f} -> {guided_roughness:.3f}"
    )
    assert guided_soft >= int(off_soft * 0.98), (
        f"guided refinement should preserve/increase soft alpha coverage, off={off_soft} guided={guided_soft}"
    )

    tiled = process_key_image(fixture.rgb, replace(guided_settings, use_tiling=True, tile_size=101, tile_overlap=20))
    full = process_key_image(fixture.rgb, replace(guided_settings, use_tiling=False))
    diff = np.abs(tiled.rgba.astype(np.int16) - full.rgba.astype(np.int16))
    max_alpha_diff = int(diff[:, :, 3].max())
    max_rgba_diff = int(diff.max())
    assert max_alpha_diff <= 1, f"guided tiled/full alpha diff should be <=1, got {max_alpha_diff}"
    assert max_rgba_diff <= 1, f"guided tiled/full RGBA diff should be <=1, got {max_rgba_diff}"

    forbidden = HEAVY_OPTIONAL_MODULES
    imported = forbidden & set(sys.modules)
    assert not imported, f"guided alpha refinement must not import heavy optional modules: {sorted(imported)}"

    print(
        "Phase 3 guided alpha checks: "
        f"roughness {off_roughness:.3f}->{guided_roughness:.3f}; "
        f"soft_px {off_soft}->{guided_soft}; edge_changed={edge_changed}; "
        f"tile_alpha_diff={max_alpha_diff}; tile_rgba_diff={max_rgba_diff}"
    )


def _tile_boundary_band_mask(shape: tuple[int, int], tile_size: int, band_radius: int = 2) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    radius = max(1, int(band_radius))
    for x in range(int(tile_size), w, int(tile_size)):
        mask[:, max(0, x - radius) : min(w, x + radius + 1)] = True
    for y in range(int(tile_size), h, int(tile_size)):
        mask[max(0, y - radius) : min(h, y + radius + 1), :] = True
    return mask


def tile_boundary_band_metrics(
    rgb: np.ndarray,
    settings: KeySettings,
    tile_sizes: tuple[int, int],
) -> dict[str, int]:
    first = process_key_image(
        rgb,
        replace(settings, use_tiling=True, tile_size=tile_sizes[0], local_screen_model=True, max_local_screen_model_pixels=1),
    )
    second = process_key_image(
        rgb,
        replace(settings, use_tiling=True, tile_size=tile_sizes[1], local_screen_model=True, max_local_screen_model_pixels=1),
    )
    band = _tile_boundary_band_mask(rgb.shape[:2], tile_sizes[0]) | _tile_boundary_band_mask(rgb.shape[:2], tile_sizes[1])
    if not np.any(band):
        raise AssertionError("tile boundary band mask should not be empty")

    diff = np.abs(first.rgba.astype(np.int16) - second.rgba.astype(np.int16))
    fringe_first = first.fringe_mask > 4 if first.fringe_mask is not None else np.zeros(rgb.shape[:2], dtype=bool)
    fringe_second = second.fringe_mask > 4 if second.fringe_mask is not None else np.zeros(rgb.shape[:2], dtype=bool)
    opaque_nonfringe = band & (first.alpha >= 250) & (second.alpha >= 250) & ~fringe_first & ~fringe_second
    visible = band & ((first.alpha > 0) | (second.alpha > 0))
    checker_diff = np.abs(checkerboard_composite(first.rgba).astype(np.int16) - checkerboard_composite(second.rgba).astype(np.int16))
    return {
        "tile_size_a": int(tile_sizes[0]),
        "tile_size_b": int(tile_sizes[1]),
        "boundary_pixels": int(np.count_nonzero(band)),
        "opaque_nonfringe_pixels": int(np.count_nonzero(opaque_nonfringe)),
        "visible_pixels": int(np.count_nonzero(visible)),
        "max_alpha_diff": int(diff[:, :, 3][band].max()),
        "max_rgb_diff_opaque_nonfringe": int(diff[:, :, :3][opaque_nonfringe].max()) if np.any(opaque_nonfringe) else 0,
        "max_checker_diff_visible": int(checker_diff[visible].max()) if np.any(visible) else 0,
        "max_rgba_diff_boundary": int(diff[band].max()),
    }


def run_phase4_tile_local_screen_tests() -> None:
    helper_rgb = np.zeros((9, 11, 3), dtype=np.uint8)
    helper_rgb[:, :, :] = (0, 220, 50)
    helper_rgb[:, 6:, :] = (0, 180, 42)
    helper_known = np.zeros((9, 11), dtype=bool)
    helper_known[:, :3] = True
    helper_known[:, 8:] = True
    helper_screen = _estimate_screen_tile(helper_rgb, helper_known, (0, 200, 45), radius=2)
    assert helper_screen.shape == helper_rgb.shape and helper_screen.dtype == np.uint8, "screen tile estimate must be uint8 HxWx3"
    assert int(helper_screen[4, 1, 1]) >= 215, "known bright screen side should estimate local green level"
    assert int(helper_screen[4, 9, 1]) <= 185, "known shadow screen side should estimate local green level"
    fallback_screen = _estimate_screen_tile(helper_rgb, np.zeros((9, 11), dtype=bool), (0, 200, 45), radius=2)
    assert np.array_equal(fallback_screen, np.broadcast_to(np.array([0, 200, 45], dtype=np.uint8), helper_rgb.shape)), (
        "screen tile estimate should fall back to global color when no connected-safe background exists"
    )

    summaries: list[str] = []
    for fixture in phase4_tile_local_screen_fixtures():
        _, _, soft_edge = _fixture_masks(fixture)
        global_fallback = process_key_image(
            fixture.rgb,
            replace(fixture.settings, local_screen_model=False, use_tiling=True, tile_size=173, tile_overlap=8),
        )
        tile_local = process_key_image(
            fixture.rgb,
            replace(
                fixture.settings,
                local_screen_model=True,
                max_local_screen_model_pixels=1,
                use_tiling=True,
                tile_size=173,
                tile_overlap=8,
            ),
        )
        global_residual = edge_key_residual(global_fallback.rgba, fixture.settings.key_color, soft_edge)
        local_residual = edge_key_residual(tile_local.rgba, fixture.settings.key_color, soft_edge)
        assert local_residual["max_positive_excess"] < global_residual["max_positive_excess"], (
            f"{fixture.name}: tile-local screen should lower max edge residual, "
            f"{local_residual['max_positive_excess']} >= {global_residual['max_positive_excess']}"
        )
        assert local_residual["p95_positive_excess"] < global_residual["p95_positive_excess"], (
            f"{fixture.name}: tile-local screen should lower p95 edge residual, "
            f"{local_residual['p95_positive_excess']} >= {global_residual['p95_positive_excess']}"
        )
        assert np.array_equal(global_fallback.alpha, tile_local.alpha), f"{fixture.name}: local screen model must not alter alpha"

        seam = tile_boundary_band_metrics(fixture.rgb, fixture.settings, tile_sizes=(137, 199))
        assert seam["opaque_nonfringe_pixels"] > 0, f"{fixture.name}: seam test should cover opaque non-fringe pixels"
        assert seam["max_alpha_diff"] <= 1, f"{fixture.name}: boundary alpha diff too high: {seam}"
        assert seam["max_rgb_diff_opaque_nonfringe"] <= 2, f"{fixture.name}: opaque boundary RGB diff too high: {seam}"
        assert seam["max_checker_diff_visible"] <= 2, f"{fixture.name}: visible checker seam diff too high: {seam}"

        summaries.append(
            f"{fixture.name}: max {global_residual['max_positive_excess']}->{local_residual['max_positive_excess']}; "
            f"p95 {global_residual['p95_positive_excess']}->{local_residual['p95_positive_excess']}; "
            f"seam alpha={seam['max_alpha_diff']} rgb={seam['max_rgb_diff_opaque_nonfringe']} "
            f"checker={seam['max_checker_diff_visible']}"
        )

    forbidden = HEAVY_OPTIONAL_MODULES
    imported = forbidden & set(sys.modules)
    assert not imported, f"tile-local screen model must not import heavy optional modules: {sorted(imported)}"

    print("Phase 4 tile-local screen checks:")
    for line in summaries:
        print(f"  {line}")


def _count_tile_progress(stages: list[str]) -> int:
    return sum(1 for stage in stages if stage.startswith("tile "))


def run_phase5_crop_render_tests() -> None:
    fixture = tile_local_diagonal_gradient_fixture()
    crop = (300, 170, 460, 330)
    x0, y0, x1, y1 = crop
    crop_h, crop_w = y1 - y0, x1 - x0
    settings = replace(
        fixture.settings,
        use_tiling=True,
        tile_size=137,
        tile_overlap=11,
        local_screen_model=True,
        max_local_screen_model_pixels=1,
        guided_alpha_refine=0.35,
        guided_radius=5,
        guided_max_pixels=1_000_000,
        fringe_band_radius=4,
    )
    alpha_hint = np.zeros(fixture.rgb.shape[:2], dtype=np.uint8)
    alpha_hint[205:295, 340:430] = 224

    full_stages: list[str] = []
    full = process_key_image(
        fixture.rgb,
        replace(settings, full_res_crop=None),
        alpha_hint=alpha_hint,
        progress_callback=lambda _value, stage: full_stages.append(stage),
    )

    crop_stages: list[str] = []
    cropped = process_key_image(
        fixture.rgb,
        replace(settings, full_res_crop=crop, preview_scale=1.0),
        alpha_hint=alpha_hint,
        progress_callback=lambda _value, stage: crop_stages.append(stage),
    )
    display_rgb = fixture.rgb[y0:y1, x0:x1]

    diff = np.abs(cropped.rgba.astype(np.int16) - full.rgba[y0:y1, x0:x1].astype(np.int16))
    max_rgba_diff = int(diff.max())
    max_alpha_diff = int(diff[:, :, 3].max())
    assert max_rgba_diff <= 1, f"crop-only render must match full-render crop, max RGBA diff={max_rgba_diff}"
    assert max_alpha_diff == 0, f"crop-only alpha must exactly match full-render crop, max alpha diff={max_alpha_diff}"

    expected_2d_shape = (crop_h, crop_w)
    expected_rgb_shape = (crop_h, crop_w, 3)
    assert cropped.rgba.shape == (crop_h, crop_w, 4), "crop-only RGBA must be crop-shaped"
    assert cropped.foreground is not None and cropped.foreground.shape == expected_rgb_shape, "foreground debug RGB must align to crop"
    for name in ("alpha", "background_mask", "edge_mask", "despill_mask", "screen_probability", "alpha_hint", "fringe_mask"):
        arr = getattr(cropped, name)
        assert arr is not None, f"{name} should be available for crop debug views"
        assert arr.shape == expected_2d_shape, f"{name} must be crop-shaped, got {arr.shape}"
    assert np.array_equal(cropped.alpha, full.alpha[y0:y1, x0:x1]), "crop alpha array must align with full alpha crop"
    assert np.array_equal(cropped.fringe_mask, full.fringe_mask[y0:y1, x0:x1]), "crop fringe mask must align with full crop"
    assert np.array_equal(cropped.despill_mask, full.despill_mask[y0:y1, x0:x1]), "crop despill mask must align with full crop"
    assert display_rgb.shape == expected_rgb_shape, "app display RGB crop must align with crop result"

    from app import debug_rgb_to_rgb, mask_to_rgb

    for name in ("alpha", "background_mask", "edge_mask", "despill_mask", "screen_probability", "alpha_hint", "fringe_mask"):
        view_rgb = mask_to_rgb(getattr(cropped, name), display_rgb.shape[:2])
        assert view_rgb.shape == display_rgb.shape, f"{name} debug view should render without shape drift"
    foreground_view = debug_rgb_to_rgb(cropped.foreground, display_rgb.shape[:2])
    assert foreground_view.shape == display_rgb.shape, "foreground debug view should render without shape drift"

    full_tiles = _count_tile_progress(full_stages)
    crop_tiles = _count_tile_progress(crop_stages)
    assert 0 < crop_tiles < full_tiles, f"crop preview should color-render fewer tiles than full render ({crop_tiles}/{full_tiles})"

    forbidden = HEAVY_OPTIONAL_MODULES
    imported = forbidden & set(sys.modules)
    assert not imported, f"crop-only render must not import heavy optional modules: {sorted(imported)}"

    print(
        "Phase 5 crop render checks: "
        f"max_rgba_diff={max_rgba_diff}; max_alpha_diff={max_alpha_diff}; "
        f"tiles {crop_tiles}/{full_tiles}; crop_shape={cropped.rgba.shape}"
    )


def run_phase6_tile_local_nearest_inner_tests() -> None:
    helper_settings = KeySettings(
        key_color=(30, 80, 235),
        auto_border_sample=False,
        edge_color_repair=1.0,
        inner_color_pull=1.0,
        clip_foreground=0.14,
    )
    helper_rgb = np.zeros((12, 14, 3), dtype=np.uint8)
    helper_rgb[:, :] = (30, 80, 235)
    helper_rgb[2:4, 2:6] = (232, 178, 92)
    helper_alpha = np.full((12, 14), 128, dtype=np.uint8)
    helper_probability = np.full((12, 14), 100, dtype=np.uint8)
    helper_background = np.zeros((12, 14), dtype=bool)
    helper_fringe = np.full((12, 14), 160, dtype=np.uint8)
    helper_alpha[2, 2:6] = 255
    helper_alpha[3, 2:5] = 255
    helper_probability[helper_alpha == 255] = 0
    helper_fringe[helper_alpha == 255] = 0
    too_few_nearest, too_few_valid = _build_tile_local_nearest_inner_rgb(
        helper_rgb,
        helper_alpha,
        helper_background,
        helper_probability,
        helper_fringe,
        helper_settings,
        max_radius=8,
    )
    assert too_few_nearest is None and too_few_valid is None, "tile-local pull requires at least 8 clean inner pixels"
    helper_alpha[3, 5] = 255
    helper_probability[3, 5] = 0
    helper_fringe[3, 5] = 0
    nearest, valid = _build_tile_local_nearest_inner_rgb(
        helper_rgb,
        helper_alpha,
        helper_background,
        helper_probability,
        helper_fringe,
        helper_settings,
        max_radius=9,
    )
    assert nearest is not None and valid is not None, "tile-local labels should build once the 8-pixel minimum is met"
    assert nearest.shape == helper_rgb.shape and nearest.dtype == np.uint8, "tile-local nearest RGB must be uint8 read-tile shaped"
    assert valid.shape == helper_alpha.shape and valid.dtype == bool, "tile-local valid mask must be bool read-tile shaped"
    assert valid[9, 11], "fringe pixels inside the bounded local radius should receive a nearest-inner color"
    assert tuple(nearest[9, 11]) == (232, 178, 92), "nearest RGB should come from a clean inner seed pixel"
    _, tight_valid = _build_tile_local_nearest_inner_rgb(
        helper_rgb,
        helper_alpha,
        helper_background,
        helper_probability,
        helper_fringe,
        helper_settings,
        max_radius=3,
    )
    assert tight_valid is not None and not tight_valid[9, 11], "pixels beyond the bounded radius must fall back to unmix+clamp"

    fixture = tile_local_nearest_inner_fixture()
    _, _, soft_edge = _fixture_masks(fixture)
    settings = fixture.settings
    assert _tile_local_nearest_inner_radius(settings) >= 8, "active tile-local pull must reserve read overlap"
    with _temporary_inner_label_cap(1):
        labels, label_to_flat = _build_nearest_inner_label_map(
            np.full((3, 3), 255, dtype=np.uint8),
            np.zeros((3, 3), dtype=bool),
            np.zeros((3, 3), dtype=np.uint8),
            np.pad(np.array([[255]], dtype=np.uint8), ((0, 2), (0, 2))),
            settings,
        )
        assert labels is None and label_to_flat is None, "forced cap should skip the global nearest-inner label map"

        no_pull = process_key_image(fixture.rgb, replace(settings, inner_color_pull=0.0))
        dt_calls: list[tuple[int, int]] = []
        original_distance_transform = keyer_module.cv2.distanceTransformWithLabels

        def recording_distance_transform(src, *args, **kwargs):
            dt_calls.append(tuple(int(v) for v in src.shape[:2]))
            return original_distance_transform(src, *args, **kwargs)

        try:
            keyer_module.cv2.distanceTransformWithLabels = recording_distance_transform
            local_pull = process_key_image(fixture.rgb, settings)
            single_tile = process_key_image(
                fixture.rgb,
                replace(settings, use_tiling=False, tile_size=max(fixture.rgb.shape[:2]) + 1),
            )
        finally:
            keyer_module.cv2.distanceTransformWithLabels = original_distance_transform

        assert single_tile.rgba.shape == local_pull.rgba.shape, "single-tile fallback probe should still render normally"
        assert dt_calls, "tiled cap-forced render should build tile-local distance labels"
        assert fixture.rgb.shape[:2] not in dt_calls, "tile-local fallback must not run distance labels over the full image"
        assert all(h * w <= keyer_module._MAX_TILE_LOCAL_INNER_LABEL_PIXELS for h, w in dt_calls), (
            f"tile-local distance labels exceeded the per-read-tile cap: {dt_calls}"
        )
        no_pull_residual = edge_key_residual(no_pull.rgba, settings.key_color, soft_edge)
        local_residual = edge_key_residual(local_pull.rgba, settings.key_color, soft_edge)
        assert local_residual["max_positive_excess"] <= no_pull_residual["max_positive_excess"], (
            f"tile-local pull should not worsen max edge residual: {no_pull_residual} -> {local_residual}"
        )
        assert local_residual["p95_positive_excess"] <= no_pull_residual["p95_positive_excess"], (
            f"tile-local pull should not worsen p95 edge residual: {no_pull_residual} -> {local_residual}"
        )
        assert local_residual["mean_positive_excess"] < no_pull_residual["mean_positive_excess"] or (
            local_residual["max_positive_excess"] < no_pull_residual["max_positive_excess"]
        ), (
            f"tile-local pull should still improve mean or max residual: {no_pull_residual} -> {local_residual}"
        )

        seam = tile_boundary_band_metrics(fixture.rgb, settings, tile_sizes=(137, 199))
        assert seam["opaque_nonfringe_pixels"] > 0, f"nearest-inner seam test should cover opaque non-fringe pixels: {seam}"
        assert seam["max_alpha_diff"] <= 1, f"nearest-inner boundary alpha diff too high: {seam}"
        assert seam["max_rgb_diff_opaque_nonfringe"] <= 2, f"nearest-inner opaque boundary RGB diff too high: {seam}"
        assert seam["max_checker_diff_visible"] <= 2, f"nearest-inner visible checker seam diff too high: {seam}"

        crop = (310, 210, 620, 450)
        x0, y0, x1, y1 = crop
        full = process_key_image(fixture.rgb, replace(settings, use_tiling=True, tile_size=157, tile_overlap=5, full_res_crop=None))
        cropped = process_key_image(
            fixture.rgb,
            replace(settings, use_tiling=True, tile_size=157, tile_overlap=5, full_res_crop=crop, preview_scale=1.0),
        )
        crop_diff = np.abs(cropped.rgba.astype(np.int16) - full.rgba[y0:y1, x0:x1].astype(np.int16))
        max_crop_rgba_diff = int(crop_diff.max())
        max_crop_alpha_diff = int(crop_diff[:, :, 3].max())
        assert max_crop_rgba_diff <= 1, f"tile-local crop render must match full crop, max RGBA diff={max_crop_rgba_diff}"
        assert max_crop_alpha_diff == 0, f"tile-local crop alpha must exactly match full crop, max alpha diff={max_crop_alpha_diff}"
        assert local_pull.rgba.dtype == np.uint8 and local_pull.foreground is not None and local_pull.foreground.dtype == np.uint8
        assert local_pull.foreground_rgb is None and local_pull.repaired_edge is None, (
            "nearest-inner fallback must not materialize full-image float32 RGB debug buffers"
        )

    forbidden = HEAVY_OPTIONAL_MODULES
    imported = forbidden & set(sys.modules)
    assert not imported, f"tile-local nearest-inner fallback must not import heavy optional modules: {sorted(imported)}"

    print(
        "Phase 6 tile-local nearest-inner checks: "
        f"residual max {no_pull_residual['max_positive_excess']}->{local_residual['max_positive_excess']}; "
        f"p95 {no_pull_residual['p95_positive_excess']}->{local_residual['p95_positive_excess']}; "
        f"seam alpha={seam['max_alpha_diff']} rgb={seam['max_rgb_diff_opaque_nonfringe']} "
        f"checker={seam['max_checker_diff_visible']}; "
        f"crop max_rgba_diff={max_crop_rgba_diff} max_alpha_diff={max_crop_alpha_diff}"
    )


def _write_edge_case_diagnostics(name: str, key_color: tuple[int, int, int]) -> list[str]:
    rgb, _, settings = _edge_fringe_fixture(key_color)
    before_settings = replace(
        settings,
        despill=0.0,
        decontaminate=0.0,
        luminance_restore=0.0,
        unmix_amount=0.0,
        fringe_remove=0.0,
        edge_color_repair=0.0,
        inner_color_pull=0.0,
        luminance_protect=0.0,
    )
    before = process_key_image(rgb, before_settings)
    after = process_key_image(rgb, settings)
    y, x = _find_fringe_sample(rgb, after, key_color)
    sample_metrics = {
        "source_excess": _dominant_excess(rgb[y, x], key_color),
        "before_excess": _dominant_excess(before.rgba[y, x, :3], key_color),
        "after_excess": _dominant_excess(after.rgba[y, x, :3], key_color),
        "alpha": int(after.alpha[y, x]),
        "fringe_mask": 0 if after.fringe_mask is None else int(after.fringe_mask[y, x]),
    }
    fringe = after.fringe_mask if after.fringe_mask is not None else np.zeros(after.alpha.shape, dtype=np.uint8)
    band = (after.alpha > 35) & (after.alpha < 225) & (fringe > 70)
    key_channel = int(np.argmax(np.asarray(key_color)))
    other = [c for c in range(3) if c != key_channel]
    lines = [
        f"{name}: sample=(y={y}, x={x}) alpha={sample_metrics['alpha']} fringe={sample_metrics['fringe_mask']}",
        f"{name}: key-channel excess source={sample_metrics['source_excess']} before={sample_metrics['before_excess']} after={sample_metrics['after_excess']}",
    ]
    if np.any(band):
        def mean_excess(arr: np.ndarray) -> float:
            vals = arr[:, :, key_channel].astype(np.int16) - np.maximum(
                arr[:, :, other[0]].astype(np.int16),
                arr[:, :, other[1]].astype(np.int16),
            )
            return float(np.mean(vals[band]))

        lines.append(
            f"{name}: fringe-band mean excess source={mean_excess(rgb):.2f} "
            f"before={mean_excess(before.rgba[:, :, :3]):.2f} after={mean_excess(after.rgba[:, :, :3]):.2f}"
        )

    _save_rgb(EDGE_ARTIFACT_DIR / f"{name}_source.png", rgb)
    _save_rgba(EDGE_ARTIFACT_DIR / f"{name}_before_repair.png", before.rgba)
    _save_rgba(EDGE_ARTIFACT_DIR / f"{name}_after_repair.png", after.rgba)
    Image.fromarray(after.alpha).save(EDGE_ARTIFACT_DIR / f"{name}_alpha.png")
    Image.fromarray(fringe).save(EDGE_ARTIFACT_DIR / f"{name}_fringe_mask.png")

    backgrounds: dict[str, tuple[int, int, int] | None] = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "checkerboard": None,
    }
    for bg_name, color in backgrounds.items():
        before_rgb = checkerboard_composite(before.rgba) if color is None else _solid_composite(before.rgba, color)
        after_rgb = checkerboard_composite(after.rgba) if color is None else _solid_composite(after.rgba, color)
        _save_rgb(EDGE_ARTIFACT_DIR / f"{name}_before_on_{bg_name}.png", before_rgb)
        _save_rgb(EDGE_ARTIFACT_DIR / f"{name}_after_on_{bg_name}.png", after_rgb)
        _save_rgb(EDGE_ARTIFACT_DIR / f"{name}_compare_on_{bg_name}.png", np.concatenate([before_rgb, after_rgb], axis=1))
    return lines


def write_edge_repair_diagnostics() -> None:
    EDGE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing edge repair diagnostics to {EDGE_ARTIFACT_DIR}")
    lines: list[str] = []
    for name, key_color in (("blue_fringe", (30, 80, 235)), ("green_fringe", (0, 220, 50))):
        lines.extend(_write_edge_case_diagnostics(name, key_color))
    metrics_path = EDGE_ARTIFACT_DIR / "metrics.txt"
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


def _assert_basic_key(fixture: DiagnosticFixture) -> KeyResult:
    result = process_key_image(fixture.rgb, fixture.settings)
    h, w = fixture.rgb.shape[:2]
    assert result.rgba.shape == (h, w, 4)
    assert result.rgba.dtype == np.uint8
    assert result.alpha.dtype == np.uint8
    assert result.background_mask is not None
    assert result.edge_mask is not None
    assert result.despill_mask is not None
    assert result.fringe_mask is not None
    assert result.fringe_mask.shape == (h, w)
    assert result.fringe_mask.dtype == np.uint8
    assert result.alpha[20, 20] <= 4, f"{fixture.name}: keyed border background should be transparent"
    assert result.rgba[20, 20, :3].max() == 0, f"{fixture.name}: transparent RGB should be zeroed"
    assert result.alpha[h // 2, w // 2] >= 248, f"{fixture.name}: foreground core should stay opaque"
    return result


def run_current_baseline() -> np.ndarray:
    fixture = green_flat_fixture()
    result = _assert_basic_key(fixture)
    rgba = result.rgba
    original_alpha = np.ones(fixture.rgb.shape[:2], dtype=np.float32)
    h, w = fixture.rgb.shape[:2]
    original_alpha[h // 2 - 16 : h // 2 + 16, w // 2 - 16 : w // 2 + 16] = 0.5
    compat = process_chroma_key(fixture.rgb, fixture.settings, original_alpha)
    assert compat.shape == rgba.shape
    assert compat.dtype == np.uint8
    assert compat[20, 20, 3] <= 4, "compat wrapper should still key background with original alpha"
    assert 120 <= compat[h // 2, w // 2, 3] <= 130, "original alpha positional path should be preserved"
    preview = checkerboard_composite(compat)
    assert preview.shape == fixture.rgb.shape
    assert preview.dtype == np.uint8
    return rgba


def run_v2_numeric_tests() -> None:
    for fixture in (green_flat_fixture(), blue_flat_fixture(), custom_flat_fixture(), uneven_gradient_fixture()):
        result = _assert_basic_key(fixture)
        h, w = fixture.rgb.shape[:2]
        assert result.alpha[h - 20, w - 20] <= 4, f"{fixture.name}: far border background should be transparent"
        if fixture.name == "green_uneven_gradient":
            foreground = _disc_alpha(h, w, 160).astype(bool)
            background = ~_disc_alpha(h, w, 190).astype(bool)
            bg_transparent = float(np.mean(result.alpha[background] <= 8))
            fg_opaque = float(np.mean(result.alpha[foreground] >= 245))
            assert bg_transparent > 0.985, f"uneven background should be mostly removed, got {bg_transparent:.3f}"
            assert fg_opaque > 0.985, f"uneven foreground should be preserved, got {fg_opaque:.3f}"

    blue = blue_flat_fixture()
    auto_blue = process_key_image(
        blue.rgb,
        replace(blue.settings, key_color=(0, 220, 50), auto_detect_key_color=True),
    )
    assert auto_blue.alpha[20, 20] <= 4, "Auto mode should detect blue border even when seeded with default green"
    assert auto_blue.screen_color is not None and auto_blue.screen_color[2] > auto_blue.screen_color[1], (
        f"Auto mode should report a blue screen color, got {auto_blue.screen_color}"
    )

    island = same_color_island_fixture()
    h, w = island.rgb.shape[:2]
    island_y = h // 2 - 20
    island_x = w // 2
    default = process_key_image(island.rgb, island.settings)
    assert default.alpha[island_y, island_x] >= 248, "default connected-background policy must preserve same-key foreground islands"
    aggressive = process_key_image(
        island.rgb,
        replace(island.settings, aggressive_interior_removal=True),
    )
    assert aggressive.alpha[island_y, island_x] <= 4, "aggressive interior removal should remove disconnected high-confidence key islands"

    keep_mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.indices((h, w))
    keep_mask[(xx - island_x) ** 2 + (yy - island_y) ** 2 < 45**2] = 255
    protected = process_key_image(
        island.rgb,
        replace(island.settings, aggressive_interior_removal=True),
        keep_mask=keep_mask,
    )
    assert protected.alpha[island_y, island_x] >= 248, "keep mask should protect same-key foreground details"

    alpha_hint = np.zeros((h, w), dtype=np.uint8)
    alpha_hint[(xx - island_x) ** 2 + (yy - island_y) ** 2 < 48**2] = 255
    hinted = process_key_image(
        island.rgb,
        replace(island.settings, aggressive_interior_removal=True),
        alpha_hint=alpha_hint,
    )
    assert hinted.alpha[island_y, island_x] >= 248, "imported matte should protect foreground details"
    assert hinted.alpha_hint is not None and hinted.alpha_hint[island_y, island_x] == 255, "imported matte should be returned for UI/debug views"

    remove_mask = np.zeros((h, w), dtype=np.uint8)
    remove_mask[(xx - island_x) ** 2 + (yy - island_y) ** 2 < 35**2] = 255
    removed = process_key_image(island.rgb, island.settings, remove_mask=remove_mask)
    assert removed.alpha[island_y, island_x] <= 4, "remove mask should force interior background cleanup"

    aa = antialiased_edge_fixture()
    aa_result = process_key_image(aa.rgb, aa.settings)
    h, w = aa.rgb.shape[:2]
    mid_y = h // 2
    mid_x = w // 2 + 185
    mid_alpha = int(aa_result.alpha[mid_y, mid_x])
    assert 35 <= mid_alpha <= 220, f"anti-aliased edge should retain soft alpha, got {mid_alpha}"
    soft_pixels = np.count_nonzero((aa_result.alpha > 12) & (aa_result.alpha < 243))
    assert soft_pixels > 900, "edge-only refinement should produce a meaningful soft edge band"
    assert aa_result.alpha[20, 20] <= 4, "flat background must remain exact alpha 0"
    assert aa_result.alpha[h // 2, w // 2] >= 248, "core foreground must remain exact alpha 255"

    src = aa.rgb[mid_y, mid_x].astype(int)
    out = aa_result.rgba[mid_y, mid_x, :3].astype(int)
    src_spill = int(src[1] - max(src[0], src[2]))
    out_spill = int(out[1] - max(out[0], out[2]))
    assert out_spill <= max(8, src_spill - 35), f"despill should reduce green edge contamination: {src_spill} -> {out_spill}"
    assert aa_result.rgba[aa_result.alpha == 0, :3].max() == 0, "transparent output RGB must be zero"

    large = large_synthetic_fixture()
    tiled_settings = replace(
        large.settings,
        tile_size=384,
        tile_overlap=48,
        use_tiling=True,
    )
    full_settings = replace(
        tiled_settings,
        use_tiling=False,
    )
    tiled = process_key_image(large.rgb, tiled_settings)
    full = process_key_image(large.rgb, full_settings)
    diff = np.abs(tiled.rgba.astype(np.int16) - full.rgba.astype(np.int16))
    assert int(diff[:, :, 3].max()) <= 0, "tiled alpha must match non-tiled global alpha exactly"
    assert int(diff.max()) <= 1, f"tiled core writes should be seam-free, max RGBA diff={int(diff.max())}"


def run_v4_edge_repair_tests() -> None:
    for name, key_color in (("blue", (30, 80, 235)), ("green", (0, 220, 50))):
        rgb, true_alpha, settings = _edge_fringe_fixture(key_color)
        result = process_key_image(rgb, settings)
        y, x = _find_fringe_sample(rgb, result, key_color)
        src_excess = _dominant_excess(rgb[y, x], key_color)
        out_excess = _dominant_excess(result.rgba[y, x, :3], key_color)
        assert src_excess > 18, f"{name}: fixture should contain measurable key-channel fringe, got {src_excess}"
        assert out_excess <= max(6, int(src_excess * 0.40)), (
            f"{name}: edge repair should reduce key-channel excess by >=60%, {src_excess} -> {out_excess}"
        )
        assert result.fringe_mask is not None and result.fringe_mask[y, x] >= 70, f"{name}: fringe mask should mark contaminated edge"
        h, w = rgb.shape[:2]
        assert result.fringe_mask[h // 2, w // 2] <= 4, f"{name}: opaque interior should not be marked as fringe"
        interior = true_alpha > 0.995
        max_delta = int(np.abs(result.rgba[interior, :3].astype(np.int16) - rgb[interior].astype(np.int16)).max())
        assert max_delta <= 3, f"{name}: opaque interior RGB should be preserved, max delta={max_delta}"
        assert result.rgba[result.alpha == 0, :3].max() == 0, f"{name}: transparent RGB must be zero after repair"

    key_color = (30, 80, 235)
    foreground = (232, 178, 92)
    rgb, _, settings = _edge_fringe_fixture(key_color, foreground=foreground)
    pull_base = replace(
        settings,
        fringe_remove=0.0,
        unmix_amount=0.0,
        edge_color_repair=1.0,
        inner_color_pull=0.0,
        decontaminate=1.0,
        luminance_restore=0.0,
        luminance_protect=0.0,
    )
    no_pull = process_key_image(rgb, pull_base)
    with_pull = process_key_image(rgb, replace(pull_base, inner_color_pull=1.0))
    y, x = _find_fringe_sample(rgb, with_pull, key_color)
    target = np.asarray(foreground, dtype=np.float32)
    no_pull_dist = float(np.linalg.norm(no_pull.rgba[y, x, :3].astype(np.float32) - target))
    pull_dist = float(np.linalg.norm(with_pull.rgba[y, x, :3].astype(np.float32) - target))
    assert pull_dist <= no_pull_dist * 0.72, (
        f"nearest-inner color pull should move fringe RGB toward foreground core, {no_pull_dist:.2f} -> {pull_dist:.2f}"
    )

    luma_settings = replace(settings, luminance_restore=0.0, luminance_protect=1.0)
    luma_result = process_key_image(rgb, luma_settings)
    y, x = _find_fringe_sample(rgb, luma_result, key_color)
    src_luma = _luma(rgb[y, x])
    out_luma = _luma(luma_result.rgba[y, x, :3])
    assert abs(out_luma - src_luma) / max(src_luma, 1.0) <= 0.15, (
        f"luminance protection should keep repaired edge luma bounded, {src_luma:.2f} -> {out_luma:.2f}"
    )

    seam_settings = replace(settings, use_tiling=True, tile_size=53, tile_overlap=7)
    tiled = process_key_image(rgb, seam_settings)
    full = process_key_image(rgb, replace(seam_settings, use_tiling=False))
    diff = np.abs(tiled.rgba.astype(np.int16) - full.rgba.astype(np.int16))
    assert int(diff[:, :, 3].max()) == 0, "v4 tiled repair alpha must match non-tiled alpha exactly"
    assert int(diff.max()) <= 1, f"v4 tiled edge repair must be seam-free, max RGBA diff={int(diff.max())}"

    compat = process_chroma_key(rgb, settings)
    assert isinstance(compat, np.ndarray) and compat.shape == (*rgb.shape[:2], 4), "process_chroma_key must still return RGBA ndarray"
    assert compat.dtype == np.uint8, "process_chroma_key compatibility output must stay uint8"
    assert compat[compat[:, :, 3] == 0, :3].max() == 0, "compat wrapper must zero transparent RGB"

    debug_result = process_key_image(rgb, settings)
    low_memory = process_key_image(rgb, settings, include_debug=False)
    assert np.array_equal(low_memory.rgba, debug_result.rgba), "low-memory result mode must preserve RGBA output"
    assert np.array_equal(compat, low_memory.rgba), "process_chroma_key must keep RGBA compatibility in low-memory mode"
    assert np.shares_memory(low_memory.alpha, low_memory.rgba), "low-memory alpha should be a view of output RGBA alpha"
    assert low_memory.foreground is None and low_memory.despill_mask is None, "low-memory mode should not retain RGB/mask debug arrays"
    for name in ("background_mask", "edge_mask", "screen_probability", "alpha_hint", "fringe_mask"):
        assert getattr(low_memory, name) is None, f"low-memory mode should not retain {name}"

    control_key = (0, 220, 50)
    control_rgb, _, control_settings = _edge_fringe_fixture(control_key)
    legacy_control_settings = replace(control_settings, transition_unmix=False)
    low_cleanup = process_key_image(control_rgb, replace(legacy_control_settings, decontaminate=0.0, despill=0.0))
    high_cleanup = process_key_image(control_rgb, replace(legacy_control_settings, decontaminate=1.0, despill=1.0))
    y, x = _find_fringe_sample(control_rgb, high_cleanup, control_key)
    low_excess = _dominant_excess(low_cleanup.rgba[y, x, :3], control_key)
    high_excess = _dominant_excess(high_cleanup.rgba[y, x, :3], control_key)
    assert high_excess <= low_excess - 20, (
        f"despill/decontaminate controls should strengthen edge cleanup, excess {low_excess} -> {high_excess}"
    )

    oversized_shape = (1, _MAX_INNER_LABEL_PIXELS + 1)
    oversized_fringe = np.zeros(oversized_shape, dtype=np.uint8)
    oversized_fringe[0, 0] = 255
    labels, label_to_flat = _build_nearest_inner_label_map(
        np.full(oversized_shape, 255, dtype=np.uint8),
        np.zeros(oversized_shape, dtype=bool),
        np.zeros(oversized_shape, dtype=np.uint8),
        oversized_fringe,
        settings,
    )
    assert labels is None and label_to_flat is None, "oversized exports should skip nearest-inner labels and fall back safely"

    forbidden = HEAVY_OPTIONAL_MODULES
    imported = forbidden & set(sys.modules)
    assert not imported, f"default v4 edge repair path must not import heavy optional modules: {sorted(imported)}"


def run_transition_unmix_baseline_tests() -> None:
    summaries: list[str] = []
    improved_residual_count = 0
    total_recovered_pixels = 0
    for fixture in transition_unmix_baseline_fixtures():
        baseline = _process_fixture_result(fixture, replace(fixture.settings, transition_unmix=False))
        result = _process_fixture_result(fixture)
        alpha_only = _process_fixture_result(
            fixture,
            replace(
                fixture.settings,
                key_vector_despill=0.0,
                foreground_reference_pull=0.0,
                preserve_foreground_luma=0.0,
            ),
        )
        assert np.array_equal(result.alpha, alpha_only.alpha), (
            f"{fixture.name}: RGB transition cleanup knobs must not mutate recovered global alpha"
        )
        assert np.array_equal(result.alpha, result.rgba[:, :, 3]), f"{fixture.name}: debug alpha must match output alpha channel"
        export = _process_fixture_result(fixture, include_debug=False)
        compat = process_chroma_key(
            fixture.rgb,
            fixture.settings,
            fixture.original_alpha,
            keep_mask=fixture.keep_mask,
            remove_mask=fixture.remove_mask,
            alpha_hint=fixture.alpha_hint,
        )
        assert np.array_equal(export.rgba, compat), f"{fixture.name}: export wrapper must match low-memory render"
        assert np.array_equal(export.rgba, result.rgba), f"{fixture.name}: preview/debug render must match export RGB cleanup"
        metrics = _transition_unmix_baseline_metrics_for_fixture(fixture, result)
        baseline_metrics = _transition_unmix_baseline_metrics_for_fixture(fixture, baseline)
        comparison = _transition_unmix_comparison_metrics(fixture, baseline, result)
        known_background, foreground_core, soft_edge = _fixture_masks(fixture)
        detail_region = _transition_detail_region(fixture, result)
        alpha_delta = result.alpha.astype(np.int16) - baseline.alpha.astype(np.int16)
        screen_like_detail = detail_region & (result.screen_probability >= int(round(fixture.settings.clip_background * 255.0)))
        protected_detail = detail_region & ~screen_like_detail
        if np.any(protected_detail):
            assert int(alpha_delta[protected_detail].min()) >= 0, (
                f"{fixture.name}: transition alpha recovery must not erode non-screen detail alpha"
            )
        total_recovered_pixels += int(np.count_nonzero(alpha_delta[detail_region] > 0))
        confident_background = np.zeros(result.alpha.shape, dtype=bool)
        if result.background_mask is not None and result.edge_mask is not None:
            confident_background = (result.background_mask > 0) & (result.edge_mask == 0)
        if np.any(confident_background):
            assert int(result.alpha[confident_background].max()) == 0, (
                f"{fixture.name}: confident connected background alpha must remain zero"
            )
        transparent = metrics["transparent_rgb_zero"]
        leak = metrics["background_alpha_leak"]
        core_delta = metrics["hard_edge_core_rgb_delta"]
        residual = metrics["transition_key_residual"]
        baseline_residual = baseline_metrics["transition_key_residual"]
        recall = metrics["alpha_detail_recall"]
        baseline_recall = baseline_metrics["alpha_detail_recall"]
        assert transparent["ok"], (
            f"{fixture.name}: transparent output RGB must stay zero, max={transparent['max_rgb_when_transparent']}"
        )
        assert metrics["transparent_rgb_residual_max"] == 0, f"{fixture.name}: transparent RGB residual must be zero"
        assert leak["max_alpha"] == 0, f"{fixture.name}: screen-like known background must clean to alpha zero: {leak}"
        if core_delta["count"]:
            assert core_delta["max_delta"] <= 8, f"{fixture.name}: hard/core RGB drift too high: {core_delta}"
        foreground_delta = metrics["foreground_core_rgb_delta"]
        if foreground_delta["count"]:
            assert foreground_delta["max_delta"] <= 5, (
                f"{fixture.name}: foreground core RGB delta must stay <=5: {foreground_delta}"
            )
        if fixture.expected_alpha is not None and fixture.expected_foreground_rgb is not None:
            exact_core = (fixture.expected_alpha >= 0.999) & (result.alpha >= 250)
            expected_rgb = _expected_foreground_array(fixture)
            if np.any(exact_core) and expected_rgb is not None:
                exact_delta = int(
                    np.abs(result.rgba[exact_core, :3].astype(np.int16) - expected_rgb[exact_core].astype(np.int16)).max()
                )
                assert exact_delta <= 5, f"{fixture.name}: exact opaque foreground core drift too high: {exact_delta}"
        assert recall["visible_recall"] >= 0.65, f"{fixture.name}: diagnostic detail recall unexpectedly low: {recall}"
        assert recall["visible_recall"] >= baseline_recall["visible_recall"], (
            f"{fixture.name}: detail visible recall decreased: {baseline_recall} -> {recall}"
        )
        assert recall["mean_alpha_ratio"] + 1e-6 >= baseline_recall["mean_alpha_ratio"], (
            f"{fixture.name}: detail alpha recall decreased: {baseline_recall} -> {recall}"
        )
        assert residual["max_positive_excess"] <= baseline_residual["max_positive_excess"], (
            f"{fixture.name}: transition RGB repair worsened max key residual: {baseline_residual} -> {residual}"
        )
        assert residual["p95_positive_excess"] <= baseline_residual["p95_positive_excess"], (
            f"{fixture.name}: transition RGB repair worsened p95 key residual: {baseline_residual} -> {residual}"
        )
        baseline_has_residual = (
            baseline_residual["mean_positive_excess"] > 0
            or baseline_residual["p95_positive_excess"] > 0
            or baseline_residual["max_positive_excess"] > 0
        )
        improved = (
            residual["mean_positive_excess"] < baseline_residual["mean_positive_excess"]
            or residual["p95_positive_excess"] < baseline_residual["p95_positive_excess"]
            or residual["max_positive_excess"] < baseline_residual["max_positive_excess"]
        )
        if baseline_has_residual:
            assert improved, f"{fixture.name}: transition key residual should decrease: {baseline_residual} -> {residual}"
            assert residual["mean_positive_excess"] < baseline_residual["mean_positive_excess"], (
                f"{fixture.name}: transition key residual mean must decrease: {baseline_residual} -> {residual}"
            )
        else:
            assert residual["mean_positive_excess"] == 0 and residual["max_positive_excess"] == 0, (
                f"{fixture.name}: screen cleanup should leave no transition key residual: {baseline_residual} -> {residual}"
            )
        _assert_transition_composites_do_not_worsen(fixture, baseline_metrics, metrics)
        improved_residual_count += 1

        if fixture.name == "transition_black_tape_edge":
            dark_edge = soft_edge & (result.alpha >= 16)
            assert np.any(dark_edge), f"{fixture.name}: black edge fixture should expose a visible transition band"
            max_dark_rgb = int(result.rgba[dark_edge, :3].max())
            assert max_dark_rgb <= 4, f"{fixture.name}: black transition edges must not be lifted/yellowed, max RGB={max_dark_rgb}"
        if fixture.name == "transition_white_black_1px_lines" and fixture.expected_foreground_rgb is not None:
            expected_rgb = _expected_foreground_array(fixture)
            assert expected_rgb is not None and fixture.expected_alpha is not None
            white_pixels = (expected_rgb[:, :, 0] >= 250) & (fixture.expected_alpha > 0) & (result.alpha > 0)
            black_pixels = (expected_rgb[:, :, 0] <= 5) & (fixture.expected_alpha > 0) & (result.alpha > 0)
            assert np.any(white_pixels) and np.any(black_pixels), "white/black line fixture should contain both polarities"
            assert int(result.rgba[white_pixels, :3].min()) >= 252, f"{fixture.name}: white edges must stay white"
            assert int(result.rgba[black_pixels, :3].max()) <= 3, f"{fixture.name}: black line edges must stay black"
        if fixture.original_alpha is not None:
            _assert_transition_source_alpha_cap(fixture, result)

        checker = (metrics.get("composite_residuals") or {}).get("checker", {})
        summaries.append(
            f"{fixture.name}: residual_max={baseline_residual['max_positive_excess']}->{residual['max_positive_excess']} "
            f"core_delta={foreground_delta['max_delta']} recall={recall['visible_recall']:.3f} "
            f"bg_leak={leak['max_alpha']} checker_mean={checker.get('mean_abs_error', 0.0):.2f} "
            f"recovered={comparison['alpha_recovered_pixel_count']}"
        )

    assert improved_residual_count == len(transition_unmix_baseline_fixtures()), "all transition fixtures should improve key residual"
    assert total_recovered_pixels > 0, "v7 alpha recovery should raise plausible transition pixels on diagnostic fixtures"
    _assert_transition_manual_mask_regressions()
    _assert_transition_imported_matte_regression()
    _assert_transition_rgb_repair_helper_contract()
    _assert_transition_unmix_mask_helpers()
    _assert_transition_foreground_reference_radius()
    _assert_transition_alpha_recovery_parity()
    print("Phase 3 transition-unmix RGB checks:")
    for line in summaries:
        print(f"  {line}")


def _assert_transition_composites_do_not_worsen(
    fixture: DiagnosticFixture,
    baseline_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    before = baseline_metrics.get("composite_residuals") or {}
    after = metrics.get("composite_residuals") or {}
    for background_name in ("black", "white", "gray", "checker"):
        before_bg = before.get(background_name)
        after_bg = after.get(background_name)
        if not isinstance(before_bg, dict) or not isinstance(after_bg, dict):
            continue
        assert after_bg["mean_abs_error"] <= before_bg["mean_abs_error"] + 1e-6, (
            f"{fixture.name}: {background_name} composite mean residual worsened: {before_bg} -> {after_bg}"
        )
        assert after_bg["max_abs_error"] <= before_bg["max_abs_error"], (
            f"{fixture.name}: {background_name} composite max residual worsened: {before_bg} -> {after_bg}"
        )


def _assert_transition_source_alpha_cap(fixture: DiagnosticFixture, result: KeyResult) -> None:
    assert fixture.original_alpha is not None
    cap_u8 = np.rint(np.clip(fixture.original_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    over_cap = int(np.maximum(result.alpha.astype(np.int16) - cap_u8.astype(np.int16), 0).max())
    assert over_cap <= 1, f"{fixture.name}: output alpha exceeded source-alpha cap by {over_cap}"

    semi_transparent_source = (cap_u8 > 0) & (cap_u8 < 255)
    assert np.any(semi_transparent_source), f"{fixture.name}: source-alpha fixture must contain semi-transparent pixels"
    semi_over_cap = int(
        np.maximum(result.alpha[semi_transparent_source].astype(np.int16) - cap_u8[semi_transparent_source].astype(np.int16), 0).max()
    )
    assert semi_over_cap <= 1, f"{fixture.name}: semi-transparent source alpha cap exceeded by {semi_over_cap}"

    transparent_source = cap_u8 == 0
    assert np.any(transparent_source), f"{fixture.name}: source-alpha fixture must contain fully transparent pixels"
    assert int(fixture.rgb[transparent_source, :3].max()) > 0, (
        f"{fixture.name}: transparent-source regression must include nonzero source RGB"
    )
    assert int(result.alpha[transparent_source].max()) == 0, f"{fixture.name}: source-transparent alpha must remain zero"
    assert int(result.rgba[transparent_source, :3].max()) == 0, f"{fixture.name}: source-transparent RGB must be zeroed"


def _assert_transition_manual_mask_regressions() -> None:
    keep_fixture = manual_keep_transition_fixture()
    keep_result = _process_fixture_result(keep_fixture)
    keep_baseline = _process_fixture_result(keep_fixture, replace(keep_fixture.settings, transition_unmix=False))
    keep = keep_fixture.keep_mask.astype(bool)
    assert np.any(keep), "manual keep transition fixture must contain keep pixels"
    _, keep_core, keep_soft = _fixture_masks(keep_fixture)
    keep_core &= keep
    keep_transition = keep_soft & keep
    assert np.any(keep_core), "manual keep fixture must cover foreground core"
    assert np.any(keep_transition), "manual keep fixture must cover transition pixels"
    assert int(keep_result.alpha[keep].min()) == 255, "manual keep must force kept transition/core alpha to 255"
    assert int((keep_result.alpha.astype(np.int16) - keep_baseline.alpha.astype(np.int16))[keep].min()) >= 0, (
        "manual keep must not reduce alpha relative to transition-unmix-disabled baseline"
    )
    expected_rgb = _expected_foreground_array(keep_fixture)
    assert expected_rgb is not None
    keep_core_delta = int(
        np.abs(keep_result.rgba[keep_core, :3].astype(np.int16) - expected_rgb[keep_core].astype(np.int16)).max()
    )
    assert keep_core_delta <= 5, f"manual keep must protect foreground core color, delta={keep_core_delta}"

    remove_fixture = manual_remove_transition_fixture()
    remove = remove_fixture.remove_mask.astype(bool)
    assert np.any(remove), "manual remove transition fixture must contain remove pixels"
    remove_result = _process_fixture_result(remove_fixture)
    _, _, remove_soft = _fixture_masks(remove_fixture)
    remove_transition = remove & remove_soft
    remove_background = remove & remove_fixture.known_background_mask.astype(bool)
    assert np.any(remove_transition), "manual remove fixture must cover transition pixels"
    assert np.any(remove_background), "manual remove fixture must cover background pixels"
    assert int(remove_result.alpha[remove].max()) == 0, "manual remove must force transition/background alpha to zero"
    assert int(remove_result.rgba[remove, :3].max()) == 0, "manual remove must force transition/background RGB to zero"

    overlap_fixture = replace(keep_fixture, remove_mask=keep_fixture.keep_mask)
    overlap_result = _process_fixture_result(overlap_fixture)
    assert int(overlap_result.alpha[keep].min()) == 255, "manual keep must override overlapping remove mask"


def _assert_transition_imported_matte_regression() -> None:
    fixture = red_slash_blue_transition_fixture()
    h, w = fixture.rgb.shape[:2]
    yy, xx = np.indices((h, w))
    central_band = (xx >= 105) & (xx <= 210) & (yy >= 42) & (yy <= 170)
    hint_region = fixture.soft_edge_mask.astype(bool) & central_band
    assert np.any(hint_region), "imported matte transition regression must hint transition pixels"
    alpha_hint = np.zeros((h, w), dtype=np.uint8)
    alpha_hint[hint_region] = 255
    settings = replace(fixture.settings, mode="ImportedMatte", alpha_hint_strength=1.0)
    control = process_key_image(fixture.rgb, settings)
    hinted = process_key_image(fixture.rgb, settings, alpha_hint=alpha_hint)
    assert hinted.alpha_hint is not None and np.array_equal(hinted.alpha_hint, alpha_hint), "imported matte must be returned for debug/UI"
    assert int((hinted.alpha.astype(np.int16) - control.alpha.astype(np.int16))[hint_region].min()) >= 0, (
        "imported matte must not reduce hinted transition alpha"
    )
    assert int(hinted.alpha[hint_region].min()) == 255, "imported matte must protect hinted transition pixels as foreground"
    if hinted.background_mask is not None:
        connected_background = hinted.background_mask > 0
        if np.any(connected_background):
            assert int(hinted.alpha[connected_background].max()) == 0, "imported matte must keep connected background alpha zero"
    transparent = transparent_rgb_zero(hinted.rgba)
    assert transparent["ok"], f"imported matte path must preserve transparent RGB zeroing: {transparent}"


def _assert_transition_rgb_repair_helper_contract() -> None:
    settings = _graphic_transition_settings((30, 80, 235))
    h, w = 5, 7
    alpha = np.tile(np.asarray([0.0, 0.18, 0.36, 0.62, 0.82, 1.0, 0.0], dtype=np.float32), (h, 1))
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = settings.key_color
    foreground = np.zeros_like(background)
    foreground[:, :] = (228, 18, 16)
    rgb = _composite_rgb_linear(background, foreground, alpha)
    alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
    known_background = alpha_u8 == 0
    edge = (alpha_u8 > 0) & (alpha_u8 < 255)
    probability = np.where(known_background, 255, 96).astype(np.uint8)
    probability[:, 5] = 0
    fringe = np.where(edge, 180, 0).astype(np.uint8)
    nearest_fg_rgb = foreground.copy()
    nearest_fg_valid = np.ones((h, w), dtype=bool)
    alpha_before = alpha_u8.copy()

    repaired, repair_mask = _repair_transition_unmix(
        rgb,
        alpha_u8,
        known_background,
        edge,
        probability,
        fringe,
        settings.key_color,
        None,
        nearest_fg_rgb,
        nearest_fg_valid,
        settings,
    )
    assert repaired.shape == rgb.shape and repaired.dtype == np.uint8, "transition helper must return uint8 RGB tile shape"
    assert repair_mask.shape == alpha_u8.shape and repair_mask.dtype == np.uint8, "transition helper mask must be uint8 alpha shape"
    assert np.array_equal(alpha_u8, alpha_before), "transition RGB helper must not mutate alpha input"
    transition = edge & (alpha_u8 > 0)
    before_residual = rgb_key_residual(rgb, settings.key_color, transition)
    after_residual = rgb_key_residual(repaired, settings.key_color, transition)
    assert after_residual["max_positive_excess"] < before_residual["max_positive_excess"], (
        f"transition helper should reduce key residual: {before_residual} -> {after_residual}"
    )
    assert np.count_nonzero(repair_mask[transition]) > 0, "transition helper should mark repaired transition pixels"
    assert np.array_equal(repaired[alpha_u8 == 255], rgb[alpha_u8 == 255]), "opaque protected helper core must stay unchanged"

    no_ref_rgb, no_ref_mask = _repair_transition_unmix(
        rgb,
        alpha_u8,
        known_background,
        edge,
        probability,
        fringe,
        settings.key_color,
        None,
        None,
        None,
        settings,
    )
    assert np.array_equal(no_ref_rgb, rgb), "helper without foreground reference must return original RGB"
    assert int(no_ref_mask.max()) == 0, "helper without foreground reference must return a zero repair mask"


def _assert_transition_foreground_reference_radius() -> None:
    settings = replace(
        _graphic_transition_settings((30, 80, 235)),
        edge_color_repair=0.0,
        inner_color_pull=0.0,
        foreground_reference_radius=4,
    )
    h, w = 10, 24
    alpha = np.full((h, w), 128, dtype=np.uint8)
    alpha[1:5, 1:3] = 255
    background_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    background_rgb[:, :] = settings.key_color
    foreground_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    foreground_rgb[:, :] = (232, 40, 32)
    rgb = _composite_rgb_linear(background_rgb, foreground_rgb, alpha.astype(np.float32) / 255.0)
    background = np.zeros((h, w), dtype=bool)
    probability = np.full((h, w), 120, dtype=np.uint8)
    probability[1:5, 1:3] = 0
    fringe = np.full((h, w), 160, dtype=np.uint8)
    fringe[1:5, 1:3] = 0

    labels, label_to_flat, distance = _build_nearest_inner_reference_map(alpha, background, probability, fringe, settings)
    assert labels is not None and label_to_flat is not None and distance is not None, (
        "v7 foreground references must build even when legacy color-pull sliders are disabled"
    )
    assert distance.dtype == np.uint16, "global foreground reference distance map must stay compact"
    ref_rgb, ref_valid, ref_distance = _foreground_reference_for_slice(
        rgb,
        labels,
        label_to_flat,
        distance,
        slice(0, h),
        slice(0, w),
        settings.foreground_reference_radius,
    )
    assert ref_rgb is not None and ref_valid is not None and ref_distance is not None
    assert ref_valid[2, 5], "near transition pixels should receive a radius-valid foreground reference"
    assert tuple(ref_rgb[2, 5]) == (232, 40, 32), "foreground reference RGB should come from a clean inner seed"
    assert not ref_valid[2, 16], "far transition pixels must not borrow unrelated foreground references"

    local_ref, local_valid, local_distance = _build_tile_local_nearest_inner_reference(
        rgb,
        alpha,
        background,
        probability,
        fringe,
        settings,
        max_radius=settings.foreground_reference_radius,
    )
    assert local_ref is not None and local_valid is not None and local_distance is not None
    assert local_valid[2, 5] and not local_valid[2, 16], "tile-local references must honor foreground_reference_radius"

    repair_rgb, repair_mask = _repair_transition_unmix(
        rgb,
        alpha,
        background,
        alpha < 255,
        probability,
        fringe,
        settings.key_color,
        None,
        ref_rgb,
        ref_valid,
        settings,
    )
    assert repair_mask[2, 5] > 0, "near radius-valid transition pixels should be repaired"
    assert repair_mask[2, 16] == 0, "far transition pixels must not be repaired without a radius-valid reference"
    assert np.array_equal(repair_rgb[2, 16], rgb[2, 16]), "far pixels must not borrow distant foreground RGB"

    render_settings = replace(
        settings,
        use_tiling=False,
        local_screen_model=False,
        edge_color_repair=0.0,
        inner_color_pull=0.0,
        despill=0.0,
        decontaminate=0.0,
    )
    matte = keyer_module._GlobalMatte(
        screen_color=settings.key_color,
        screen_probability=probability,
        screen_map=None,
        background_mask=background,
        edge_mask=alpha < 255,
        alpha=alpha,
        color_alpha=None,
        alpha_hint=None,
        fringe_mask=fringe,
        inner_labels=labels,
        inner_label_to_flat=label_to_flat,
        inner_distance=distance,
    )
    rgba, repair_debug = keyer_module._render_tiled_rgba(rgb, render_settings, matte, None, None, include_debug=True)
    assert repair_debug is not None
    assert repair_debug[2, 5] > 0, "render path should repair radius-valid transition pixels"
    assert repair_debug[2, 16] == 0, "render path must honor foreground_reference_radius for transition RGB repair"
    assert np.array_equal(rgba[2, 16, :3], rgb[2, 16]), "render path must not repair beyond-radius transition RGB"

    tile_h, tile_w = 40, 120
    tile_alpha = np.full((tile_h, tile_w), 128, dtype=np.uint8)
    tile_alpha[4:18, 8:10] = 255
    tile_bg_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile_bg_rgb[:, :] = settings.key_color
    tile_fg_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile_fg_rgb[:, :] = (232, 40, 32)
    tile_rgb = _composite_rgb_linear(tile_bg_rgb, tile_fg_rgb, tile_alpha.astype(np.float32) / 255.0)
    tile_background = np.zeros((tile_h, tile_w), dtype=bool)
    tile_probability = np.full((tile_h, tile_w), 120, dtype=np.uint8)
    tile_probability[tile_alpha == 255] = 0
    tile_fringe = np.where(tile_alpha < 255, 180, 0).astype(np.uint8)
    tile_settings = replace(
        settings,
        foreground_reference_radius=4,
        use_tiling=True,
        tile_size=24,
        tile_overlap=0,
        local_screen_model=False,
        edge_refine_radius=8,
        edge_color_repair=1e-6,
        inner_color_pull=1e-6,
        despill=0.0,
        fringe_remove=0.0,
        unmix_amount=0.0,
        decontaminate=0.0,
    )
    tile_matte = keyer_module._GlobalMatte(
        screen_color=settings.key_color,
        screen_probability=tile_probability,
        screen_map=None,
        background_mask=tile_background,
        edge_mask=tile_alpha < 255,
        alpha=tile_alpha,
        color_alpha=None,
        alpha_hint=None,
        fringe_mask=tile_fringe,
        inner_labels=None,
        inner_label_to_flat=None,
        inner_distance=None,
    )
    tiled_rgba, tiled_debug = keyer_module._render_tiled_rgba(tile_rgb, tile_settings, tile_matte, None, None, include_debug=True)
    assert tiled_debug is not None
    assert tiled_debug[10, 12] > 0, "tile-local transition repair should use references inside foreground_reference_radius"
    assert tiled_debug[10, 35] == 0, "tile-local transition repair must use its own radius, not the larger legacy radius"
    assert np.array_equal(tiled_rgba[10, 35, :3], tile_rgb[10, 35]), "tile-local repair must not borrow beyond-radius foreground RGB"


def _assert_transition_alpha_recovery_parity() -> None:
    fixture = black_tape_edge_transition_fixture()
    settings = replace(fixture.settings, use_tiling=True, tile_size=53, tile_overlap=4)
    tiled = process_key_image(fixture.rgb, settings)
    full = process_key_image(fixture.rgb, replace(settings, use_tiling=False))
    diff = np.abs(tiled.rgba.astype(np.int16) - full.rgba.astype(np.int16))
    assert int(diff[:, :, 3].max()) == 0, "transition alpha recovery must be identical for tiled/full render"
    assert int(diff.max()) <= 1, f"transition alpha recovery tiled/full RGBA drift too high: {int(diff.max())}"

    crop = (38, 40, 240, 150)
    x0, y0, x1, y1 = crop
    cropped = process_key_image(
        fixture.rgb,
        replace(settings, full_res_crop=crop, preview_scale=1.0),
    )
    crop_diff = np.abs(cropped.rgba.astype(np.int16) - full.rgba[y0:y1, x0:x1].astype(np.int16))
    assert int(crop_diff[:, :, 3].max()) == 0, "transition alpha recovery crop alpha must match full crop"
    assert int(crop_diff.max()) <= 1, f"transition alpha recovery crop RGBA drift too high: {int(crop_diff.max())}"


def _assert_transition_unmix_mask_helpers() -> None:
    settings = _graphic_transition_settings((30, 80, 235))
    for fixture in (
        red_slash_blue_transition_fixture(),
        white_black_barcode_transition_fixture(),
        black_tape_edge_transition_fixture(),
    ):
        known_background, foreground_core, soft_edge = _fixture_masks(fixture)
        alpha_u8 = np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        probability = np.where(known_background, 255, 0).astype(np.uint8)
        fringe = np.where(soft_edge, 180, 0).astype(np.uint8)
        core = _build_foreground_core_mask(
            alpha_u8,
            known_background,
            probability,
            fringe,
            None,
            None,
            fixture.settings,
        )
        expected_core_count = int(np.count_nonzero(foreground_core))
        assert expected_core_count > 0, f"{fixture.name}: synthetic fixture should have a foreground core"
        found_core = int(np.count_nonzero(core & foreground_core))
        assert found_core >= int(expected_core_count * 0.95), (
            f"{fixture.name}: foreground-core helper missed too many core pixels {found_core}/{expected_core_count}"
        )
        assert not np.any(core & known_background), f"{fixture.name}: foreground-core helper included background"

        spill = _compute_key_spill_strength(fixture.rgb, fixture.settings.key_color)
        transition = _build_transition_repair_mask(
            alpha_u8,
            soft_edge,
            fringe,
            spill,
            known_background,
            None,
            None,
            core,
            fixture.settings,
        )
        if np.any(soft_edge):
            assert np.count_nonzero(transition & soft_edge) > 0, f"{fixture.name}: transition mask should catch soft edge pixels"
        assert not np.any(transition & known_background), f"{fixture.name}: transition mask included known background"
        protected_opaque = core & (alpha_u8 == 255) & (fringe == 0)
        assert not np.any(transition & protected_opaque), f"{fixture.name}: transition mask included protected opaque core"

    alpha = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 255, 128, 128, 255, 255],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    background = np.zeros(alpha.shape, dtype=bool)
    background[0, :] = True
    probability = np.zeros(alpha.shape, dtype=np.uint8)
    fringe = np.zeros(alpha.shape, dtype=np.uint8)
    fringe[1, 4] = 160
    edge = np.zeros(alpha.shape, dtype=bool)
    edge[1, 1:5] = True
    spill = np.zeros(alpha.shape, dtype=np.float32)
    spill[1, 1:5] = 0.50
    keep = np.zeros(alpha.shape, dtype=bool)
    remove = np.zeros(alpha.shape, dtype=bool)
    remove[1, 2] = True
    foreground_core = np.zeros(alpha.shape, dtype=bool)
    foreground_core[1, 1] = True
    foreground_core[1, 3] = True
    foreground_core[1, 4] = True
    mask = _build_transition_repair_mask(alpha, edge, fringe, spill, background, keep, remove, foreground_core, settings)
    assert not mask[0, 1], "transition mask must exclude known background"
    assert not mask[1, 1], "transition mask must exclude protected opaque core on edge/spill alone"
    assert not mask[1, 2], "transition mask must exclude manual remove pixels"
    assert mask[1, 3], "semi-transparent transition pixels remain eligible even when core-protected"
    assert mask[1, 4], "fringe pixels remain eligible even when core-protected"
    keep[1, 2] = True
    keep_core = _build_foreground_core_mask(alpha, background, probability, fringe, keep, remove, settings)
    keep_mask = _build_transition_repair_mask(alpha, edge, fringe, spill, background, keep, remove, keep_core, settings)
    assert keep_mask[1, 2], "keep mask should override manual remove for semi-transparent transition eligibility"


def run_gpu_runtime_probe_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert "torch" not in sys.modules, "smoke test start should not have torch imported"
    gpu_runtime = importlib.import_module("gpu_runtime")
    subprocess_utils = importlib.import_module("subprocess_utils")
    hidden_kwargs = subprocess_utils.hidden_subprocess_kwargs()
    if sys.platform == "win32":
        assert hidden_kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW, "probe subprocesses must use CREATE_NO_WINDOW on Windows"
        startupinfo = hidden_kwargs.get("startupinfo")
        assert startupinfo is not None, "probe subprocesses must provide hidden-window STARTUPINFO on Windows"
        assert startupinfo.dwFlags & subprocess.STARTF_USESHOWWINDOW, "probe subprocesses must request hidden startup windows"
        assert startupinfo.wShowWindow == subprocess.SW_HIDE, "probe subprocesses must set SW_HIDE on Windows"
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing gpu_runtime must not import heavy runtimes: {after_import - before}"

    fake_smi = {
        "available": True,
        "path": "C:/Windows/System32/nvidia-smi.exe",
        "error": None,
        "driver_version": "999.99",
        "cuda_version": "12.9",
        "gpus": [
            {
                "name": "NVIDIA Test GPU",
                "driver_version": "999.99",
                "memory_total_mib": 8192,
                "memory_free_mib": 4096,
                "compute_capability": "12.0",
            }
        ],
    }

    def missing_dll_probe(*, dll_path: str | None = None) -> dict[str, Any]:
        del dll_path
        return {
            "backend": "compact_cuda_dll",
            "backend_name": "compact CUDA DLL",
            "status": "unavailable",
            "available": False,
            "reason": "cuda_dll_unavailable",
            "message": "Compact CUDA DLL backend is unavailable: test missing DLL. CPU color path will be used.",
            "device": None,
            "device_index": None,
            "device_count": 0,
            "version": None,
            "dll_path": None,
            "load_error": "test missing DLL",
        }

    missing_probe = gpu_runtime.probe_gpu(gpu_accel_probe=missing_dll_probe, nvidia_smi_probe=lambda: fake_smi, run_kernel_smoke=False)
    backend_ids = [item["id"] for item in missing_probe["backend_registry"]["backends"]]
    assert "cuda_compat" in backend_ids
    assert "vulkan_compute" in backend_ids
    cuda_registry = next(item for item in missing_probe["backend_registry"]["backends"] if item["id"] == "cuda_compat")
    assert cuda_registry["legacy_backend"]["id"] == "compact_cuda_dll"
    vulkan_registry = next(item for item in missing_probe["backend_registry"]["backends"] if item["id"] == "vulkan_compute")
    assert vulkan_registry["available"] is False
    assert vulkan_registry["reason"] in {"vulkan_toolchain_incomplete", "vulkan_backend_deferred", "vulkan_runtime_unavailable", "vulkan_loader_unavailable"}
    assert "runtime_probe" in vulkan_registry and "toolchain" in vulkan_registry
    missing_selected = missing_probe["backend_registry"]["selected_backend"]
    if missing_selected["available"]:
        assert missing_probe["status"] == "available"
        assert missing_probe["available"] is True
        assert missing_selected["backend"] in {"d3d12_compute", "cuda_compat"}
    else:
        assert missing_probe["status"] == "unavailable"
        assert missing_probe["available"] is False
        assert missing_probe["reason"] == "cuda_dll_unavailable"
        assert missing_probe["backend"]["id"] == "compact_cuda_dll"
        assert missing_selected["status"] == "unavailable"
    assert missing_probe["cuda_dll"]["available"] is False
    assert missing_probe["cuda_dll"]["load_success"] is False
    assert missing_probe["cuda"]["is_available"] is False
    assert missing_probe["transition_repair_smoke"]["ran"] is False
    assert "compact cuda dll" in missing_probe["message"].lower()

    def available_dll_probe(*, dll_path: str | None = None) -> dict[str, Any]:
        del dll_path
        return {
            "backend": "compact_cuda_dll",
            "backend_name": "compact CUDA DLL",
            "status": "available",
            "available": True,
            "reason": None,
            "message": "Compact CUDA DLL backend available (1 CUDA device(s)).",
            "device": "CUDA device 0 (1 device(s) visible)",
            "device_index": 0,
            "device_count": 1,
            "version": 1,
            "dll_path": "C:/imgkey/imgkey_cuda.dll",
        }

    dll_probe = gpu_runtime.probe_gpu(gpu_accel_probe=available_dll_probe, nvidia_smi_probe=lambda: fake_smi, run_kernel_smoke=False)
    assert dll_probe["status"] == "available"
    assert dll_probe["available"] is True
    assert dll_probe["cuda_dll"]["version"] == 1
    assert dll_probe["cuda_dll"]["device_count"] == 1
    assert dll_probe["cuda"]["is_available"] is True
    assert dll_probe["cuda"]["device_name"] == "NVIDIA Test GPU"
    assert dll_probe["cuda"]["device_capability"] == [12, 0]
    assert dll_probe["backend_registry"]["selected_backend"]["backend"] in {"d3d12_compute", "cuda_compat"}
    assert dll_probe["backend_registry"]["selected_backend"]["status"] == "selected"
    assert "rgb_only" in dll_probe["backend_registry"]["selected_backend"]["capabilities"]
    assert dll_probe["transition_repair_smoke"]["ran"] is False

    for probe in (missing_probe, dll_probe):
        round_tripped = json.loads(json.dumps(probe))
        for key in ("schema_version", "status", "message", "backend", "backend_registry", "cuda_dll", "cuda", "nvidia_smi", "transition_repair_smoke", "native_toolchain", "vulkan_runtime"):
            assert key in round_tripped, f"gpu runtime probe JSON missing {key}"
        decision = round_tripped["native_toolchain"]["packaging_decision"]
        assert decision["status"] in {"approved", "pending_native_build", "blocked"}
        assert decision["primary_artifact"] == "dist/ImgKey.exe"
        assert decision["bundled_native_backend"] == "imgkey_gpu.dll"
        assert round_tripped["native_toolchain"]["components"]["vulkan"]["enabled"] is True
        assert round_tripped["native_toolchain"]["vulkan_gate"]["status"] in {"ready", "blocked"}
        assert round_tripped["vulkan_runtime"]["status"] in {"available", "unavailable"}
        assert "torch" not in round_tripped, "compact DLL runtime probe must not expose a torch probe section"

    after_probe = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_probe == before, f"compact DLL gpu probe tests must not import heavy runtimes: {after_probe - before}"


def _gpu_transition_tile(shape: tuple[int, int] = (96, 160)) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int, int],
    KeySettings,
]:
    h, w = shape
    key_color = (30, 80, 235)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = np.asarray(key_color, dtype=np.uint8)
    foreground = np.zeros((h, w, 3), dtype=np.uint8)
    foreground[:, :] = (226, 28, 20)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    y = np.linspace(-0.10, 0.10, h, dtype=np.float32).reshape(h, 1)
    alpha_f = np.clip((x + y - 0.10) / 0.78, 0.0, 1.0)
    alpha_u8 = np.rint(alpha_f * 255.0).astype(np.uint8)
    rgb = _composite_rgb_linear(background, foreground, alpha_f)
    background_mask = alpha_u8 == 0
    edge_mask = (alpha_u8 > 0) & (alpha_u8 < 255)
    probability = np.rint((1.0 - alpha_f) * 255.0).astype(np.uint8)
    fringe = np.where(edge_mask, 180, 0).astype(np.uint8)
    nearest_valid = alpha_u8 > 0
    settings = KeySettings(
        key_color=key_color,
        auto_border_sample=False,
        clip_foreground=0.0,
        transition_unmix=True,
        alpha_recover_strength=0.0,
        foreground_reference_pull=0.65,
        key_vector_despill=0.75,
        transition_reconstruction_error=0.08,
        gpu_acceleration="Off",
    )
    return rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings


def run_gpu_accel_backend_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert "torch" not in sys.modules, "gpu_accel backend tests start before torch is imported"
    gpu_accel = importlib.import_module("gpu_accel")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing gpu_accel must not import heavy runtimes: {after_import - before}"

    missing_dll = Path("native/imgkey_cuda/build/missing-imgkey-cuda.dll")
    unavailable = gpu_accel.is_available(dll_path=missing_dll, refresh=True)
    assert unavailable["available"] is False
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "cuda_dll_unavailable"
    assert unavailable["backend"] == "compact_cuda_dll"
    assert unavailable["backend_name"] == "compact CUDA DLL"

    rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings = _gpu_transition_tile((32, 48))
    transition_strength = gpu_accel.transition_repair_strength_mask_v1(
        rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings
    )
    foreground_valid_u8 = np.ascontiguousarray(nearest_valid.astype(np.uint8) * 255)

    invalid_calls = [
        ("rgb dtype", lambda: gpu_accel.transition_repair_dll_v1(rgb.astype(np.float32), alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings, dll_path=missing_dll)),
        ("rgb contiguity", lambda: gpu_accel.transition_repair_dll_v1(rgb[:, :, ::-1], alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings, dll_path=missing_dll)),
        ("alpha dimensionality", lambda: gpu_accel.transition_repair_dll_v1(rgb, alpha_u8[:, :, None], transition_strength, foreground, foreground_valid_u8, key_color, settings, dll_path=missing_dll)),
        ("mask shape", lambda: gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength[:-1], foreground, foreground_valid_u8[:-1], key_color, settings, dll_path=missing_dll)),
    ]
    for label, call in invalid_calls:
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:  # pragma: no cover - guard failure path
            raise AssertionError(f"CUDA DLL Python validation failed to reject {label}")

    try:
        gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings, dll_path=missing_dll)
    except gpu_accel.CudaDllUnavailable:
        pass
    else:  # pragma: no cover - guard failure path
        raise AssertionError("valid CUDA DLL call with a missing DLL path must fail before ctypes launch")

    radius_disabled = gpu_accel.process_color_tile_gpu(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        replace(settings, foreground_reference_radius=0),
        force_gpu=True,
    )
    assert radius_disabled["used"] is False
    assert radius_disabled["reason"] == "reference_radius_disabled"

    availability = gpu_accel.is_available(refresh=True)
    if availability.get("available"):
        cpu_rgb, cpu_mask = gpu_accel.transition_repair_cpu_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
        dll_rgb, dll_mask = gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
        max_rgb_diff = int(np.max(np.abs(cpu_rgb.astype(np.int16) - dll_rgb.astype(np.int16))))
        max_mask_diff = int(np.max(np.abs(cpu_mask.astype(np.int16) - dll_mask.astype(np.int16))))
        assert max_rgb_diff <= 2, f"CUDA DLL smoke RGB parity max diff too high: {max_rgb_diff}"
        assert max_mask_diff <= 2, f"CUDA DLL smoke mask parity max diff too high: {max_mask_diff}"
        backend = gpu_accel._load_cuda_dll()
        status = int(backend.library.imgkey_cuda_transition_repair_v1(None, None, None, None, None, None, None, None))
        assert status == gpu_accel.IMGKEY_CUDA_INVALID_ARGUMENT
        assert "params" in backend.last_error().lower()
        out_rgb = np.empty_like(rgb)
        out_mask = np.empty_like(alpha_u8)
        bad_version = gpu_accel._params_for_call(rgb, alpha_u8, transition_strength, out_rgb, key_color, settings)
        bad_version.version = 999
        try:
            backend.transition_repair(bad_version, rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, out_rgb, out_mask)
        except gpu_accel.CudaDllError as exc:
            assert exc.status == gpu_accel.IMGKEY_CUDA_UNSUPPORTED_VERSION
        else:  # pragma: no cover - guard failure path
            raise AssertionError("CUDA DLL ABI accepted an unsupported params version")
        bad_stride = gpu_accel._params_for_call(rgb, alpha_u8, transition_strength, out_rgb, key_color, settings)
        bad_stride.rgb_stride_bytes = 0
        try:
            backend.transition_repair(bad_stride, rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, out_rgb, out_mask)
        except gpu_accel.CudaDllError as exc:
            assert exc.status == gpu_accel.IMGKEY_CUDA_INVALID_ARGUMENT
        else:  # pragma: no cover - guard failure path
            raise AssertionError("CUDA DLL ABI accepted an invalid RGB stride")
    else:
        print(f"CUDA DLL smoke parity skipped: {availability.get('reason')} - {availability.get('message')}")

    preview_fallback = gpu_accel.process_preview_gpu()
    assert preview_fallback["reason"] == "not_implemented"
    assert preview_fallback["used"] is False

    after_tests = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_tests == before, f"fake gpu_accel tests must not import heavy runtimes: {after_tests - before}"


def run_gpu_backend_registry_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    gpu_backend = importlib.import_module("gpu_backend")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing gpu_backend must not import heavy runtimes: {after_import - before}"

    fake = gpu_backend.FakeNativeBackend(capabilities={"constant_screen", "rgb_only", "screen_tile"})
    probed = gpu_backend.probe_backends(backends=[fake], include_cpu=True)
    assert [item["id"] for item in probed] == ["fake_native", "cpu_fallback"]
    assert {"constant_screen", "rgb_only", "screen_tile"}.issubset(set(probed[0]["capabilities"]))
    selected = gpu_backend.select_backend("Auto", {"constant_screen", "rgb_only"}, backends=[fake], probed_backends=probed)
    assert selected.backend is fake and selected.status == "selected"
    assert selected.as_dict()["backend"] == "fake_native"
    off = gpu_backend.select_backend("Off", {"rgb_only"}, backends=[fake])
    assert off.status == "off" and off.reason == "gpu_off"
    constant_only = gpu_backend.FakeNativeBackend(capabilities={"constant_screen", "rgb_only"})
    no_screen_tile = gpu_backend.select_backend("Auto", {"screen_tile"}, backends=[constant_only])
    assert no_screen_tile.status == "unavailable" and no_screen_tile.backend is None

    rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings = _gpu_transition_tile((32, 48))
    session = gpu_backend.begin_render(
        settings,
        rgb.shape,
        mode="Auto",
        required_capabilities={"constant_screen", "rgb_only"},
        backends=[fake],
    )
    fake_result = session.process_color_tile(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        settings,
    )
    session.end_render()
    assert fake_result["used"] is True and fake_result["backend"] == "fake_native"
    assert np.array_equal(fake_result["rgb"][alpha_u8 > 0], rgb[alpha_u8 > 0])
    assert np.count_nonzero(fake_result["repair_mask"]) == 0

    missing_vulkan = gpu_backend.VulkanComputeBackend(
        toolchain_probe=lambda: {
            "components": {
                "vulkan": {
                    "enabled": True,
                    "available": False,
                    "headers": None,
                    "import_lib": None,
                    "loader_dll": None,
                    "message": "test Vulkan headers/import lib missing",
                }
            },
            "vulkan_gate": {
                "status": "blocked",
                "reason": "vulkan_toolchain_incomplete",
                "message": "test Vulkan headers/import lib missing",
            },
        },
        runtime_probe=lambda: {
            "schema_version": 1,
            "probe": "imgkey_vulkan_runtime",
            "status": "unavailable",
            "available": False,
            "reason": "vulkan_loader_unavailable",
            "message": "test Vulkan loader missing",
            "device_count": 0,
            "compute_device_count": 0,
            "devices": [],
        },
    )
    missing_vulkan_info = missing_vulkan.probe(refresh=True)
    assert missing_vulkan_info["id"] == "vulkan_compute"
    assert missing_vulkan_info["available"] is False
    assert missing_vulkan_info["status"] == "deferred"
    assert missing_vulkan_info["reason"] == "vulkan_toolchain_incomplete"
    assert missing_vulkan_info["runtime_probe"]["reason"] == "vulkan_loader_unavailable"
    vulkan_session = missing_vulkan.begin_render(settings, rgb.shape, force_gpu=True)
    vulkan_result = vulkan_session.process_full_color_tile(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        settings,
    )
    vulkan_session.end_render()
    assert vulkan_result["used"] is False and vulkan_result["reason"] == "vulkan_backend_deferred"

    d3d12 = gpu_backend.D3D12ComputeBackend()
    d3d12_info = d3d12.probe(refresh=True)
    if d3d12_info.get("available"):
        assert "screen_tile" in d3d12_info.get("capabilities", [])
        assert "full_color_tile" in d3d12_info.get("capabilities", [])
        identity_rgba = np.arange(17 * 19 * 4, dtype=np.uint8).reshape(17, 19, 4)
        identity = d3d12.identity_rgba(identity_rgba)
        assert identity["used"] is True, f"D3D12 identity expected use, got {identity.get('reason')}: {identity.get('message')}"
        assert np.array_equal(identity["rgba"], identity_rgba), "D3D12 identity kernel must be byte-exact"

        odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, odd_foreground, odd_valid, odd_key_color, odd_settings = _gpu_transition_tile((65, 97))
        d3d12_settings = replace(odd_settings, gpu_acceleration="Force GPU")
        cpu_rgb, cpu_mask = _repair_transition_unmix(odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, odd_key_color, None, odd_foreground, odd_valid, odd_settings)
        d3d12_session = gpu_backend.begin_render(d3d12_settings, odd_rgb.shape, required_capabilities={"constant_screen", "rgb_only"}, backends=[d3d12])
        d3d12_result = d3d12_session.process_color_tile(odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, None, odd_foreground, odd_valid, odd_key_color, d3d12_settings)
        d3d12_session.end_render()
        assert d3d12_result["used"] is True, f"D3D12 constant-screen expected use, got {d3d12_result.get('reason')}: {d3d12_result.get('message')}"
        d3d12_rgb = d3d12_result["rgb"]
        d3d12_mask = d3d12_result["repair_mask"]
        rgb_delta = np.abs(cpu_rgb.astype(np.int16) - d3d12_rgb.astype(np.int16))
        mask_delta = np.abs(cpu_mask.astype(np.int16) - d3d12_mask.astype(np.int16))
        assert int(rgb_delta.max()) <= 2 and float(np.percentile(rgb_delta, 99)) <= 1.0, "D3D12 constant-screen RGB parity exceeded tolerance"
        assert int(mask_delta.max()) == 0, "D3D12 constant-screen repair mask must match CPU exactly"
        assert int(d3d12_rgb[odd_alpha == 0].max(initial=0)) == 0, "D3D12 transparent RGB must remain zero"

        h, w = odd_alpha.shape
        screen_tile = np.empty_like(odd_rgb)
        screen_tile[:, :, 0] = odd_key_color[0]
        screen_tile[:, :, 1] = np.linspace(max(0, odd_key_color[1] - 10), min(255, odd_key_color[1] + 10), w, dtype=np.uint8).reshape(1, w)
        screen_tile[:, :, 2] = np.linspace(max(0, odd_key_color[2] - 15), odd_key_color[2], h, dtype=np.uint8).reshape(h, 1)
        cpu_screen_rgb, cpu_screen_mask = _repair_transition_unmix(odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, odd_key_color, screen_tile, odd_foreground, odd_valid, odd_settings)
        d3d12_screen_session = gpu_backend.begin_render(d3d12_settings, odd_rgb.shape, required_capabilities={"screen_tile", "rgb_only"}, backends=[d3d12])
        d3d12_screen = d3d12_screen_session.process_color_tile(odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, screen_tile, odd_foreground, odd_valid, odd_key_color, d3d12_settings)
        d3d12_screen_session.end_render()
        assert d3d12_screen["used"] is True, f"D3D12 screen_tile expected use, got {d3d12_screen.get('reason')}: {d3d12_screen.get('message')}"
        screen_rgb_delta = np.abs(cpu_screen_rgb.astype(np.int16) - d3d12_screen["rgb"].astype(np.int16))
        screen_mask_delta = np.abs(cpu_screen_mask.astype(np.int16) - d3d12_screen["repair_mask"].astype(np.int16))
        assert int(screen_rgb_delta.max()) <= 2 and float(np.percentile(screen_rgb_delta, 99)) <= 1.0, "D3D12 screen_tile RGB parity exceeded tolerance"
        assert int(screen_mask_delta.max()) == 0, "D3D12 screen_tile repair mask must match CPU exactly"

        cpu_full_rgb, cpu_full_mask = _process_color_tile(odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, screen_tile, odd_foreground, odd_valid, odd_key_color, replace(odd_settings, gpu_acceleration="Off"))
        d3d12_full_session = gpu_backend.begin_render(d3d12_settings, odd_rgb.shape, required_capabilities={"screen_tile", "rgb_only", "full_color_tile"}, backends=[d3d12])
        d3d12_full = d3d12_full_session.process_full_color_tile(odd_rgb, odd_alpha, odd_background, odd_edge, odd_probability, odd_fringe, screen_tile, odd_foreground, odd_valid, odd_key_color, d3d12_settings)
        d3d12_full_session.end_render()
        assert d3d12_full["used"] is True, f"D3D12 full color tile expected use, got {d3d12_full.get('reason')}: {d3d12_full.get('message')}"
        full_rgb_delta = np.abs(cpu_full_rgb.astype(np.int16) - d3d12_full["rgb"].astype(np.int16))
        full_mask_delta = np.abs(cpu_full_mask.astype(np.int16) - d3d12_full["repair_mask"].astype(np.int16))
        assert int(full_rgb_delta.max()) <= 2 and float(np.percentile(full_rgb_delta, 99)) <= 1.0, "D3D12 full color tile RGB parity exceeded tolerance"
        assert int(full_mask_delta.max()) <= 1, "D3D12 full color tile spill/repair mask parity exceeded tolerance"
        assert int(d3d12_full["rgb"][odd_alpha == 0].max(initial=0)) == 0, "D3D12 full color tile transparent RGB must remain zero"

        split_rgb, split_alpha, split_background, split_edge, split_probability, split_fringe, split_foreground, split_valid, split_key, split_settings = _gpu_transition_tile((513, 513))
        split_cpu_rgb, split_cpu_mask = _process_color_tile(split_rgb, split_alpha, split_background, split_edge, split_probability, split_fringe, None, split_foreground, split_valid, split_key, replace(split_settings, gpu_acceleration="Off"))
        split_gpu_settings = replace(split_settings, gpu_acceleration="Force GPU")
        split_session = gpu_backend.begin_render(split_gpu_settings, split_rgb.shape, required_capabilities={"constant_screen", "rgb_only", "full_color_tile"}, backends=[d3d12])
        split_gpu = split_session.process_full_color_tile(split_rgb, split_alpha, split_background, split_edge, split_probability, split_fringe, None, split_foreground, split_valid, split_key, split_gpu_settings)
        split_session.end_render()
        assert split_gpu["used"] is True and int(split_gpu.get("subtile_dispatches") or 0) > 1, "D3D12 full color tile must split larger tiles into persistent TDR-bounded subdispatches"
        split_rgb_delta = np.abs(split_cpu_rgb.astype(np.int16) - split_gpu["rgb"].astype(np.int16))
        split_mask_delta = np.abs(split_cpu_mask.astype(np.int16) - split_gpu["repair_mask"].astype(np.int16))
        assert int(split_rgb_delta.max()) <= 2 and float(np.percentile(split_rgb_delta, 99)) <= 1.0, "D3D12 split full color tile RGB parity exceeded tolerance"
        assert int(split_mask_delta.max()) <= 1, "D3D12 split full color tile mask parity exceeded tolerance"
    else:
        print(f"D3D12 native backend tests skipped: {d3d12_info.get('reason')} - {d3d12_info.get('message')}")

    call = gpu_backend.validate_native_color_tile_inputs(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        settings,
    )
    assert call.params.struct_size == ctypes.sizeof(gpu_backend.ImgKeyNativeColorTileParamsV1)
    assert call.buffers["rgb"].row_stride_bytes == rgb.strides[0]
    fake_abi = gpu_backend.FakeNativeCAbi()
    assert fake_abi.process_color_tile_v1(call.params, rgb=call.buffers["rgb"], alpha=call.buffers["alpha"]) == gpu_backend.IMGKEY_GPU_OK

    invalid_calls = [
        ("bad rgb dtype", lambda: gpu_backend.validate_native_color_tile_inputs(rgb.astype(np.float32), alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, settings)),
        ("bad rgb stride", lambda: gpu_backend.validate_native_color_tile_inputs(rgb[:, ::-1, :], alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground[:, ::-1, :], nearest_valid[:, ::-1], key_color, settings)),
        ("bad mask shape", lambda: gpu_backend.validate_native_color_tile_inputs(rgb, alpha_u8[:-1], background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, settings)),
        ("null foreground", lambda: gpu_backend.validate_native_color_tile_inputs(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, None, nearest_valid, key_color, settings)),
    ]
    for label, call_invalid in invalid_calls:
        try:
            call_invalid()
        except gpu_backend.NativeAbiError:
            pass
        else:  # pragma: no cover - guard failure path
            raise AssertionError(f"native ABI validation failed to reject {label}")

    bad_version = gpu_backend.validate_native_color_tile_inputs(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, settings)
    bad_version.params.version = 999
    assert fake_abi.process_color_tile_v1(bad_version.params, rgb=bad_version.buffers["rgb"]) == gpu_backend.IMGKEY_GPU_UNSUPPORTED_VERSION
    assert "version" in gpu_backend.native_last_error().lower()
    bad_stride = gpu_backend.validate_native_color_tile_inputs(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, settings)
    bad_stride.buffers["rgb"].row_stride_bytes = 0
    assert fake_abi.process_color_tile_v1(bad_stride.params, rgb=bad_stride.buffers["rgb"]) == gpu_backend.IMGKEY_GPU_INVALID_ARGUMENT
    assert "stride" in gpu_backend.native_last_error().lower()
    null_data = gpu_backend.validate_native_color_tile_inputs(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, settings)
    null_data.buffers["rgb"].data = None
    assert fake_abi.process_color_tile_v1(null_data.params, rgb=null_data.buffers["rgb"]) == gpu_backend.IMGKEY_GPU_INVALID_ARGUMENT
    assert "null" in gpu_backend.native_last_error().lower()

    after_tests = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_tests == before, f"gpu_backend registry/ABI tests must not import heavy runtimes: {after_tests - before}"


def run_gpu_parity_tests() -> None:
    gpu_accel = importlib.import_module("gpu_accel")
    gpu_backend = importlib.import_module("gpu_backend")
    availability = gpu_accel.is_available(refresh=True)
    backend_objects = gpu_backend.registered_backends()
    selected = gpu_backend.select_backend("Force GPU", {"constant_screen", "rgb_only"}, backends=backend_objects, refresh=True)
    if not availability.get("available") and not selected.available:
        print(f"gpu parity skipped: {availability.get('reason')} - {availability.get('message')}")
        return

    rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings = _gpu_transition_tile((128, 192))
    transition_strength = gpu_accel.transition_repair_strength_mask_v1(
        rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings
    )
    foreground_valid_u8 = np.ascontiguousarray(nearest_valid.astype(np.uint8) * 255)
    cpu_rgb, cpu_mask = gpu_accel.transition_repair_cpu_v1(
        rgb,
        alpha_u8,
        transition_strength,
        foreground,
        foreground_valid_u8,
        key_color,
        settings,
    )
    max_rgb_diff = max_mask_diff = None
    if availability.get("available"):
        gpu = gpu_accel.process_color_tile_gpu(
            rgb,
            alpha_u8,
            background_mask,
            edge_mask,
            probability,
            fringe,
            None,
            foreground,
            nearest_valid,
            key_color,
            replace(settings, gpu_acceleration="Force GPU"),
            force_gpu=True,
        )
        assert gpu["used"] is True, f"GPU parity expected backend use, got {gpu.get('reason')}: {gpu.get('message')}"
        gpu_rgb = gpu["rgb"]
        gpu_mask = gpu["repair_mask"]
        assert gpu_rgb is not None and gpu_mask is not None
        max_rgb_diff = int(np.max(np.abs(cpu_rgb.astype(np.int16) - gpu_rgb.astype(np.int16))))
        max_mask_diff = int(np.max(np.abs(cpu_mask.astype(np.int16) - gpu_mask.astype(np.int16))))
        assert max_rgb_diff <= 2, f"GPU transition RGB parity max diff too high: {max_rgb_diff}"
        assert max_mask_diff <= 2, f"GPU transition mask parity max diff too high: {max_mask_diff}"

    backend_session = gpu_backend.begin_render(replace(settings, gpu_acceleration="Force GPU"), rgb.shape, required_capabilities={"constant_screen", "rgb_only"})
    backend_gpu = backend_session.process_color_tile(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        replace(settings, gpu_acceleration="Force GPU"),
    )
    backend_session.end_render()
    assert backend_gpu["used"] is True, f"backend registry expected use, got {backend_gpu.get('reason')}: {backend_gpu.get('message')}"
    backend_rgb = backend_gpu["rgb"]
    backend_mask = backend_gpu["repair_mask"]
    assert backend_rgb is not None and backend_mask is not None
    max_backend_rgb_diff = int(np.max(np.abs(cpu_rgb.astype(np.int16) - backend_rgb.astype(np.int16))))
    max_backend_mask_diff = int(np.max(np.abs(cpu_mask.astype(np.int16) - backend_mask.astype(np.int16))))
    assert max_backend_rgb_diff <= 2, f"backend registry RGB parity max diff too high: {max_backend_rgb_diff}"
    assert max_backend_mask_diff <= 2, f"backend registry mask parity max diff too high: {max_backend_mask_diff}"

    max_direct_rgb_diff = max_direct_mask_diff = None
    if availability.get("available"):
        dll_rgb, dll_mask = gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
        max_direct_rgb_diff = int(np.max(np.abs(cpu_rgb.astype(np.int16) - dll_rgb.astype(np.int16))))
        max_direct_mask_diff = int(np.max(np.abs(cpu_mask.astype(np.int16) - dll_mask.astype(np.int16))))
        assert max_direct_rgb_diff <= 2, f"direct CUDA DLL RGB parity max diff too high: {max_direct_rgb_diff}"
        assert max_direct_mask_diff <= 2, f"direct CUDA DLL mask parity max diff too high: {max_direct_mask_diff}"

    disabled_settings = replace(settings, gpu_acceleration="Force GPU", foreground_reference_radius=0)
    disabled = gpu_accel.process_color_tile_gpu(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        disabled_settings,
        force_gpu=True,
    )
    assert disabled["used"] is False and disabled["reason"] == "reference_radius_disabled"
    cpu_disabled, _ = _process_color_tile(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        replace(disabled_settings, gpu_acceleration="Off"),
    )
    gpu_disabled, _ = _process_color_tile(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        None,
        foreground,
        nearest_valid,
        key_color,
        disabled_settings,
    )
    assert np.array_equal(cpu_disabled, gpu_disabled), "Force GPU must honor disabled transition reference radius"

    opaque_alpha = np.full_like(alpha_u8, 255)
    opaque_background = np.zeros_like(background_mask, dtype=bool)
    opaque_edge = np.zeros_like(edge_mask, dtype=bool)
    opaque_probability = np.zeros_like(probability, dtype=np.uint8)
    opaque_fringe = np.zeros_like(fringe, dtype=np.uint8)
    if availability.get("available"):
        opaque_noop = gpu_accel.process_color_tile_gpu(
            foreground,
            opaque_alpha,
            opaque_background,
            opaque_edge,
            opaque_probability,
            opaque_fringe,
            None,
            foreground,
            np.ones_like(nearest_valid, dtype=bool),
            key_color,
            replace(settings, gpu_acceleration="Force GPU"),
            force_gpu=True,
        )
        assert opaque_noop["used"] is False and opaque_noop["reason"] == "no_eligible_pixels"
    print(
        "gpu parity ok "
        f"cuda_device={availability.get('device')} backend={backend_gpu.get('backend')} max_rgb_diff={max_rgb_diff} "
        f"max_mask_diff={max_mask_diff} max_direct_rgb_diff={max_direct_rgb_diff} "
        f"max_backend_rgb_diff={max_backend_rgb_diff}"
    )


def _median_time_ms(callback, *, repeat: int = 5, warmup: int = 1) -> float:
    for _ in range(max(0, warmup)):
        callback()
    samples: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        callback()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def write_gpu_benchmarks() -> None:
    gpu_accel = importlib.import_module("gpu_accel")
    gpu_backend = importlib.import_module("gpu_backend")
    availability = gpu_accel.is_available(refresh=True)
    d3d12 = gpu_backend.D3D12ComputeBackend()
    d3d12_info = d3d12.probe(refresh=True)
    vulkan_info = gpu_backend.VulkanComputeBackend().probe(refresh=True)
    GPU_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "generated_by": "python smoke_test.py --gpu-benchmark",
        "backend": {
            "id": "backend_registry",
            "name": "ImgKey GPU backend registry",
        },
        "availability": {
            "cuda_compat": availability,
            "d3d12_compute": d3d12_info,
            "vulkan_compute": vulkan_info,
        },
        "notes": [
            "CPU remains the correctness reference.",
            "CUDA DLL timings include host/device copies performed inside imgkey_cuda_transition_repair_v1.",
            "D3D12 timings use the persistent D3D12 context. Full-color tiles larger than the native-call safety budget are split into TDR-bounded persistent-buffer subdispatches.",
            "Vulkan is optional/experimental. If the SDK headers/import lib or installed loader/device are missing, the benchmark records a clean deferred/unavailable probe and does not run Vulkan timings.",
        ],
        "benchmarks": {},
    }
    if not availability.get("available") and not d3d12_info.get("available"):
        (GPU_BENCHMARK_DIR / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"gpu benchmark skipped: cuda={availability.get('reason')} d3d12={d3d12_info.get('reason')}")
        return

    rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings = _gpu_transition_tile((1024, 1024))
    settings_gpu = replace(settings, gpu_acceleration="Force GPU")
    transition_strength = gpu_accel.transition_repair_strength_mask_v1(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        foreground,
        nearest_valid,
        key_color,
        settings,
    )
    foreground_valid_u8 = np.ascontiguousarray(nearest_valid.astype(np.uint8) * 255)

    def cpu_transition_reference() -> None:
        gpu_accel.transition_repair_cpu_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)

    def cuda_dll_transition_direct() -> None:
        gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)

    cpu_direct_ms = _median_time_ms(cpu_transition_reference, repeat=5, warmup=1)
    cpu_rgb, cpu_mask = gpu_accel.transition_repair_cpu_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
    if availability.get("available"):
        cuda_direct_ms = _median_time_ms(cuda_dll_transition_direct, repeat=5, warmup=2)
        dll_rgb, dll_mask = gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
        direct_max_rgb_diff = int(np.max(np.abs(cpu_rgb.astype(np.int16) - dll_rgb.astype(np.int16))))
        direct_max_mask_diff = int(np.max(np.abs(cpu_mask.astype(np.int16) - dll_mask.astype(np.int16))))
        summary["benchmarks"]["cuda_transition_repair_tile_1024_direct"] = {
            "tile_shape": list(rgb.shape[:2]),
            "active_repair_pixels": int(np.count_nonzero(transition_strength)),
            "cpu_reference_ms": cpu_direct_ms,
            "cuda_dll_ms_including_transfer": cuda_direct_ms,
            "speedup": cpu_direct_ms / cuda_direct_ms if cuda_direct_ms > 0 else None,
            "transfer_included": True,
            "transfer_scope": "imgkey_cuda_transition_repair_v1 copies uint8 host inputs to device and RGB/mask outputs back to host before returning.",
            "max_rgb_diff_vs_cpu": direct_max_rgb_diff,
            "max_mask_diff_vs_cpu": direct_max_mask_diff,
            "faster_than_cpu": cuda_direct_ms < cpu_direct_ms,
        }

    def cpu_transition_dispatch() -> None:
        _repair_transition_unmix(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, key_color, None, foreground, nearest_valid, settings)

    def cuda_dll_transition_dispatch() -> None:
        result = gpu_accel.process_color_tile_gpu(
            rgb,
            alpha_u8,
            background_mask,
            edge_mask,
            probability,
            fringe,
            None,
            foreground,
            nearest_valid,
            key_color,
            settings_gpu,
            force_gpu=True,
        )
        assert result["used"], result.get("message")

    cpu_dispatch_ms = _median_time_ms(cpu_transition_dispatch, repeat=5, warmup=1)
    cpu_dispatch_rgb, _ = _repair_transition_unmix(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, key_color, None, foreground, nearest_valid, settings)
    if availability.get("available"):
        cuda_dispatch_ms = _median_time_ms(cuda_dll_transition_dispatch, repeat=5, warmup=2)
        gpu_result = gpu_accel.process_color_tile_gpu(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, settings_gpu, force_gpu=True)
        dispatch_max_rgb_diff = int(np.max(np.abs(cpu_dispatch_rgb.astype(np.int16) - gpu_result["rgb"].astype(np.int16))))
        summary["benchmarks"]["cuda_transition_repair_tile_1024_dispatch"] = {
            "tile_shape": list(rgb.shape[:2]),
            "cpu_full_transition_ms": cpu_dispatch_ms,
            "cuda_dll_dispatch_ms_including_transfer": cuda_dispatch_ms,
            "speedup": cpu_dispatch_ms / cuda_dispatch_ms if cuda_dispatch_ms > 0 else None,
            "transfer_included": True,
            "max_rgb_diff_vs_cpu": dispatch_max_rgb_diff,
            "faster_than_cpu": cuda_dispatch_ms < cpu_dispatch_ms,
            "auto_fallback_policy": "GPU remains optional; Auto uses CPU fallback on missing DLL, no device, unsupported tile, or execution error.",
        }

    if d3d12_info.get("available"):
        d3d12_rgb, d3d12_alpha, d3d12_bg, d3d12_edge, d3d12_probability, d3d12_fringe, d3d12_foreground, d3d12_valid, d3d12_key, d3d12_settings = _gpu_transition_tile((512, 512))
        d3d12_settings_gpu = replace(d3d12_settings, gpu_acceleration="Force GPU")

        def d3d12_cpu_transition_dispatch() -> None:
            _repair_transition_unmix(d3d12_rgb, d3d12_alpha, d3d12_bg, d3d12_edge, d3d12_probability, d3d12_fringe, d3d12_key, None, d3d12_foreground, d3d12_valid, d3d12_settings)

        d3d12_cpu_ms = _median_time_ms(d3d12_cpu_transition_dispatch, repeat=5, warmup=1)
        d3d12_cpu_rgb, d3d12_cpu_mask = _repair_transition_unmix(d3d12_rgb, d3d12_alpha, d3d12_bg, d3d12_edge, d3d12_probability, d3d12_fringe, d3d12_key, None, d3d12_foreground, d3d12_valid, d3d12_settings)
        d3d12_session = gpu_backend.begin_render(d3d12_settings_gpu, d3d12_rgb.shape, required_capabilities={"constant_screen", "rgb_only"}, backends=[d3d12])

        def d3d12_transition_dispatch() -> None:
            result = d3d12_session.process_color_tile(d3d12_rgb, d3d12_alpha, d3d12_bg, d3d12_edge, d3d12_probability, d3d12_fringe, None, d3d12_foreground, d3d12_valid, d3d12_key, d3d12_settings_gpu)
            assert result["used"], result.get("message")

        try:
            d3d12_dispatch_ms = _median_time_ms(d3d12_transition_dispatch, repeat=5, warmup=2)
            d3d12_result = d3d12_session.process_color_tile(d3d12_rgb, d3d12_alpha, d3d12_bg, d3d12_edge, d3d12_probability, d3d12_fringe, None, d3d12_foreground, d3d12_valid, d3d12_key, d3d12_settings_gpu)
        finally:
            d3d12_session.end_render()
        d3d12_rgb_delta = np.abs(d3d12_cpu_rgb.astype(np.int16) - d3d12_result["rgb"].astype(np.int16))
        d3d12_mask_delta = np.abs(d3d12_cpu_mask.astype(np.int16) - d3d12_result["repair_mask"].astype(np.int16))
        summary["benchmarks"]["d3d12_transition_repair_tile_512_dispatch"] = {
            "tile_shape": list(d3d12_rgb.shape[:2]),
            "cpu_full_transition_ms": d3d12_cpu_ms,
            "d3d12_dispatch_ms_including_transfer": d3d12_dispatch_ms,
            "speedup": d3d12_cpu_ms / d3d12_dispatch_ms if d3d12_dispatch_ms > 0 else None,
            "transfer_included": True,
            "max_rgb_diff_vs_cpu": int(d3d12_rgb_delta.max()),
            "p99_rgb_diff_vs_cpu": float(np.percentile(d3d12_rgb_delta, 99)),
            "max_mask_diff_vs_cpu": int(d3d12_mask_delta.max()),
            "faster_than_cpu": d3d12_dispatch_ms < d3d12_cpu_ms,
            "auto_fallback_policy": "Backend registry selects D3D12 for constant-screen and screen_tile tiles when available; CPU remains fallback.",
        }

        full_rgb, full_alpha, full_bg, full_edge, full_probability, full_fringe, full_foreground, full_valid, full_key, full_settings = _gpu_transition_tile((2048, 2048))
        full_settings_gpu = replace(full_settings, gpu_acceleration="Force GPU")

        def d3d12_cpu_full_color_dispatch() -> None:
            _process_color_tile(full_rgb, full_alpha, full_bg, full_edge, full_probability, full_fringe, None, full_foreground, full_valid, full_key, replace(full_settings, gpu_acceleration="Off"))

        d3d12_full_cpu_ms = _median_time_ms(d3d12_cpu_full_color_dispatch, repeat=3, warmup=1)
        d3d12_full_cpu_rgb, d3d12_full_cpu_mask = _process_color_tile(full_rgb, full_alpha, full_bg, full_edge, full_probability, full_fringe, None, full_foreground, full_valid, full_key, replace(full_settings, gpu_acceleration="Off"))
        d3d12_full_session = gpu_backend.begin_render(full_settings_gpu, full_rgb.shape, required_capabilities={"constant_screen", "rgb_only", "full_color_tile"}, backends=[d3d12])

        def d3d12_full_color_dispatch() -> None:
            result = d3d12_full_session.process_full_color_tile(full_rgb, full_alpha, full_bg, full_edge, full_probability, full_fringe, None, full_foreground, full_valid, full_key, full_settings_gpu)
            assert result["used"], result.get("message")

        try:
            d3d12_full_dispatch_ms = _median_time_ms(d3d12_full_color_dispatch, repeat=5, warmup=2)
            d3d12_full_result = d3d12_full_session.process_full_color_tile(full_rgb, full_alpha, full_bg, full_edge, full_probability, full_fringe, None, full_foreground, full_valid, full_key, full_settings_gpu)
        finally:
            d3d12_full_session.end_render()
        full_rgb_delta = np.abs(d3d12_full_cpu_rgb.astype(np.int16) - d3d12_full_result["rgb"].astype(np.int16))
        full_mask_delta = np.abs(d3d12_full_cpu_mask.astype(np.int16) - d3d12_full_result["repair_mask"].astype(np.int16))
        summary["benchmarks"]["d3d12_full_color_tile_2048_dispatch"] = {
            "tile_shape": list(full_rgb.shape[:2]),
            "cpu_full_color_ms": d3d12_full_cpu_ms,
            "d3d12_dispatch_ms_including_transfer": d3d12_full_dispatch_ms,
            "speedup": d3d12_full_cpu_ms / d3d12_full_dispatch_ms if d3d12_full_dispatch_ms > 0 else None,
            "transfer_included": True,
            "subtile_dispatches": int(d3d12_full_result.get("subtile_dispatches") or 1),
            "subtile_max_pixels": int(d3d12_full_result.get("subtile_max_pixels") or 0),
            "max_rgb_diff_vs_cpu": int(full_rgb_delta.max()),
            "p99_rgb_diff_vs_cpu": float(np.percentile(full_rgb_delta, 99)),
            "max_mask_diff_vs_cpu": int(full_mask_delta.max()),
            "faster_than_cpu": d3d12_full_dispatch_ms < d3d12_full_cpu_ms,
            "auto_fallback_policy": "Full color D3D12 path is optional and falls back by capability/error; CPU remains correctness reference.",
        }

    (GPU_BENCHMARK_DIR / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"gpu benchmark summary written to {GPU_BENCHMARK_DIR / 'summary.json'}")


def _perf_ms(value: float) -> float:
    return float(round(float(value), 3))


def _record_perf_timing(report: dict[str, Any], stage: str, elapsed_ms: float) -> None:
    timings = report.setdefault("timings", {})
    record = timings.setdefault(stage, {"calls": 0, "total_ms": 0.0, "samples_ms": []})
    record["calls"] = int(record["calls"]) + 1
    record["total_ms"] = float(record["total_ms"]) + float(elapsed_ms)
    samples = record.setdefault("samples_ms", [])
    if isinstance(samples, list):
        samples.append(float(elapsed_ms))


def _finalize_perf_timings(report: dict[str, Any]) -> None:
    for record in report.get("timings", {}).values():
        samples = np.asarray(record.get("samples_ms", []), dtype=np.float64)
        if samples.size:
            record["total_ms"] = _perf_ms(float(np.sum(samples)))
            record["median_ms"] = _perf_ms(float(np.median(samples)))
            record["max_ms"] = _perf_ms(float(np.max(samples)))
            record["min_ms"] = _perf_ms(float(np.min(samples)))
            record["mean_ms"] = _perf_ms(float(np.mean(samples)))
            record["samples_ms"] = [_perf_ms(float(sample)) for sample in samples]
        else:
            record["total_ms"] = _perf_ms(float(record.get("total_ms", 0.0)))
            record["median_ms"] = 0.0
            record["max_ms"] = 0.0
            record["min_ms"] = 0.0
            record["mean_ms"] = 0.0


def _record_perf_tile(report: dict[str, Any], args: tuple[Any, ...]) -> None:
    if not args:
        return
    rgb_tile = np.asarray(args[0])
    if rgb_tile.ndim < 2:
        return
    h, w = rgb_tile.shape[:2]
    tiles = report.setdefault("tiles", {"count": 0, "read_shapes": {}})
    tiles["count"] = int(tiles.get("count", 0)) + 1
    shape_key = f"{int(w)}x{int(h)}"
    read_shapes = tiles.setdefault("read_shapes", {})
    read_shapes[shape_key] = int(read_shapes.get(shape_key, 0)) + 1


def _record_perf_gpu_result(report: dict[str, Any], result: Any, wall_ms: float) -> None:
    gpu = report.setdefault(
        "gpu_tile_dispatch",
        {
            "calls": 0,
            "used_tiles": 0,
            "fallback_tiles": 0,
            "reported_elapsed_ms_total": 0.0,
            "wall_ms_total": 0.0,
            "reasons": {},
        },
    )
    gpu["calls"] = int(gpu.get("calls", 0)) + 1
    gpu["wall_ms_total"] = float(gpu.get("wall_ms_total", 0.0)) + float(wall_ms)
    if not isinstance(result, dict):
        reasons = gpu.setdefault("reasons", {})
        reasons["non_dict_result"] = int(reasons.get("non_dict_result", 0)) + 1
        return
    elapsed = result.get("elapsed_ms")
    if elapsed is not None:
        gpu["reported_elapsed_ms_total"] = float(gpu.get("reported_elapsed_ms_total", 0.0)) + float(elapsed)
    used = bool(result.get("used"))
    if used:
        gpu["used_tiles"] = int(gpu.get("used_tiles", 0)) + 1
    else:
        gpu["fallback_tiles"] = int(gpu.get("fallback_tiles", 0)) + 1
    reason = str(result.get("reason") or ("used" if used else "unknown"))
    reasons = gpu.setdefault("reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


@contextmanager
def _pipeline_perf_instrumentation(report: dict[str, Any]):
    originals: list[tuple[Any, str, Any]] = []

    def patch(target: Any, name: str, stage: str, after: Any | None = None) -> None:
        if not hasattr(target, name):
            return
        original = getattr(target, name)

        def wrapped(*args: Any, **kwargs: Any):
            start = time.perf_counter()
            ok = False
            result: Any = None
            try:
                result = original(*args, **kwargs)
                ok = True
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                _record_perf_timing(report, stage, elapsed_ms)
                if ok and after is not None:
                    after(report, result, elapsed_ms, args, kwargs)

        setattr(target, name, wrapped)
        originals.append((target, name, original))

    def after_tile(case_report: dict[str, Any], _result: Any, _elapsed_ms: float, args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        _record_perf_tile(case_report, args)

    def after_gpu(case_report: dict[str, Any], result: Any, elapsed_ms: float, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        _record_perf_gpu_result(case_report, result, elapsed_ms)

    patch(keyer_module, "_build_global_matte", "global_matte_total")
    patch(keyer_module, "_estimate_screen_map", "screen_model_full_image")
    patch(keyer_module, "_estimate_screen_tile", "screen_model_local_plate_tile")
    patch(keyer_module, "_build_nearest_inner_reference_map", "nearest_inner_reference_global")
    patch(keyer_module, "_build_tile_local_nearest_inner_rgb", "nearest_inner_reference_tile_local")
    patch(keyer_module, "_recover_transition_alpha_global", "transition_alpha_recovery")
    patch(keyer_module, "_render_tiled_rgba", "tiled_rgba_render_total")
    patch(keyer_module, "_process_color_tile", "per_tile_color_render", after_tile)
    try:
        gpu_accel = importlib.import_module("gpu_accel")
        patch(gpu_accel, "process_color_tile_gpu", "gpu_transfer_dispatch_readback", after_gpu)
    except Exception as exc:  # pragma: no cover - diagnostic-only guard
        report["gpu_instrumentation_error"] = f"{type(exc).__name__}: {exc}"
    try:
        yield
    finally:
        for target, name, original in reversed(originals):
            setattr(target, name, original)


def _resampling_bilinear() -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "BILINEAR", Image.BILINEAR)


def _large_perf_source_from_case(case: GeometricBenchmarkCase, size: int) -> np.ndarray:
    image = Image.fromarray(case.source_rgb, mode="RGB")
    resized = image.resize((int(size), int(size)), resample=_resampling_bilinear())
    return np.asarray(resized, dtype=np.uint8).copy()


def _large_perf_case_specs() -> list[tuple[str, GeometricBenchmarkCase, int]]:
    cases = {case.background_name: case for case in geometric_benchmark_cases()}
    return [
        ("large_geometric_4096_blue_flat", cases["blue_flat"], 4096),
        ("large_geometric_8192_blue_gradient", cases["blue_uneven_gradient"], 8192),
    ]


def _profile_large_pipeline_case(name: str, template_case: GeometricBenchmarkCase, size: int) -> dict[str, Any]:
    source_dir = PERF_BASELINE_DIR / "sources"
    export_dir = PERF_BASELINE_DIR / "exports"
    source_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / f"{name}_source.png"
    export_path = export_dir / f"{name}_result.png"
    settings = replace(template_case.settings, use_tiling=True, tile_size=2048, tile_overlap=128, gpu_acceleration="Off")
    report: dict[str, Any] = {
        "name": name,
        "shape": [int(size), int(size), 3],
        "background_name": template_case.background_name,
        "key_color": list(template_case.key_color),
        "notes": "Source is the current geometric benchmark case resized to the target square dimension; keyer settings keep tiled export enabled and GPU off for CPU-reference baseline.",
        "settings": asdict(settings),
        "source_path": str(source_path),
        "export_path": str(export_path),
        "timings": {},
    }

    start = time.perf_counter()
    source_rgb = _large_perf_source_from_case(template_case, size)
    _record_perf_timing(report, "fixture_generation", (time.perf_counter() - start) * 1000.0)
    report["source_mean_rgb"] = [float(v) for v in np.mean(source_rgb.reshape(-1, 3), axis=0)]

    start = time.perf_counter()
    _save_rgb(source_path, source_rgb)
    _record_perf_timing(report, "source_png_write", (time.perf_counter() - start) * 1000.0)
    report["source_png_bytes"] = int(source_path.stat().st_size)
    del source_rgb
    gc.collect()

    start = time.perf_counter()
    rgb, original_alpha = keyer_module.read_image_rgb(source_path)
    _record_perf_timing(report, "image_load", (time.perf_counter() - start) * 1000.0)
    report["original_alpha_present"] = original_alpha is not None

    start = time.perf_counter()
    preview_rgb, preview_scale = keyer_module.resize_for_preview(rgb)
    _record_perf_timing(report, "preview_resize", (time.perf_counter() - start) * 1000.0)
    report["preview"] = {"shape": list(preview_rgb.shape), "scale": float(preview_scale)}
    del preview_rgb
    gc.collect()

    with _pipeline_perf_instrumentation(report):
        start = time.perf_counter()
        result = process_key_image(rgb, settings, original_alpha=original_alpha, include_debug=False)
        _record_perf_timing(report, "process_key_image_total", (time.perf_counter() - start) * 1000.0)

    report["result"] = {
        "rgba_shape": list(result.rgba.shape),
        "alpha_nonzero_pixels": int(np.count_nonzero(result.alpha)),
        "alpha_mean": float(np.mean(result.alpha)),
        "transparent_rgb_zero": transparent_rgb_zero(result.rgba),
        "gpu_acceleration": result.gpu_acceleration,
    }
    start = time.perf_counter()
    _save_rgba(export_path, result.rgba)
    _record_perf_timing(report, "png_encode_export", (time.perf_counter() - start) * 1000.0)
    report["export_png_bytes"] = int(export_path.stat().st_size)
    del result
    del rgb
    gc.collect()
    _finalize_perf_timings(report)
    if "gpu_tile_dispatch" in report:
        gpu = report["gpu_tile_dispatch"]
        gpu["reported_elapsed_ms_total"] = _perf_ms(float(gpu.get("reported_elapsed_ms_total", 0.0)))
        gpu["wall_ms_total"] = _perf_ms(float(gpu.get("wall_ms_total", 0.0)))
    return report


def _profile_compact_cuda_transfer_dispatch() -> dict[str, Any]:
    gpu_accel = importlib.import_module("gpu_accel")
    availability = gpu_accel.is_available(refresh=True)
    report: dict[str, Any] = {
        "backend": {"id": "compact_cuda_dll", "name": "compact CUDA DLL"},
        "availability": availability,
        "status": "skipped",
        "reason": availability.get("reason"),
        "message": availability.get("message"),
        "notes": [
            "CUDA DLL v1 exposes one combined host call; timings include Python validation plus host/device transfer, kernel dispatch, synchronization, and readback.",
            "Direct transfer/dispatch/readback sub-timers are not exposed by the current compact ABI.",
        ],
    }
    if not availability.get("available"):
        return report

    rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings = _gpu_transition_tile((1024, 1024))
    transition_strength = gpu_accel.transition_repair_strength_mask_v1(
        rgb,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        fringe,
        foreground,
        nearest_valid,
        key_color,
        settings,
    )
    foreground_valid_u8 = np.ascontiguousarray(nearest_valid.astype(np.uint8) * 255)

    def cpu_direct() -> None:
        gpu_accel.transition_repair_cpu_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)

    def cuda_direct() -> None:
        gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)

    def cuda_dispatch() -> None:
        result = gpu_accel.process_color_tile_gpu(
            rgb,
            alpha_u8,
            background_mask,
            edge_mask,
            probability,
            fringe,
            None,
            foreground,
            nearest_valid,
            key_color,
            replace(settings, gpu_acceleration="Force GPU"),
            force_gpu=True,
        )
        if not result.get("used"):
            raise RuntimeError(str(result.get("message") or result.get("reason") or "GPU dispatch skipped"))

    try:
        cpu_ms = _median_time_ms(cpu_direct, repeat=3, warmup=1)
        cuda_direct_ms = _median_time_ms(cuda_direct, repeat=3, warmup=1)
        cuda_dispatch_ms = _median_time_ms(cuda_dispatch, repeat=3, warmup=1)
        cpu_rgb, cpu_mask = gpu_accel.transition_repair_cpu_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
        dll_rgb, dll_mask = gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid_u8, key_color, settings)
        dispatch_result = gpu_accel.process_color_tile_gpu(
            rgb,
            alpha_u8,
            background_mask,
            edge_mask,
            probability,
            fringe,
            None,
            foreground,
            nearest_valid,
            key_color,
            replace(settings, gpu_acceleration="Force GPU"),
            force_gpu=True,
        )
    except Exception as exc:
        report.update({"status": "error", "reason": "cuda_profile_failed", "message": f"{type(exc).__name__}: {exc}"})
        return report

    report.update(
        {
            "status": "measured",
            "reason": None,
            "message": "Compact CUDA transfer/dispatch/readback profile completed.",
            "tile_shape": list(rgb.shape[:2]),
            "active_repair_pixels": int(np.count_nonzero(transition_strength)),
            "cpu_reference_ms": _perf_ms(cpu_ms),
            "cuda_dll_ms_including_transfer": _perf_ms(cuda_direct_ms),
            "cuda_process_color_tile_ms_including_transfer": _perf_ms(cuda_dispatch_ms),
            "cuda_reported_elapsed_ms": _perf_ms(float(dispatch_result.get("elapsed_ms") or 0.0)),
            "speedup_direct_vs_cpu": float(cpu_ms / cuda_direct_ms) if cuda_direct_ms > 0 else None,
            "speedup_dispatch_vs_cpu": float(cpu_ms / cuda_dispatch_ms) if cuda_dispatch_ms > 0 else None,
            "max_rgb_diff_vs_cpu": int(np.max(np.abs(cpu_rgb.astype(np.int16) - dll_rgb.astype(np.int16)))),
            "max_mask_diff_vs_cpu": int(np.max(np.abs(cpu_mask.astype(np.int16) - dll_mask.astype(np.int16)))),
            "dispatch_result": {k: v for k, v in dispatch_result.items() if k not in {"rgb", "repair_mask"}},
        }
    )
    return report


def _profile_d3d12_full_color_tile_dispatch() -> dict[str, Any]:
    gpu_backend = importlib.import_module("gpu_backend")
    d3d12 = gpu_backend.D3D12ComputeBackend()
    availability = d3d12.probe(refresh=True)
    report: dict[str, Any] = {
        "backend": {"id": "d3d12_compute", "name": "D3D12 compute backend"},
        "availability": availability,
        "status": "skipped",
        "reason": availability.get("reason"),
        "message": availability.get("message"),
        "notes": [
            "D3D12 Phase 6 path fuses linear conversion, unmix/clamp/despill/luma protect, nearest-inner pull, transition reference repair, transparent-RGB enforcement, and screen-tile support in the native full-color tile kernel.",
            "Tiles above max_native_call_pixels are split into TDR-bounded native calls while reusing the render-session D3D12 context and persistent buffers.",
        ],
    }
    if not availability.get("available"):
        return report

    rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, nearest_valid, key_color, settings = _gpu_transition_tile((2048, 2048))
    cpu_settings = replace(settings, gpu_acceleration="Off")
    gpu_settings = replace(settings, gpu_acceleration="Force GPU")

    def cpu_full_color() -> None:
        _process_color_tile(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, cpu_settings)

    session = gpu_backend.begin_render(gpu_settings, rgb.shape, required_capabilities={"constant_screen", "rgb_only", "full_color_tile"}, backends=[d3d12])

    def d3d12_full_color() -> None:
        result = session.process_full_color_tile(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, gpu_settings)
        if not result.get("used"):
            raise RuntimeError(str(result.get("message") or result.get("reason") or "D3D12 dispatch skipped"))

    try:
        cpu_ms = _median_time_ms(cpu_full_color, repeat=3, warmup=1)
        d3d12_ms = _median_time_ms(d3d12_full_color, repeat=5, warmup=2)
        cpu_rgb, cpu_mask = _process_color_tile(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, cpu_settings)
        result = session.process_full_color_tile(rgb, alpha_u8, background_mask, edge_mask, probability, fringe, None, foreground, nearest_valid, key_color, gpu_settings)
    except Exception as exc:
        report.update({"status": "error", "reason": "d3d12_profile_failed", "message": f"{type(exc).__name__}: {exc}"})
        return report
    finally:
        session.end_render()

    rgb_delta = np.abs(cpu_rgb.astype(np.int16) - result["rgb"].astype(np.int16))
    mask_delta = np.abs(cpu_mask.astype(np.int16) - result["repair_mask"].astype(np.int16))
    report.update(
        {
            "status": "measured",
            "reason": None,
            "message": "D3D12 full color tile transfer/dispatch/readback profile completed.",
            "tile_shape": list(rgb.shape[:2]),
            "cpu_full_color_ms": _perf_ms(cpu_ms),
            "d3d12_ms_including_transfer": _perf_ms(d3d12_ms),
            "d3d12_reported_elapsed_ms": _perf_ms(float(result.get("elapsed_ms") or 0.0)),
            "subtile_dispatches": int(result.get("subtile_dispatches") or 1),
            "subtile_max_pixels": int(result.get("subtile_max_pixels") or 0),
            "speedup_vs_cpu": float(cpu_ms / d3d12_ms) if d3d12_ms > 0 else None,
            "faster_than_cpu": bool(d3d12_ms < cpu_ms),
            "max_rgb_diff_vs_cpu": int(rgb_delta.max()),
            "p99_rgb_diff_vs_cpu": float(np.percentile(rgb_delta, 99)),
            "max_mask_diff_vs_cpu": int(mask_delta.max()),
            "dispatch_result": {k: v for k, v in result.items() if k not in {"rgb", "repair_mask"}},
        }
    )
    return report


def _perf_case_stage(case: dict[str, Any], stage: str) -> dict[str, Any]:
    return case.get("timings", {}).get(stage, {"total_ms": 0.0, "calls": 0, "median_ms": 0.0, "max_ms": 0.0})


def _perf_report_text(summary: dict[str, Any]) -> str:
    lines = [
        "ImgKey pipeline performance baseline",
        "====================================",
        f"Generated by: {summary['generated_by']}",
        f"Artifact dir: {summary['artifact_dir']}",
        "",
        "Large tiled CPU-reference pipeline cases:",
    ]
    tracked = [
        "image_load",
        "preview_resize",
        "global_matte_total",
        "screen_model_full_image",
        "screen_model_local_plate_tile",
        "nearest_inner_reference_global",
        "nearest_inner_reference_tile_local",
        "transition_alpha_recovery",
        "tiled_rgba_render_total",
        "per_tile_color_render",
        "png_encode_export",
        "process_key_image_total",
    ]
    for case in summary.get("cases", []):
        lines.append("")
        lines.append(f"- {case['name']} ({case['shape'][0]}x{case['shape'][1]}, {case['background_name']}):")
        tiles = case.get("tiles", {})
        lines.append(f"  tiles={tiles.get('count', 0)} read_shapes={tiles.get('read_shapes', {})}")
        for stage in tracked:
            record = _perf_case_stage(case, stage)
            if float(record.get("total_ms", 0.0)) <= 0.0:
                continue
            lines.append(
                f"  {stage}: total={record['total_ms']:.3f} ms calls={record['calls']} "
                f"median={record.get('median_ms', 0.0):.3f} ms max={record.get('max_ms', 0.0):.3f} ms"
            )
        top = sorted(
            (
                (stage, float(case.get("timings", {}).get(stage, {}).get("total_ms", 0.0)))
                for stage in tracked
                if stage not in {"process_key_image_total"}
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        lines.append("  top stages: " + ", ".join(f"{stage}={elapsed:.1f} ms" for stage, elapsed in top if elapsed > 0.0))
        result = case.get("result", {})
        lines.append(
            f"  alpha_nonzero={result.get('alpha_nonzero_pixels')} alpha_mean={float(result.get('alpha_mean', 0.0)):.2f} "
            f"transparent_rgb_zero={result.get('transparent_rgb_zero', {}).get('ok')} export_bytes={case.get('export_png_bytes')}"
        )

    gpu = summary.get("compact_cuda_transfer_dispatch", {})
    lines.extend(["", "Compact CUDA transfer/dispatch/readback:"])
    lines.append(
        f"- status={gpu.get('status')} reason={gpu.get('reason')} message={gpu.get('message')}"
    )
    if gpu.get("status") == "measured":
        lines.append(
            f"  tile={gpu.get('tile_shape')} active_pixels={gpu.get('active_repair_pixels')} "
            f"cpu={gpu.get('cpu_reference_ms'):.3f} ms cuda_direct={gpu.get('cuda_dll_ms_including_transfer'):.3f} ms "
            f"cuda_dispatch={gpu.get('cuda_process_color_tile_ms_including_transfer'):.3f} ms "
            f"speedup_direct={gpu.get('speedup_direct_vs_cpu'):.2f}x max_rgb_diff={gpu.get('max_rgb_diff_vs_cpu')}"
        )
    d3d12 = summary.get("d3d12_full_color_tile_dispatch", {})
    lines.extend(["", "D3D12 full color tile transfer/dispatch/readback:"])
    lines.append(
        f"- status={d3d12.get('status')} reason={d3d12.get('reason')} message={d3d12.get('message')}"
    )
    if d3d12.get("status") == "measured":
        lines.append(
            f"  tile={d3d12.get('tile_shape')} subtiles={d3d12.get('subtile_dispatches')} "
            f"cpu={d3d12.get('cpu_full_color_ms'):.3f} ms d3d12={d3d12.get('d3d12_ms_including_transfer'):.3f} ms "
            f"speedup={d3d12.get('speedup_vs_cpu'):.2f}x max_rgb_diff={d3d12.get('max_rgb_diff_vs_cpu')}"
        )
    lines.extend(
        [
            "",
            "Baseline interpretation:",
            "- This command only profiles current behavior; it does not change keyer settings or output logic.",
            "- Large cases use resized geometric benchmark sources: 4096 flat blue screen and 8192 uneven-gradient blue screen.",
            "- Current compact CUDA timing is reported as one combined host call because the CUDA v1 ABI does not expose separate transfer, dispatch, and readback timers.",
            "- D3D12 full-color timing reports the Phase 6 fused tile graph independently from the CPU-reference large export cases so fallback correctness and acceleration evidence are both visible.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_perf_baseline() -> None:
    PERF_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing pipeline performance baseline to {PERF_BASELINE_DIR}")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "python smoke_test.py --write-perf-baseline",
        "artifact_dir": str(PERF_BASELINE_DIR),
        "python": sys.version,
        "cases": [],
    }
    for name, template_case, size in _large_perf_case_specs():
        print(f"profiling {name} ({size}x{size})")
        summary["cases"].append(_profile_large_pipeline_case(name, template_case, size))
    summary["compact_cuda_transfer_dispatch"] = _profile_compact_cuda_transfer_dispatch()
    summary["d3d12_full_color_tile_dispatch"] = _profile_d3d12_full_color_tile_dispatch()
    report_text = _perf_report_text(summary)
    summary_path = PERF_BASELINE_DIR / "pipeline_baseline.json"
    report_path = PERF_BASELINE_DIR / "pipeline_baseline.txt"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text.rstrip())
    print(f"wrote pipeline performance baseline JSON to {summary_path}")
    print(f"wrote pipeline performance baseline report to {report_path}")


GEOMETRIC_PRIMARY_FEATURES = (
    "transparency_bands",
    "dots_speckles",
    "lines_1px",
    "lines_2px",
    "diagonal_lines",
    "curves_rings",
    "holes",
    "interior_key_colored",
    "hard_edges",
    "anti_aliased_edges",
)

GEOMETRIC_COLOR_FEATURES = (
    "foreground_color_red",
    "foreground_color_white",
    "foreground_color_black",
    "foreground_color_gray",
    "foreground_color_yellow",
    "foreground_color_saturated",
    "foreground_color_key",
)

GEOMETRIC_TUNING_SCORE_WEIGHTS = {
    "thin_line_visible_recall": 28.0,
    "thin_line_alpha_ratio": 10.0,
    "dot_visible_recall": 8.0,
    "background_leak": 22.0,
    "foreground_loss": 10.0,
    "edge_key_residual": 14.0,
    "foreground_core_rgb": 12.0,
    "alpha_mae": 8.0,
}

GEOMETRIC_TUNING_PROMOTION_TOLERANCES = {
    "minimum_score_improvement_fraction": 0.05,
    "detail_recall_epsilon": 1e-9,
    "background_leak_pixel_slack": 5,
    "background_leak_rate_slack": 0.001,
    "foreground_loss_rate_slack": 0.002,
    "foreground_core_rgb_mean_abs_error_max": 5.0,
    "legacy_alpha_mae_slack": 0.005,
    "legacy_detail_recall_slack": 0.001,
    "legacy_core_rgb_delta_slack": 1.0,
}


def _geometric_circle_mask(shape: tuple[int, int], cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    return (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= float(radius) ** 2


def _geometric_line_alpha(
    shape: tuple[int, int],
    points: list[tuple[int, int]],
    *,
    width: int = 1,
    antialias: bool = False,
) -> np.ndarray:
    h, w = shape
    if antialias:
        return _draw_antialiased_alpha(
            shape,
            lambda draw, scale: draw.line(
                [(int(x * scale), int(y * scale)) for x, y in points],
                fill=255,
                width=max(1, int(width * scale)),
            ),
            scale=6,
        )
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).line(points, fill=255, width=max(1, int(width)))
    return np.asarray(mask, dtype=np.float32) / 255.0


def _geometric_ring_alpha(
    shape: tuple[int, int],
    outer_box: tuple[int, int, int, int],
    inner_box: tuple[int, int, int, int],
) -> np.ndarray:
    outer = _draw_antialiased_alpha(
        shape,
        lambda draw, scale: draw.ellipse([int(v * scale) for v in outer_box], fill=255),
        scale=6,
    )
    inner = _draw_antialiased_alpha(
        shape,
        lambda draw, scale: draw.ellipse([int(v * scale) for v in inner_box], fill=255),
        scale=6,
    )
    return np.clip(outer - inner, 0.0, 1.0)


def _geometric_expected_rgba(expected_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.zeros((*alpha_u8.shape, 4), dtype=np.uint8)
    visible = alpha_u8 > 0
    rgba[visible, :3] = expected_rgb[visible]
    rgba[:, :, 3] = alpha_u8
    return rgba


def generate_geometric_benchmark_asset() -> GeometricBenchmarkAsset:
    shape = (360, 520)
    h, w = shape
    alpha = np.zeros(shape, dtype=np.float32)
    foreground = np.zeros((h, w, 3), dtype=np.uint8)
    key_color_region = np.zeros(shape, dtype=bool)
    feature_masks: dict[str, np.ndarray] = {}

    def feature(name: str) -> np.ndarray:
        if name not in feature_masks:
            feature_masks[name] = np.zeros(shape, dtype=bool)
        return feature_masks[name]

    def add_region(
        region_alpha: np.ndarray,
        color: tuple[int, int, int],
        names: tuple[str, ...],
        color_feature: str | None = None,
        *,
        key_colored: bool = False,
    ) -> None:
        visible = np.asarray(region_alpha, dtype=np.float32) > (0.5 / 255.0)
        if not np.any(visible):
            return
        region = np.clip(region_alpha.astype(np.float32), 0.0, 1.0)
        replace_pixels = visible & (region >= alpha)
        alpha[visible] = np.maximum(alpha[visible], region[visible])
        for group_name in GEOMETRIC_COLOR_FEATURES:
            if group_name in feature_masks:
                feature_masks[group_name][replace_pixels] = False
        foreground[replace_pixels] = np.asarray(color, dtype=np.uint8)
        for name in names:
            feature(name)[visible] = True
        if color_feature is not None:
            feature(color_feature)[replace_pixels] = True
        if key_colored:
            key_color_region[replace_pixels] = True

    # Hard-edged opaque solids and color groups.
    hard_red = np.zeros(shape, dtype=np.float32)
    hard_red[28:112, 30:128] = 1.0
    add_region(hard_red, (230, 34, 28), ("hard_edges",), "foreground_color_red")

    hard_yellow = np.zeros(shape, dtype=np.float32)
    hard_yellow[34:122, 146:238] = 1.0
    hard_yellow[70:96, 184:238] = 0.0
    add_region(hard_yellow, (245, 214, 36), ("hard_edges",), "foreground_color_yellow")

    # Opaque plate with true holes and key-colored foreground islands.
    panel = np.zeros(shape, dtype=np.float32)
    panel[134:238, 28:202] = 1.0
    add_region(panel, (164, 164, 164), ("hard_edges", "holes_interior_key_regions"), "foreground_color_gray")
    holes = (
        _geometric_circle_mask(shape, 76, 182, 18)
        | _geometric_circle_mask(shape, 118, 205, 10)
        | ((np.indices(shape)[0] >= 154) & (np.indices(shape)[0] < 171) & (np.indices(shape)[1] >= 156) & (np.indices(shape)[1] < 184))
    )
    holes &= panel > 0
    alpha[holes] = 0.0
    foreground[holes] = 0
    key_color_region[holes] = False
    for group_name in GEOMETRIC_COLOR_FEATURES:
        if group_name in feature_masks:
            feature_masks[group_name][holes] = False
    for name in ("holes", "holes_interior_key_regions"):
        feature(name)[holes] = True

    key_island = (_geometric_circle_mask(shape, 158, 178, 15) | ((np.indices(shape)[0] >= 207) & (np.indices(shape)[0] < 226) & (np.indices(shape)[1] >= 132) & (np.indices(shape)[1] < 190)))
    key_island &= panel > 0
    add_region(
        key_island.astype(np.float32),
        (0, 0, 0),
        ("interior_key_colored", "holes_interior_key_regions", "hard_edges"),
        "foreground_color_key",
        key_colored=True,
    )

    # Analytic transparency ramps over several foreground colors.
    ramp = np.linspace(0.08, 0.92, w, dtype=np.float32)
    for y0, y1, color, color_name in (
        (256, 274, (232, 42, 34), "foreground_color_red"),
        (280, 298, (255, 255, 255), "foreground_color_white"),
        (304, 322, (3, 3, 3), "foreground_color_black"),
        (328, 344, (70, 70, 220), "foreground_color_saturated"),
    ):
        band = np.zeros(shape, dtype=np.float32)
        band[y0:y1, 26:246] = ramp[26:246]
        add_region(band, color, ("transparency_bands",), color_name)

    # Deterministic dots and speckles: sub-3px details, mixed alpha, mixed colors.
    rng = np.random.default_rng(20260519)
    dot_colors = [
        ((255, 255, 255), "foreground_color_white"),
        ((2, 2, 2), "foreground_color_black"),
        ((230, 40, 32), "foreground_color_red"),
        ((250, 214, 28), "foreground_color_yellow"),
        ((34, 208, 230), "foreground_color_saturated"),
    ]
    for index in range(86):
        cx = int(rng.integers(284, 504))
        cy = int(rng.integers(30, 134))
        radius = int(rng.choice(np.asarray([1, 1, 1, 2, 2, 3], dtype=np.int32)))
        dot_alpha = 0.52 if index % 7 == 0 else 1.0
        color, color_name = dot_colors[index % len(dot_colors)]
        add_region(
            _geometric_circle_mask(shape, cx, cy, radius).astype(np.float32) * dot_alpha,
            color,
            ("dots_speckles",),
            color_name,
        )

    # Single- and two-pixel barcode strokes.
    line1_white = np.zeros(shape, dtype=np.float32)
    line1_black = np.zeros(shape, dtype=np.float32)
    line1_red = np.zeros(shape, dtype=np.float32)
    for idx, x in enumerate(range(280, 360, 6)):
        target = (line1_white, line1_black, line1_red)[idx % 3]
        target[154:238, x] = 1.0
    for idx, y in enumerate(range(160, 236, 9)):
        target = (line1_white, line1_black, line1_red)[(idx + 1) % 3]
        target[y, 366:504] = 1.0
    add_region(line1_white, (255, 255, 255), ("lines_1px", "hard_edges"), "foreground_color_white")
    add_region(line1_black, (0, 0, 0), ("lines_1px", "hard_edges"), "foreground_color_black")
    add_region(line1_red, (230, 36, 32), ("lines_1px", "hard_edges"), "foreground_color_red")

    line2_yellow = np.zeros(shape, dtype=np.float32)
    line2_sat = np.zeros(shape, dtype=np.float32)
    for idx, x in enumerate(range(276, 506, 14)):
        target = line2_yellow if idx % 2 == 0 else line2_sat
        target[244:322, x : x + 2] = 1.0
    for idx, y in enumerate(range(250, 324, 14)):
        target = line2_sat if idx % 2 == 0 else line2_yellow
        target[y : y + 2, 276:506] = 1.0
    add_region(line2_yellow, (248, 210, 36), ("lines_2px", "hard_edges"), "foreground_color_yellow")
    add_region(line2_sat, (40, 210, 228), ("lines_2px", "hard_edges"), "foreground_color_saturated")

    diagonal_hard = _geometric_line_alpha(shape, [(282, 332), (506, 250)], width=1)
    add_region(diagonal_hard, (255, 255, 255), ("diagonal_lines", "lines_1px", "hard_edges"), "foreground_color_white")
    diagonal_wide = _geometric_line_alpha(shape, [(284, 252), (505, 336)], width=2)
    add_region(diagonal_wide, (0, 0, 0), ("diagonal_lines", "lines_2px", "hard_edges"), "foreground_color_black")
    diagonal_soft = _geometric_line_alpha(shape, [(270, 142), (506, 224)], width=4, antialias=True)
    add_region(diagonal_soft, (236, 44, 172), ("diagonal_lines", "anti_aliased_edges"), "foreground_color_saturated")

    # Anti-aliased rings and curves generated by supersampling/downsampling.
    ring = _geometric_ring_alpha(shape, (330, 246, 488, 344), (356, 270, 462, 320))
    add_region(ring, (244, 210, 42), ("curves_rings", "anti_aliased_edges"), "foreground_color_yellow")

    ellipse = _draw_antialiased_alpha(
        shape,
        lambda draw, scale: draw.ellipse([int(v * scale) for v in (342, 138, 492, 224)], fill=255),
        scale=6,
    )
    add_region(ellipse, (186, 186, 186), ("curves_rings", "anti_aliased_edges"), "foreground_color_gray")

    arc = _draw_antialiased_alpha(
        shape,
        lambda draw, scale: draw.arc(
            [int(v * scale) for v in (238, 20, 506, 210)],
            start=188,
            end=342,
            fill=255,
            width=max(1, int(5 * scale)),
        ),
        scale=6,
    )
    add_region(arc, (255, 255, 255), ("curves_rings", "anti_aliased_edges"), "foreground_color_white")

    transition_pixels = (alpha > 0.002) & (alpha < 0.998) & ~feature("transparency_bands")
    feature("anti_aliased_edges")[transition_pixels] = True
    key_color_region &= alpha > 0.0

    ordered_names = GEOMETRIC_PRIMARY_FEATURES + GEOMETRIC_COLOR_FEATURES + ("holes_interior_key_regions",)
    ordered_masks = {name: feature_masks[name] for name in ordered_names if name in feature_masks and np.any(feature_masks[name])}
    return GeometricBenchmarkAsset(
        name="geometric_asset_v1",
        alpha=alpha,
        foreground_rgb_template=foreground,
        key_color_region=key_color_region,
        feature_masks=ordered_masks,
        primary_feature_order=GEOMETRIC_PRIMARY_FEATURES,
        notes="Deterministic synthetic RGBA geometry fixture with analytic/supersampled alpha and per-feature masks.",
    )


def _geometric_current_default_settings(key_color: tuple[int, int, int]) -> KeySettings:
    return KeySettings(
        key_color=key_color,
        tolerance=0.26,
        softness=0.02,
        edge_blur=(24 - 1) / 4.0,
        cleanup=0,
        despill=0.80,
        sample_size=10,
        auto_border_sample=True,
        auto_detect_key_color=False,
        clip_background=0.95,
        clip_foreground=0.08,
        matte_gamma=1.60,
        core_strength=0.45,
        edge_refine_radius=24,
        edge_softness=0.04,
        erode_expand=-4,
        despeckle_min_area=0,
        aggressive_interior_removal=True,
        decontaminate=0.70,
        luminance_restore=0.85,
        luminance_protect=0.85,
        fringe_remove=0.85,
        edge_color_repair=0.80,
        inner_color_pull=0.60,
        fringe_band_radius=5,
        transition_unmix=True,
        alpha_recover_strength=0.90,
        key_vector_despill=0.85,
        foreground_reference_pull=0.75,
        screen_cleanup_strength=1.00,
        screen_cleanup_similarity=8,
        gpu_acceleration="Off",
    )


def _geometric_benchmark_settings(key_color: tuple[int, int, int]) -> KeySettings:
    return _geometric_current_default_settings(key_color)


def _geometric_strict_asset_settings(key_color: tuple[int, int, int]) -> KeySettings:
    return replace(
        _geometric_current_default_settings(key_color),
        tolerance=0.01,
        softness=0.01,
        clip_background=0.97,
        clip_foreground=0.33,
        matte_gamma=2.20,
        core_strength=0.38,
        edge_refine_radius=32,
        edge_blur=(32 - 1) / 4.0,
        edge_softness=0.00,
        erode_expand=-8,
        despeckle_min_area=0,
        aggressive_interior_removal=True,
        despill=1.00,
        decontaminate=1.00,
        luminance_restore=1.00,
        luminance_protect=1.00,
        fringe_remove=1.00,
        edge_color_repair=1.00,
        inner_color_pull=1.00,
        fringe_band_radius=12,
        transition_unmix=True,
        alpha_recover_strength=1.00,
        key_vector_despill=1.00,
        foreground_reference_pull=1.00,
        screen_cleanup_strength=1.00,
        screen_cleanup_similarity=8,
    )


def _geometric_moderated_strict_settings(key_color: tuple[int, int, int]) -> KeySettings:
    return replace(
        _geometric_current_default_settings(key_color),
        tolerance=0.06,
        softness=0.01,
        clip_background=0.97,
        clip_foreground=0.18,
        matte_gamma=2.20,
        core_strength=0.38,
        edge_refine_radius=32,
        edge_blur=(32 - 1) / 4.0,
        edge_softness=0.00,
        erode_expand=-8,
        despeckle_min_area=0,
        aggressive_interior_removal=True,
        despill=0.90,
        decontaminate=0.85,
        luminance_restore=0.90,
        luminance_protect=0.90,
        fringe_remove=0.90,
        edge_color_repair=0.85,
        inner_color_pull=0.75,
        fringe_band_radius=8,
        transition_unmix=True,
        alpha_recover_strength=0.92,
        key_vector_despill=0.90,
        foreground_reference_pull=0.85,
        screen_cleanup_strength=1.00,
        screen_cleanup_similarity=8,
    )


def _geometric_green_cyan_safe_settings(key_color: tuple[int, int, int]) -> KeySettings:
    return replace(
        _geometric_current_default_settings(key_color),
        tolerance=0.26,
        softness=0.02,
        clip_background=0.95,
        clip_foreground=0.08,
        matte_gamma=1.60,
        core_strength=0.45,
        edge_refine_radius=24,
        edge_blur=(24 - 1) / 4.0,
        edge_softness=0.04,
        erode_expand=-4,
        despeckle_min_area=0,
        aggressive_interior_removal=True,
        despill=0.80,
        decontaminate=0.70,
        luminance_restore=0.85,
        luminance_protect=0.85,
        fringe_remove=0.85,
        edge_color_repair=0.80,
        inner_color_pull=0.60,
        fringe_band_radius=5,
        transition_unmix=True,
        alpha_recover_strength=0.90,
        key_vector_despill=0.85,
        foreground_reference_pull=0.75,
        screen_cleanup_strength=1.00,
        screen_cleanup_similarity=8,
    )


def geometric_tuning_profiles(key_color: tuple[int, int, int]) -> list[GeometricTuningProfile]:
    return [
        GeometricTuningProfile(
            name="current_app_default",
            label="Current app default",
            description="Existing High Accuracy Graphic default settings, with the benchmark case key color substituted.",
            settings=_geometric_current_default_settings(key_color),
        ),
        GeometricTuningProfile(
            name="asset_strict_screenshot",
            label="Asset Strict screenshot",
            description="User screenshot strict asset profile: narrow tolerance, clip FG 0.33, full transition and color-repair strengths.",
            settings=_geometric_strict_asset_settings(key_color),
        ),
        GeometricTuningProfile(
            name="moderated_strict",
            label="Moderated strict",
            description="Strict-profile fallback with less foreground clipping and reduced repair pull to avoid over-cleaning.",
            settings=_geometric_moderated_strict_settings(key_color),
        ),
        GeometricTuningProfile(
            name="green_cyan_safe",
            label="Green/cyan safe",
            description="Moderate tolerance and repair strengths aimed at green, cyan, and uneven gradient robustness.",
            settings=_geometric_green_cyan_safe_settings(key_color),
        ),
    ]


def _geometric_background(
    shape: tuple[int, int],
    key_color: tuple[int, int, int],
    *,
    uneven: bool,
) -> np.ndarray:
    h, w = shape
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = np.asarray(key_color, dtype=np.uint8)
    if not uneven:
        return background
    x_grad = np.linspace(-1.0, 1.0, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-1.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    base = np.asarray(key_color, dtype=np.float32).reshape(1, 1, 3)
    uneven_rgb = np.zeros((h, w, 3), dtype=np.float32)
    uneven_rgb[:, :, 0] = base[:, :, 0] + 8.0 * x_grad + 5.0 * y_grad
    uneven_rgb[:, :, 1] = base[:, :, 1] + 24.0 * x_grad + 15.0 * y_grad
    uneven_rgb[:, :, 2] = base[:, :, 2] - 20.0 * x_grad + 12.0 * y_grad
    return np.clip(uneven_rgb, 0, 255).astype(np.uint8)


def geometric_benchmark_cases() -> list[GeometricBenchmarkCase]:
    asset = generate_geometric_benchmark_asset()
    specs = [
        ("blue_flat", (30, 80, 235), False),
        ("green_flat", (0, 220, 50), False),
        ("cyan_flat", (0, 190, 210), False),
        ("blue_uneven_gradient", (30, 80, 235), True),
        ("green_uneven_gradient", (0, 220, 50), True),
        ("cyan_uneven_gradient", (0, 190, 210), True),
    ]
    cases: list[GeometricBenchmarkCase] = []
    for background_name, key_color, uneven in specs:
        expected_foreground = asset.foreground_rgb_template.copy()
        expected_foreground[asset.key_color_region] = np.asarray(key_color, dtype=np.uint8)
        background = _geometric_background(asset.alpha.shape, key_color, uneven=uneven)
        source = _composite_rgb_linear(background, expected_foreground, asset.alpha)
        expected_rgba = _geometric_expected_rgba(expected_foreground, asset.alpha)
        case_name = f"geometric_{background_name}"
        cases.append(
            GeometricBenchmarkCase(
                name=case_name,
                background_name=background_name,
                key_color=key_color,
                background_rgb=background,
                source_rgb=source,
                expected_alpha=asset.alpha,
                expected_foreground_rgb=expected_foreground,
                expected_rgba=expected_rgba,
                feature_masks=asset.feature_masks,
                settings=_geometric_benchmark_settings(key_color),
                notes=f"{asset.notes} Background: {background_name}.",
            )
        )
    return cases


def _geometric_alpha_metrics(expected_alpha: np.ndarray, actual_alpha: np.ndarray, mask: np.ndarray) -> dict[str, int | float]:
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return {
            "count": 0,
            "alpha_mae": 0.0,
            "alpha_mae_u8": 0.0,
            "alpha_max_abs_error": 0,
            "alpha_precision": 1.0,
            "alpha_recall": 1.0,
            "false_background_loss_pixels": 0,
            "false_foreground_leak_pixels": 0,
            "false_background_leak_pixels": 0,
            "false_background_loss_rate": 0.0,
            "false_foreground_leak_rate": 0.0,
        }
    expected_u8 = np.rint(np.clip(expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    expected = expected_u8[mask].astype(np.int16)
    actual = actual_alpha[mask].astype(np.int16)
    abs_delta = np.abs(actual - expected)
    expected_visible = expected > 0
    actual_visible = actual > 0
    true_positive = expected_visible & actual_visible
    false_loss = expected_visible & ~actual_visible
    false_leak = ~expected_visible & actual_visible
    expected_visible_count = int(np.count_nonzero(expected_visible))
    actual_visible_count = int(np.count_nonzero(actual_visible))
    expected_background_count = int(expected.size - expected_visible_count)
    return {
        "count": int(expected.size),
        "alpha_mae": float(np.mean(abs_delta) / 255.0),
        "alpha_mae_u8": float(np.mean(abs_delta)),
        "alpha_max_abs_error": int(abs_delta.max()),
        "alpha_precision": float(np.count_nonzero(true_positive) / actual_visible_count) if actual_visible_count else 1.0,
        "alpha_recall": float(np.count_nonzero(true_positive) / expected_visible_count) if expected_visible_count else 1.0,
        "false_background_loss_pixels": int(np.count_nonzero(false_loss)),
        "false_foreground_leak_pixels": int(np.count_nonzero(false_leak)),
        "false_background_leak_pixels": int(np.count_nonzero(false_leak)),
        "false_background_loss_rate": float(np.count_nonzero(false_loss) / expected_visible_count) if expected_visible_count else 0.0,
        "false_foreground_leak_rate": float(np.count_nonzero(false_leak) / expected_background_count) if expected_background_count else 0.0,
    }


def _geometric_rgb_error(
    result_rgba: np.ndarray,
    expected_rgb: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int | float]:
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return {"count": 0, "mean_abs_error": 0.0, "max_abs_error": 0, "mean_l2_error": 0.0}
    delta = np.abs(result_rgba[mask, :3].astype(np.int16) - expected_rgb[mask].astype(np.int16))
    l2 = np.linalg.norm(delta.astype(np.float32), axis=1)
    return {
        "count": int(delta.shape[0]),
        "mean_abs_error": float(np.mean(delta)),
        "max_abs_error": int(delta.max()),
        "mean_l2_error": float(np.mean(l2)),
    }


def _geometric_region_metrics(
    case: GeometricBenchmarkCase,
    result: KeyResult,
    mask: np.ndarray,
) -> dict[str, Any]:
    expected_u8 = np.rint(np.clip(case.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask = mask.astype(bool, copy=False)
    translucent = mask & (expected_u8 > 0) & (expected_u8 < 255) & (result.alpha > 0)
    visible = mask & (expected_u8 > 0) & (result.alpha > 0)
    core = mask & (expected_u8 >= 254) & (result.alpha >= 250)
    return {
        "alpha": _geometric_alpha_metrics(case.expected_alpha, result.alpha, mask),
        "rgb_error_visible": _geometric_rgb_error(result.rgba, case.expected_foreground_rgb, visible),
        "rgb_error_translucent": _geometric_rgb_error(result.rgba, case.expected_foreground_rgb, translucent),
        "foreground_core_rgb_delta": _geometric_rgb_error(result.rgba, case.expected_foreground_rgb, core),
        "composite_error": composite_black_white_gray_error(result.rgba, case.expected_rgba, mask),
    }


def _geometric_masked_recall(case: GeometricBenchmarkCase, result: KeyResult, mask: np.ndarray) -> dict[str, int | float]:
    expected = np.where(mask.astype(bool, copy=False), case.expected_alpha, 0.0)
    return alpha_detail_recall(expected, result.alpha)


def _geometric_case_metrics(case: GeometricBenchmarkCase, result: KeyResult) -> dict[str, Any]:
    shape = case.expected_alpha.shape
    whole = np.ones(shape, dtype=bool)
    background = case.expected_alpha <= 0.002
    anti_edge = case.feature_masks.get("anti_aliased_edges", np.zeros(shape, dtype=bool))
    curves = case.feature_masks.get("curves_rings", np.zeros(shape, dtype=bool))
    edge_mask = anti_edge | curves
    thin_mask = (
        case.feature_masks.get("lines_1px", np.zeros(shape, dtype=bool))
        | case.feature_masks.get("lines_2px", np.zeros(shape, dtype=bool))
        | case.feature_masks.get("diagonal_lines", np.zeros(shape, dtype=bool))
    )
    dots_mask = case.feature_masks.get("dots_speckles", np.zeros(shape, dtype=bool))
    feature_metrics = {
        name: _geometric_region_metrics(case, result, mask)
        for name, mask in case.feature_masks.items()
        if np.any(mask)
    }
    return {
        "shape": list(case.source_rgb.shape),
        "settings": asdict(case.settings),
        "feature_counts": {name: int(np.count_nonzero(mask)) for name, mask in case.feature_masks.items()},
        "whole": _geometric_region_metrics(case, result, whole),
        "features": feature_metrics,
        "thin_line_recall": _geometric_masked_recall(case, result, thin_mask),
        "dot_preservation": _geometric_masked_recall(case, result, dots_mask),
        "background_leak": background_alpha_leak(result.alpha, background),
        "edge_key_color_residual": edge_key_residual(result.rgba, case.key_color, edge_mask),
        "transparent_rgb_residual": transparent_rgb_zero(result.rgba),
        "result_gpu_acceleration": result.gpu_acceleration,
    }


def _geometric_alpha_diff_heatmap(actual_alpha: np.ndarray, expected_alpha: np.ndarray) -> np.ndarray:
    expected_u8 = np.rint(np.clip(expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    delta = actual_alpha.astype(np.int16) - expected_u8.astype(np.int16)
    positive = np.clip(delta, 0, 255).astype(np.uint8)
    negative = np.clip(-delta, 0, 255).astype(np.uint8)
    magnitude = np.clip(np.abs(delta), 0, 255).astype(np.uint8)
    heat = np.zeros((*actual_alpha.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = positive
    heat[:, :, 1] = magnitude // 2
    heat[:, :, 2] = negative
    return heat


def _geometric_rgb_diff_heatmap(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    magnitude = np.clip(np.mean(delta, axis=2) * 4.0, 0, 255).astype(np.uint8)
    heat = np.zeros_like(actual)
    heat[:, :, 0] = magnitude
    heat[:, :, 1] = magnitude // 3
    return heat


def _geometric_error_overlay(source: np.ndarray, actual_alpha: np.ndarray, expected_alpha: np.ndarray) -> np.ndarray:
    expected_u8 = np.rint(np.clip(expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    overlay = source.copy()
    loss = (expected_u8 > 0) & (actual_alpha == 0)
    leak = (expected_u8 == 0) & (actual_alpha > 0)
    large = np.abs(actual_alpha.astype(np.int16) - expected_u8.astype(np.int16)) > 48
    overlay[large] = (255, 210, 0)
    overlay[loss] = (255, 0, 0)
    overlay[leak] = (0, 96, 255)
    return overlay


def _write_geometric_feature_artifacts(asset: GeometricBenchmarkAsset) -> None:
    masks_path = GEOMETRIC_BENCHMARK_DIR / "geometric_feature_masks.npz"
    np.savez_compressed(
        masks_path,
        **{name: mask.astype(np.uint8) for name, mask in asset.feature_masks.items()},
    )
    labels = np.zeros(asset.alpha.shape, dtype=np.uint8)
    for index, name in enumerate(asset.primary_feature_order, start=1):
        mask = asset.feature_masks.get(name)
        if mask is not None:
            labels[(labels == 0) & mask] = index
    label_colors = np.asarray(
        [
            (0, 0, 0),
            (255, 128, 0),
            (255, 255, 0),
            (255, 255, 255),
            (190, 190, 190),
            (255, 0, 180),
            (0, 220, 255),
            (140, 80, 255),
            (0, 180, 80),
            (255, 0, 0),
            (0, 100, 255),
        ],
        dtype=np.uint8,
    )
    label_rgb = label_colors[np.clip(labels, 0, len(label_colors) - 1)]
    _save_rgb(GEOMETRIC_BENCHMARK_DIR / "geometric_feature_labels.png", label_rgb)
    (GEOMETRIC_BENCHMARK_DIR / "geometric_feature_labels.json").write_text(
        json.dumps(
            {
                "asset": asset.name,
                "primary_labels": {str(index): name for index, name in enumerate(asset.primary_feature_order, start=1)},
                "feature_counts": {name: int(np.count_nonzero(mask)) for name, mask in asset.feature_masks.items()},
                "masks_npz": masks_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_geometric_case_artifacts(case: GeometricBenchmarkCase, result: KeyResult) -> None:
    prefix = GEOMETRIC_BENCHMARK_DIR / case.name
    _save_rgb(prefix.with_name(f"{case.name}_source.png"), case.source_rgb)
    _save_rgb(prefix.with_name(f"{case.name}_background.png"), case.background_rgb)
    _save_rgba(prefix.with_name(f"{case.name}_expected_result.png"), case.expected_rgba)
    _save_mask(prefix.with_name(f"{case.name}_expected_alpha.png"), case.expected_rgba[:, :, 3])
    expected_foreground_preview = case.expected_foreground_rgb.copy()
    expected_foreground_preview[case.expected_rgba[:, :, 3] == 0] = 0
    _save_rgb(prefix.with_name(f"{case.name}_expected_foreground.png"), expected_foreground_preview)

    _save_rgba(prefix.with_name(f"{case.name}_imgkey_result.png"), result.rgba)
    _save_mask(prefix.with_name(f"{case.name}_imgkey_alpha.png"), result.alpha)
    _save_rgb(prefix.with_name(f"{case.name}_alpha_diff_heatmap.png"), _geometric_alpha_diff_heatmap(result.alpha, case.expected_alpha))
    _save_rgb(prefix.with_name(f"{case.name}_error_overlay.png"), _geometric_error_overlay(case.source_rgb, result.alpha, case.expected_alpha))

    for background_name, color in (("black", (0, 0, 0)), ("white", (255, 255, 255)), ("gray", (128, 128, 128)), ("checker", None)):
        actual = checkerboard_composite(result.rgba) if color is None else _solid_composite(result.rgba, color)
        expected = checkerboard_composite(case.expected_rgba) if color is None else _solid_composite(case.expected_rgba, color)
        _save_rgb(prefix.with_name(f"{case.name}_imgkey_on_{background_name}.png"), actual)
        _save_rgb(prefix.with_name(f"{case.name}_expected_on_{background_name}.png"), expected)
        _save_rgb(prefix.with_name(f"{case.name}_diff_on_{background_name}.png"), _geometric_rgb_diff_heatmap(actual, expected))
        _save_rgb(prefix.with_name(f"{case.name}_compare_on_{background_name}.png"), np.concatenate([expected, actual], axis=1))


def _geometric_gpu_parity(cases: list[GeometricBenchmarkCase]) -> dict[str, Any]:
    gpu_backend = importlib.import_module("gpu_backend")
    backend_objects = gpu_backend.registered_backends()
    backends = gpu_backend.probe_backends(backends=backend_objects, include_cpu=False, refresh=True)
    selection = gpu_backend.select_backend("Force GPU", {"constant_screen", "rgb_only"}, backends=backend_objects, probed_backends=backends)
    parity: dict[str, Any] = {
        "backend": {"id": selection.as_dict().get("backend"), "name": selection.as_dict().get("backend_name")},
        "availability": {"backends": backends, "selected_backend": selection.as_dict()},
        "status": "skipped",
        "reason": selection.reason,
        "message": selection.message,
        "cases": {},
    }
    if not selection.available:
        return parity

    max_rgba_diff = 0
    max_alpha_diff = 0
    used_case_count = 0
    for case in cases:
        parity_cpu_settings = replace(
            case.settings,
            auto_border_sample=False,
            local_screen_model=False,
            gpu_acceleration="Off",
        )
        parity_gpu_settings = replace(parity_cpu_settings, gpu_acceleration="Force GPU")
        cpu_result = process_key_image(case.source_rgb, parity_cpu_settings)
        gpu_result = process_key_image(case.source_rgb, parity_gpu_settings)
        cpu_metrics = _geometric_case_metrics(case, cpu_result)
        gpu_metrics = _geometric_case_metrics(case, gpu_result)
        diff = np.abs(cpu_result.rgba.astype(np.int16) - gpu_result.rgba.astype(np.int16))
        case_rgba_diff = int(diff.max())
        case_alpha_diff = int(diff[:, :, 3].max())
        gpu_stats = gpu_result.gpu_acceleration or {}
        used_tiles = int(gpu_stats.get("used_tiles", 0)) if isinstance(gpu_stats, dict) else 0
        used_case_count += int(used_tiles > 0)
        max_rgba_diff = max(max_rgba_diff, case_rgba_diff)
        max_alpha_diff = max(max_alpha_diff, case_alpha_diff)
        parity["cases"][case.name] = {
            "max_rgba_diff_vs_cpu": case_rgba_diff,
            "max_alpha_diff_vs_cpu": case_alpha_diff,
            "used_tiles": used_tiles,
            "gpu_acceleration": gpu_stats,
            "metric_deltas_vs_cpu": {
                "alpha_mae": float(gpu_metrics["whole"]["alpha"]["alpha_mae"] - cpu_metrics["whole"]["alpha"]["alpha_mae"]),
                "thin_line_visible_recall": float(gpu_metrics["thin_line_recall"]["visible_recall"] - cpu_metrics["thin_line_recall"]["visible_recall"]),
                "dot_visible_recall": float(gpu_metrics["dot_preservation"]["visible_recall"] - cpu_metrics["dot_preservation"]["visible_recall"]),
            },
        }

    parity.update(
        {
            "status": "measured",
            "reason": None,
            "message": "Geometry CPU/GPU parity completed.",
            "within_tolerance": bool(max_rgba_diff <= 2 and max_alpha_diff <= 1 and used_case_count > 0),
            "max_rgba_diff_vs_cpu": max_rgba_diff,
            "max_alpha_diff_vs_cpu": max_alpha_diff,
            "cases_with_gpu_tiles": used_case_count,
            "interpretation": {
                "rgb_only_mismatch": bool(max_rgba_diff > 2 and max_alpha_diff <= 1 and used_case_count > 0),
                "blocks_cpu_default_scoring": False,
                "blocks_gpu_parity_gate": bool(max_rgba_diff > 2 or max_alpha_diff > 1 or used_case_count <= 0),
                "note": (
                    "Geometry parity is RGB-only outside tolerance; CPU remains the tuning reference, "
                    "but geometry-level GPU parity should not be considered passing. Direct tile parity "
                    "can still pass because it compares one backend kernel against the tile CPU mirror, "
                    "while this benchmark compares the full keyer CPU path against forced GPU transition repair."
                    if max_rgba_diff > 2 and max_alpha_diff <= 1 and used_case_count > 0
                    else "Geometry parity is within tolerance."
                    if max_rgba_diff <= 2 and max_alpha_diff <= 1 and used_case_count > 0
                    else "Geometry GPU parity did not use enough GPU tiles or exceeded alpha tolerance."
                ),
            },
        }
    )
    return parity


def _geometric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    case_records = metrics["cases"]
    case_summaries: dict[str, Any] = {}
    alpha_mae_values: list[float] = []
    thin_values: list[float] = []
    dot_values: list[float] = []
    total_background_leak = 0
    worst_alpha_case = None
    worst_alpha_value = -1.0
    for case_name, record in case_records.items():
        case_metrics = record["metrics"]
        alpha_mae = float(case_metrics["whole"]["alpha"]["alpha_mae"])
        thin_recall = float(case_metrics["thin_line_recall"]["visible_recall"])
        dot_recall = float(case_metrics["dot_preservation"]["visible_recall"])
        leak_pixels = int(case_metrics["background_leak"]["leaking_pixels"])
        alpha_mae_values.append(alpha_mae)
        thin_values.append(thin_recall)
        dot_values.append(dot_recall)
        total_background_leak += leak_pixels
        if alpha_mae > worst_alpha_value:
            worst_alpha_value = alpha_mae
            worst_alpha_case = case_name
        case_summaries[case_name] = {
            "alpha_mae": alpha_mae,
            "alpha_precision": case_metrics["whole"]["alpha"]["alpha_precision"],
            "alpha_recall": case_metrics["whole"]["alpha"]["alpha_recall"],
            "thin_line_visible_recall": thin_recall,
            "dot_visible_recall": dot_recall,
            "background_leaking_pixels": leak_pixels,
            "foreground_core_rgb_mean_abs_error": case_metrics["whole"]["foreground_core_rgb_delta"]["mean_abs_error"],
            "edge_key_residual_p95": case_metrics["edge_key_color_residual"]["p95_positive_excess"],
            "transparent_rgb_max": case_metrics["transparent_rgb_residual"]["max_rgb_when_transparent"],
        }
    return {
        "schema_version": 1,
        "generated_by": "python smoke_test.py --write-geometric-benchmark",
        "artifact_dir": str(GEOMETRIC_BENCHMARK_DIR),
        "case_count": len(case_records),
        "feature_counts": metrics["feature_counts"],
        "aggregate": {
            "alpha_mae_mean": float(np.mean(alpha_mae_values)) if alpha_mae_values else 0.0,
            "alpha_mae_max": float(np.max(alpha_mae_values)) if alpha_mae_values else 0.0,
            "worst_alpha_case": worst_alpha_case,
            "thin_line_visible_recall_min": float(np.min(thin_values)) if thin_values else 1.0,
            "dot_visible_recall_min": float(np.min(dot_values)) if dot_values else 1.0,
            "background_leaking_pixels_total": total_background_leak,
        },
        "gpu_parity": metrics["gpu_parity"],
        "cases": case_summaries,
    }


def run_geometric_benchmark_gate_tests() -> None:
    cases = geometric_benchmark_cases()
    failures: list[str] = []
    for case in cases:
        result = process_key_image(case.source_rgb, case.settings)
        metrics = _geometric_case_metrics(case, result)
        thin_recall = float(metrics["thin_line_recall"]["visible_recall"])
        dot_recall = float(metrics["dot_preservation"]["visible_recall"])
        band_alpha = metrics["features"]["transparency_bands"]["alpha"]
        band_alpha_mae_u8 = float(band_alpha["alpha_mae_u8"])
        background_leak_pixels = int(metrics["background_leak"]["leaking_pixels"])
        edge_residual_p95 = float(metrics["edge_key_color_residual"]["p95_positive_excess"])
        transparent_rgb_max = int(metrics["transparent_rgb_residual"]["max_rgb_when_transparent"])
        if thin_recall < 0.75:
            failures.append(f"{case.name}: thin-line recall {thin_recall:.3f} < 0.75")
        if dot_recall < 0.75:
            failures.append(f"{case.name}: dot preservation {dot_recall:.3f} < 0.75")
        if band_alpha_mae_u8 > 100.0:
            failures.append(f"{case.name}: transparency-band alpha MAE {band_alpha_mae_u8:.2f} > 100.0")
        if background_leak_pixels != 0:
            failures.append(f"{case.name}: background leak {background_leak_pixels} pixels != 0")
        if edge_residual_p95 > 2.0:
            failures.append(f"{case.name}: edge key residual p95 {edge_residual_p95:.2f} > 2.0")
        if transparent_rgb_max != 0:
            failures.append(f"{case.name}: transparent RGB max {transparent_rgb_max} != 0")

    assert not failures, "geometric benchmark gates failed:\n" + "\n".join(failures)

    gpu_parity = _geometric_gpu_parity(cases)
    if gpu_parity.get("status") == "skipped":
        print(f"geometry GPU parity skipped: {gpu_parity.get('reason')} - {gpu_parity.get('message')}")
    else:
        assert gpu_parity.get("within_tolerance"), (
            "geometry GPU parity max diff too high: "
            f"rgba={gpu_parity.get('max_rgba_diff_vs_cpu')} alpha={gpu_parity.get('max_alpha_diff_vs_cpu')}"
        )


def _score_high_is_good(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _score_low_is_good(value: float, full_penalty_at: float) -> float:
    if full_penalty_at <= 0:
        return 1.0 if value <= 0 else 0.0
    return float(1.0 - np.clip(value / full_penalty_at, 0.0, 1.0))


def _geometric_tuning_case_score(case_metrics: dict[str, Any]) -> dict[str, Any]:
    alpha_metrics = case_metrics["whole"]["alpha"]
    background_leak = case_metrics["background_leak"]
    edge_residual = case_metrics["edge_key_color_residual"]
    core_rgb = case_metrics["whole"]["foreground_core_rgb_delta"]
    thin = case_metrics["thin_line_recall"]
    dots = case_metrics["dot_preservation"]

    background_count = max(int(background_leak["count"]), 1)
    background_leak_rate = float(background_leak["leaking_pixels"]) / float(background_count)
    components = {
        "thin_line_visible_recall": {
            "value": float(thin["visible_recall"]),
            "score": _score_high_is_good(float(thin["visible_recall"])),
        },
        "thin_line_alpha_ratio": {
            "value": float(thin["mean_alpha_ratio"]),
            "score": _score_high_is_good(float(thin["mean_alpha_ratio"])),
        },
        "dot_visible_recall": {
            "value": float(dots["visible_recall"]),
            "score": _score_high_is_good(float(dots["visible_recall"])),
        },
        "background_leak": {
            "value": background_leak_rate,
            "pixels": int(background_leak["leaking_pixels"]),
            "count": int(background_leak["count"]),
            "score": _score_low_is_good(background_leak_rate, 0.01),
        },
        "foreground_loss": {
            "value": float(alpha_metrics["false_background_loss_rate"]),
            "pixels": int(alpha_metrics["false_background_loss_pixels"]),
            "score": _score_low_is_good(float(alpha_metrics["false_background_loss_rate"]), 0.05),
        },
        "edge_key_residual": {
            "value": float(edge_residual["p95_positive_excess"]),
            "score": _score_low_is_good(float(edge_residual["p95_positive_excess"]), 64.0),
        },
        "foreground_core_rgb": {
            "value": float(core_rgb["mean_abs_error"]),
            "score": _score_low_is_good(float(core_rgb["mean_abs_error"]), 12.0),
        },
        "alpha_mae": {
            "value": float(alpha_metrics["alpha_mae"]),
            "score": _score_low_is_good(float(alpha_metrics["alpha_mae"]), 0.08),
        },
    }
    weighted_total = 0.0
    weight_sum = 0.0
    for name, component in components.items():
        weight = float(GEOMETRIC_TUNING_SCORE_WEIGHTS[name])
        component["weight"] = weight
        weighted_total += float(component["score"]) * weight
        weight_sum += weight
    return {
        "weighted_score": float(100.0 * weighted_total / weight_sum) if weight_sum else 0.0,
        "components": components,
    }


def _geometric_tuning_case_digest(case_metrics: dict[str, Any]) -> dict[str, Any]:
    alpha_metrics = case_metrics["whole"]["alpha"]
    background_leak = case_metrics["background_leak"]
    return {
        "alpha_mae": float(alpha_metrics["alpha_mae"]),
        "alpha_precision": float(alpha_metrics["alpha_precision"]),
        "alpha_recall": float(alpha_metrics["alpha_recall"]),
        "foreground_loss_pixels": int(alpha_metrics["false_background_loss_pixels"]),
        "foreground_loss_rate": float(alpha_metrics["false_background_loss_rate"]),
        "background_leaking_pixels": int(background_leak["leaking_pixels"]),
        "background_count": int(background_leak["count"]),
        "background_mean_alpha": float(background_leak["mean_alpha"]),
        "thin_line_visible_recall": float(case_metrics["thin_line_recall"]["visible_recall"]),
        "thin_line_alpha_ratio": float(case_metrics["thin_line_recall"]["mean_alpha_ratio"]),
        "dot_visible_recall": float(case_metrics["dot_preservation"]["visible_recall"]),
        "dot_alpha_ratio": float(case_metrics["dot_preservation"]["mean_alpha_ratio"]),
        "foreground_core_rgb_mean_abs_error": float(case_metrics["whole"]["foreground_core_rgb_delta"]["mean_abs_error"]),
        "foreground_core_rgb_max_abs_error": int(case_metrics["whole"]["foreground_core_rgb_delta"]["max_abs_error"]),
        "edge_key_residual_p95": float(case_metrics["edge_key_color_residual"]["p95_positive_excess"]),
        "edge_key_residual_max": int(case_metrics["edge_key_color_residual"]["max_positive_excess"]),
        "transparent_rgb_max": int(case_metrics["transparent_rgb_residual"]["max_rgb_when_transparent"]),
    }


def _geometric_key_family(case: GeometricBenchmarkCase) -> str:
    if case.key_color == (30, 80, 235):
        return "blue"
    if case.key_color == (0, 220, 50):
        return "green"
    if case.key_color == (0, 190, 210):
        return "cyan"
    return "custom"


def _profile_for_key(profile_name: str, key_color: tuple[int, int, int]) -> GeometricTuningProfile:
    profiles = {profile.name: profile for profile in geometric_tuning_profiles(key_color)}
    return profiles[profile_name]


def _geometric_cases_for_profile(cases: list[GeometricBenchmarkCase], profile_name: str) -> list[GeometricBenchmarkCase]:
    return [
        replace(case, settings=_profile_for_key(profile_name, case.key_color).settings)
        for case in cases
    ]


def _summarize_tuning_profile(profile_name: str, cases: list[GeometricBenchmarkCase]) -> dict[str, Any]:
    reference_profile = _profile_for_key(profile_name, (30, 80, 235))
    case_records: dict[str, Any] = {}
    score_values: list[float] = []
    thin_values: list[float] = []
    dot_values: list[float] = []
    alpha_values: list[float] = []
    core_values: list[float] = []
    background_leaks = 0
    family_scores: dict[str, list[float]] = {}

    for case in cases:
        profile = _profile_for_key(profile_name, case.key_color)
        candidate_case = replace(case, settings=profile.settings)
        result = process_key_image(case.source_rgb, profile.settings)
        metrics = _geometric_case_metrics(candidate_case, result)
        score = _geometric_tuning_case_score(metrics)
        digest = _geometric_tuning_case_digest(metrics)
        family = _geometric_key_family(case)
        weighted_score = float(score["weighted_score"])
        score_values.append(weighted_score)
        family_scores.setdefault(family, []).append(weighted_score)
        thin_values.append(float(digest["thin_line_visible_recall"]))
        dot_values.append(float(digest["dot_visible_recall"]))
        alpha_values.append(float(digest["alpha_mae"]))
        core_values.append(float(digest["foreground_core_rgb_mean_abs_error"]))
        background_leaks += int(digest["background_leaking_pixels"])
        case_records[case.name] = {
            "background_name": case.background_name,
            "key_family": family,
            "key_color": list(case.key_color),
            "metrics": digest,
            "score": score,
        }

    return {
        "name": reference_profile.name,
        "label": reference_profile.label,
        "description": reference_profile.description,
        "settings_blue_key": asdict(reference_profile.settings),
        "aggregate": {
            "weighted_score": float(np.mean(score_values)) if score_values else 0.0,
            "weighted_score_min": float(np.min(score_values)) if score_values else 0.0,
            "thin_line_visible_recall_min": float(np.min(thin_values)) if thin_values else 1.0,
            "dot_visible_recall_min": float(np.min(dot_values)) if dot_values else 1.0,
            "alpha_mae_mean": float(np.mean(alpha_values)) if alpha_values else 0.0,
            "alpha_mae_max": float(np.max(alpha_values)) if alpha_values else 0.0,
            "foreground_core_rgb_mean_abs_error_max": float(np.max(core_values)) if core_values else 0.0,
            "background_leaking_pixels_total": int(background_leaks),
            "families": {
                family: {
                    "weighted_score": float(np.mean(values)),
                    "weighted_score_min": float(np.min(values)),
                    "case_count": len(values),
                }
                for family, values in family_scores.items()
            },
        },
        "cases": case_records,
    }


def _geometric_legacy_tuning_fixtures() -> list[DiagnosticFixture]:
    fixtures = [
        blue_gradient_screen_fixture(),
        green_gradient_screen_fixture(),
        _cyanish_screen_fixture(),
        hair_lines_fixture(),
        semi_transparent_glass_fixture(),
        white_gray_black_composite_fixture(),
    ]
    return [fixture for fixture in fixtures if fixture.expected_alpha is not None]


def _legacy_tuning_metrics(fixture: DiagnosticFixture, settings: KeySettings) -> dict[str, Any]:
    result = _process_fixture_result(fixture, settings, include_debug=False)
    known_background, foreground_core, soft_edge = _fixture_masks(fixture)
    if fixture.expected_alpha is None:
        alpha_mae = 0.0
        expected_for_detail = None
    else:
        expected_u8 = np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        alpha_mae = float(np.mean(np.abs(result.alpha.astype(np.int16) - expected_u8.astype(np.int16))) / 255.0)
        detail_mask = soft_edge if np.any(soft_edge) else fixture.expected_alpha > 0.0
        expected_for_detail = np.where(detail_mask, fixture.expected_alpha, 0.0)
    detail = alpha_detail_recall(expected_for_detail, result.alpha)
    background = background_alpha_leak(result.alpha, known_background)
    core_delta = foreground_core_rgb_delta(fixture, result)
    edge_residual = edge_key_residual(result.rgba, settings.key_color, soft_edge)
    return {
        "alpha_mae": alpha_mae,
        "detail_visible_recall": float(detail["visible_recall"]),
        "detail_alpha_ratio": float(detail["mean_alpha_ratio"]),
        "background_leaking_pixels": int(background["leaking_pixels"]),
        "background_count": int(background["count"]),
        "foreground_core_rgb_mean_delta": float(core_delta["mean_delta"]),
        "foreground_core_rgb_max_delta": int(core_delta["max_delta"]),
        "edge_key_residual_p95": float(edge_residual["p95_positive_excess"]),
    }


def _legacy_tuning_summary(profile_names: list[str]) -> dict[str, Any]:
    fixtures = _geometric_legacy_tuning_fixtures()
    baseline: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    for profile_name in profile_names:
        records: dict[str, Any] = {}
        regressions: list[dict[str, Any]] = []
        for fixture in fixtures:
            profile = _profile_for_key(profile_name, fixture.settings.key_color)
            metrics = _legacy_tuning_metrics(fixture, profile.settings)
            records[fixture.name] = metrics
            if profile_name == "current_app_default":
                baseline[fixture.name] = metrics
                continue
            base = baseline[fixture.name]
            leak_slack = max(
                int(GEOMETRIC_TUNING_PROMOTION_TOLERANCES["background_leak_pixel_slack"]),
                int(round(float(base["background_count"]) * GEOMETRIC_TUNING_PROMOTION_TOLERANCES["background_leak_rate_slack"])),
            )
            if float(metrics["alpha_mae"]) > float(base["alpha_mae"]) + GEOMETRIC_TUNING_PROMOTION_TOLERANCES["legacy_alpha_mae_slack"]:
                regressions.append({"fixture": fixture.name, "metric": "alpha_mae", "baseline": base["alpha_mae"], "candidate": metrics["alpha_mae"]})
            if int(metrics["background_leaking_pixels"]) > int(base["background_leaking_pixels"]) + leak_slack:
                regressions.append({"fixture": fixture.name, "metric": "background_leak", "baseline": base["background_leaking_pixels"], "candidate": metrics["background_leaking_pixels"], "slack": leak_slack})
            if float(metrics["detail_visible_recall"]) + GEOMETRIC_TUNING_PROMOTION_TOLERANCES["legacy_detail_recall_slack"] < float(base["detail_visible_recall"]):
                regressions.append({"fixture": fixture.name, "metric": "detail_visible_recall", "baseline": base["detail_visible_recall"], "candidate": metrics["detail_visible_recall"]})
            core_limit = min(
                GEOMETRIC_TUNING_PROMOTION_TOLERANCES["foreground_core_rgb_mean_abs_error_max"],
                float(base["foreground_core_rgb_mean_delta"]) + GEOMETRIC_TUNING_PROMOTION_TOLERANCES["legacy_core_rgb_delta_slack"],
            )
            if float(metrics["foreground_core_rgb_mean_delta"]) > core_limit:
                regressions.append({"fixture": fixture.name, "metric": "foreground_core_rgb_mean_delta", "limit": core_limit, "candidate": metrics["foreground_core_rgb_mean_delta"]})
        profiles[profile_name] = {"fixtures": records, "regressions": regressions}
    return {
        "fixture_names": [fixture.name for fixture in fixtures],
        "profiles": profiles,
    }


def _geometric_promotion_checks(profile_summaries: dict[str, Any], legacy_summary: dict[str, Any]) -> dict[str, Any]:
    baseline = profile_summaries["current_app_default"]
    baseline_score = float(baseline["aggregate"]["weighted_score"])
    checks: dict[str, Any] = {}
    for profile_name, summary in profile_summaries.items():
        score = float(summary["aggregate"]["weighted_score"])
        improvement_fraction = (score - baseline_score) / max(abs(baseline_score), 1e-9)
        detail_regressions: list[dict[str, Any]] = []
        leak_regressions: list[dict[str, Any]] = []
        foreground_loss_regressions: list[dict[str, Any]] = []
        core_blocks: list[dict[str, Any]] = []
        if profile_name != "current_app_default":
            for case_name, record in summary["cases"].items():
                candidate_metrics = record["metrics"]
                baseline_metrics = baseline["cases"][case_name]["metrics"]
                if float(candidate_metrics["thin_line_visible_recall"]) + GEOMETRIC_TUNING_PROMOTION_TOLERANCES["detail_recall_epsilon"] < float(baseline_metrics["thin_line_visible_recall"]):
                    detail_regressions.append({"case": case_name, "metric": "thin_line_visible_recall", "baseline": baseline_metrics["thin_line_visible_recall"], "candidate": candidate_metrics["thin_line_visible_recall"]})
                if float(candidate_metrics["dot_visible_recall"]) + GEOMETRIC_TUNING_PROMOTION_TOLERANCES["detail_recall_epsilon"] < float(baseline_metrics["dot_visible_recall"]):
                    detail_regressions.append({"case": case_name, "metric": "dot_visible_recall", "baseline": baseline_metrics["dot_visible_recall"], "candidate": candidate_metrics["dot_visible_recall"]})
                leak_slack = max(
                    int(GEOMETRIC_TUNING_PROMOTION_TOLERANCES["background_leak_pixel_slack"]),
                    int(round(float(baseline_metrics["background_count"]) * GEOMETRIC_TUNING_PROMOTION_TOLERANCES["background_leak_rate_slack"])),
                )
                if int(candidate_metrics["background_leaking_pixels"]) > int(baseline_metrics["background_leaking_pixels"]) + leak_slack:
                    leak_regressions.append({"case": case_name, "baseline": baseline_metrics["background_leaking_pixels"], "candidate": candidate_metrics["background_leaking_pixels"], "slack": leak_slack})
                if float(candidate_metrics["foreground_loss_rate"]) > float(baseline_metrics["foreground_loss_rate"]) + GEOMETRIC_TUNING_PROMOTION_TOLERANCES["foreground_loss_rate_slack"]:
                    foreground_loss_regressions.append({"case": case_name, "baseline": baseline_metrics["foreground_loss_rate"], "candidate": candidate_metrics["foreground_loss_rate"]})
                if float(candidate_metrics["foreground_core_rgb_mean_abs_error"]) > GEOMETRIC_TUNING_PROMOTION_TOLERANCES["foreground_core_rgb_mean_abs_error_max"]:
                    core_blocks.append({"case": case_name, "candidate": candidate_metrics["foreground_core_rgb_mean_abs_error"], "limit": GEOMETRIC_TUNING_PROMOTION_TOLERANCES["foreground_core_rgb_mean_abs_error_max"]})
        legacy_regressions = legacy_summary["profiles"].get(profile_name, {}).get("regressions", [])
        eligible = bool(
            profile_name != "current_app_default"
            and improvement_fraction >= GEOMETRIC_TUNING_PROMOTION_TOLERANCES["minimum_score_improvement_fraction"]
            and not detail_regressions
            and not leak_regressions
            and not foreground_loss_regressions
            and not core_blocks
            and not legacy_regressions
        )
        checks[profile_name] = {
            "score_improvement_fraction": float(improvement_fraction),
            "score_improvement_percent": float(improvement_fraction * 100.0),
            "meets_score_improvement": bool(improvement_fraction >= GEOMETRIC_TUNING_PROMOTION_TOLERANCES["minimum_score_improvement_fraction"]),
            "detail_regressions": detail_regressions,
            "background_leak_regressions": leak_regressions,
            "foreground_loss_regressions": foreground_loss_regressions,
            "foreground_core_rgb_blocks": core_blocks,
            "legacy_regressions": legacy_regressions,
            "eligible_for_global_default": eligible,
        }
    return checks


def _geometric_tuning_recommendations(
    profile_summaries: dict[str, Any],
    promotion_checks: dict[str, Any],
    profile_gpu_parity: dict[str, Any],
) -> dict[str, Any]:
    weighted_winner = max(profile_summaries, key=lambda name: float(profile_summaries[name]["aggregate"]["weighted_score"]))
    eligible = [name for name, check in promotion_checks.items() if check["eligible_for_global_default"]]
    if eligible:
        global_profile = max(eligible, key=lambda name: float(profile_summaries[name]["aggregate"]["weighted_score"]))
    else:
        global_profile = "current_app_default"
    selected_gpu_parity = profile_gpu_parity.get(global_profile, {})
    gpu_blocks_default = bool(selected_gpu_parity.get("interpretation", {}).get("blocks_gpu_parity_gate"))
    current_matches_green_cyan_safe = bool(
        profile_summaries.get("current_app_default", {}).get("settings_blue_key")
        == profile_summaries.get("green_cyan_safe", {}).get("settings_blue_key")
    )
    if eligible:
        global_action = "defer_global_default_until_gpu_geometry_parity_resolved" if gpu_blocks_default else "promote_global_default_candidate"
    elif current_matches_green_cyan_safe:
        global_action = "current_global_default_matches_green_cyan_safe"
    else:
        global_action = "keep_current_global_default"

    blue_scores = {
        name: float(summary["aggregate"]["families"].get("blue", {}).get("weighted_score", 0.0))
        for name, summary in profile_summaries.items()
    }
    strict_blue_winner = bool(blue_scores and blue_scores.get("asset_strict_screenshot", -1.0) >= max(blue_scores.values()) - 1e-9)
    strict_blocked = not bool(promotion_checks["asset_strict_screenshot"]["eligible_for_global_default"])
    named_preset = None
    if strict_blue_winner and strict_blocked:
        named_preset = {
            "action": "recommend_named_preset",
            "preset_name": "Asset Strict",
            "profile": "asset_strict_screenshot",
            "reason": "Strict screenshot profile is strongest on blue/graphic geometry but fails at least one global promotion rule.",
        }

    return {
        "weighted_winner": weighted_winner,
        "global_default": {
            "action": global_action,
            "profile": global_profile,
            "reason": (
                "CPU metrics pass the objective promotion rules, but geometry-level GPU parity is outside tolerance. Resolve or explicitly accept the GPU RGB mismatch before changing the global default."
                if eligible and gpu_blocks_default
                else
                "Candidate passed the objective promotion rules."
                if eligible
                else
                "Current app defaults already match the green/cyan-safe geometric benchmark profile."
                if current_matches_green_cyan_safe
                else "No candidate passed all objective global-default promotion rules; keep the current default for now."
            ),
            "cpu_metric_eligible_profiles": eligible,
            "gpu_parity_profile": global_profile,
        },
        "named_preset": named_preset,
        "blue_family_scores": blue_scores,
    }


def _geometric_tuning_report(summary: dict[str, Any]) -> str:
    lines = [
        "Geometric default tuning sweep",
        "================================",
        f"Generated by: {summary['generated_by']}",
        "",
        "Candidate ranking:",
    ]
    ranked = sorted(
        summary["profiles"].items(),
        key=lambda item: float(item[1]["aggregate"]["weighted_score"]),
        reverse=True,
    )
    for name, profile in ranked:
        aggregate = profile["aggregate"]
        check = summary["promotion_checks"][name]
        lines.append(
            f"- {profile['label']} ({name}): score={aggregate['weighted_score']:.2f}, "
            f"thin_min={aggregate['thin_line_visible_recall_min']:.3f}, "
            f"dot_min={aggregate['dot_visible_recall_min']:.3f}, "
            f"alpha_mae_mean={aggregate['alpha_mae_mean']:.4f}, "
            f"bg_leak_total={aggregate['background_leaking_pixels_total']}, "
            f"core_rgb_max={aggregate['foreground_core_rgb_mean_abs_error_max']:.2f}, "
            f"improvement={check['score_improvement_percent']:.2f}%, "
            f"global_eligible={check['eligible_for_global_default']}"
        )
    recommendation = summary["recommendation"]
    lines.extend(
        [
            "",
            f"Weighted winner: {recommendation['weighted_winner']}",
            f"Global default recommendation: {recommendation['global_default']['action']} -> {recommendation['global_default']['profile']}",
            f"Reason: {recommendation['global_default']['reason']}",
        ]
    )
    if recommendation.get("named_preset"):
        named = recommendation["named_preset"]
        lines.append(f"Named preset recommendation: {named['preset_name']} from {named['profile']} ({named['reason']})")
    parity = summary["geometry_gpu_parity"]
    parity_profile = recommendation["global_default"].get("gpu_parity_profile", recommendation["global_default"]["profile"])
    lines.extend(
        [
            "",
            f"Geometry GPU parity ({parity_profile}): status={parity.get('status')} within_tolerance={parity.get('within_tolerance')} "
            f"max_rgba_diff={parity.get('max_rgba_diff_vs_cpu')} max_alpha_diff={parity.get('max_alpha_diff_vs_cpu')}",
            f"GPU parity note: {parity.get('interpretation', {}).get('note', parity.get('message'))}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_geometric_tuning_summary() -> None:
    GEOMETRIC_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"running geometric default tuning sweep; writing summary to {GEOMETRIC_BENCHMARK_DIR}")
    cases = geometric_benchmark_cases()
    profile_names = [profile.name for profile in geometric_tuning_profiles((30, 80, 235))]
    profiles = {name: _summarize_tuning_profile(name, cases) for name in profile_names}
    legacy_summary = _legacy_tuning_summary(profile_names)
    promotion_checks = _geometric_promotion_checks(profiles, legacy_summary)
    profile_gpu_parity = {
        name: _geometric_gpu_parity(_geometric_cases_for_profile(cases, name))
        for name in profile_names
    }
    recommendation = _geometric_tuning_recommendations(profiles, promotion_checks, profile_gpu_parity)
    selected_gpu_profile = recommendation["global_default"]["gpu_parity_profile"]
    gpu_parity = profile_gpu_parity[selected_gpu_profile]
    summary = {
        "schema_version": 1,
        "generated_by": "python smoke_test.py --tune-geometric-defaults",
        "artifact_dir": str(GEOMETRIC_BENCHMARK_DIR),
        "score_weights": GEOMETRIC_TUNING_SCORE_WEIGHTS,
        "promotion_tolerances": GEOMETRIC_TUNING_PROMOTION_TOLERANCES,
        "profiles": profiles,
        "legacy_checks": legacy_summary,
        "promotion_checks": promotion_checks,
        "recommendation": recommendation,
        "geometry_gpu_parity": gpu_parity,
        "geometry_gpu_parity_by_profile": profile_gpu_parity,
    }
    summary_path = GEOMETRIC_BENCHMARK_DIR / "tuning_summary.json"
    report_path = GEOMETRIC_BENCHMARK_DIR / "tuning_report.txt"
    report = _geometric_tuning_report(summary)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(report.rstrip())
    print(f"wrote geometric tuning summary to {summary_path}")
    print(f"wrote geometric tuning report to {report_path}")


def write_geometric_benchmark() -> None:
    GEOMETRIC_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing geometric benchmark diagnostics to {GEOMETRIC_BENCHMARK_DIR}")
    asset = generate_geometric_benchmark_asset()
    cases = geometric_benchmark_cases()
    _write_geometric_feature_artifacts(asset)
    all_metrics: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "python smoke_test.py --write-geometric-benchmark",
        "asset": asset.name,
        "notes": asset.notes,
        "feature_counts": {name: int(np.count_nonzero(mask)) for name, mask in asset.feature_masks.items()},
        "feature_mask_artifact": "geometric_feature_masks.npz",
        "feature_label_artifact": "geometric_feature_labels.png",
        "cases": {},
        "gpu_parity": None,
    }

    for case in cases:
        result = process_key_image(case.source_rgb, case.settings)
        metrics = _geometric_case_metrics(case, result)
        all_metrics["cases"][case.name] = {
            "background_name": case.background_name,
            "key_color": case.key_color,
            "notes": case.notes,
            "metrics": metrics,
            "artifacts": {
                "source": f"{case.name}_source.png",
                "expected_alpha": f"{case.name}_expected_alpha.png",
                "expected_foreground": f"{case.name}_expected_foreground.png",
                "imgkey_result": f"{case.name}_imgkey_result.png",
                "imgkey_alpha": f"{case.name}_imgkey_alpha.png",
                "alpha_diff_heatmap": f"{case.name}_alpha_diff_heatmap.png",
                "error_overlay": f"{case.name}_error_overlay.png",
            },
        }
        _write_geometric_case_artifacts(case, result)
        print(
            f"geometric {case.name}: alpha_mae={metrics['whole']['alpha']['alpha_mae']:.4f} "
            f"thin_recall={metrics['thin_line_recall']['visible_recall']:.3f} "
            f"dot_recall={metrics['dot_preservation']['visible_recall']:.3f} "
            f"bg_leak={metrics['background_leak']['leaking_pixels']}"
        )

    all_metrics["gpu_parity"] = _geometric_gpu_parity(cases)
    summary = _geometric_summary(all_metrics)
    metrics_path = GEOMETRIC_BENCHMARK_DIR / "metrics.json"
    summary_path = GEOMETRIC_BENCHMARK_DIR / "summary.json"
    metrics_path.write_text(json.dumps(_json_ready(all_metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote geometric benchmark metrics to {metrics_path}")
    print(f"wrote geometric benchmark summary to {summary_path}")


def run_app_ui_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app_module = importlib.import_module("app")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing app for UI probe must not import heavy runtimes: {after_import - before}"

    from PySide6.QtWidgets import QApplication

    created_app = QApplication.instance() is None
    qt_app = QApplication.instance() or QApplication(["imgkey-ui-probe"])
    window = app_module.MainWindow()
    try:
        assert window.gpu_status_action.text() == "GPU Status"
        assert window.gpu_status_btn.text() == "GPU Status"
        assert [window.gpu_acceleration.itemText(i) for i in range(window.gpu_acceleration.count())] == ["Auto", "Off", "Force GPU"]
        assert window.output_mode.findText("Classical") >= 0
        assert window.output_mode.findText("Imported Matte") >= 0
        assert window.alpha_hint_mask is None
        assert KeySettings().gpu_acceleration == "Off", "base KeySettings default should stay conservative for library callers"
        defaults = app_module.APP_DEFAULT_SETTINGS
        assert defaults.gpu_acceleration == "Auto", "APP default should auto-enable native GPU acceleration with CPU fallback"
        promoted_expected = {
            "tolerance": 0.26,
            "softness": 0.02,
            "clip_background": 0.95,
            "clip_foreground": 0.08,
            "matte_gamma": 1.60,
            "core_strength": 0.45,
            "edge_refine_radius": 24,
            "edge_softness": 0.04,
            "erode_expand": -4,
            "despill": 0.80,
            "decontaminate": 0.70,
            "luminance_restore": 0.85,
            "luminance_protect": 0.85,
            "fringe_remove": 0.85,
            "edge_color_repair": 0.80,
            "inner_color_pull": 0.60,
            "fringe_band_radius": 5,
            "alpha_recover_strength": 0.90,
            "key_vector_despill": 0.85,
            "foreground_reference_pull": 0.75,
            "screen_cleanup_strength": 1.00,
            "screen_cleanup_similarity": 8,
        }
        for attr, expected in promoted_expected.items():
            actual = getattr(defaults, attr)
            if isinstance(expected, float):
                assert abs(float(actual) - expected) < 1e-9, f"APP default {attr} mismatch: {actual} != {expected}"
            else:
                assert actual == expected, f"APP default {attr} mismatch: {actual} != {expected}"
        assert window.gpu_acceleration.currentText() == defaults.gpu_acceleration
        assert "Auto" in window.gpu_probe_status.text(), "initial GPU status label should reflect Auto default"
        assert window.transition_unmix.text() == "Transition Unmix"
        assert window.transition_unmix.isChecked() is bool(defaults.transition_unmix)
        assert abs(float(window.alpha_recover.value()) - defaults.alpha_recover_strength) < 1e-9
        assert abs(float(window.key_vector_despill.value()) - defaults.key_vector_despill) < 1e-9
        assert abs(float(window.foreground_reference_pull.value()) - defaults.foreground_reference_pull) < 1e-9
        ui_settings = window.current_settings()
        assert ui_settings.transition_unmix is bool(defaults.transition_unmix)
        for attr, expected in promoted_expected.items():
            actual = getattr(ui_settings, attr)
            if isinstance(expected, float):
                assert abs(float(actual) - expected) < 1e-9, f"UI default {attr} mismatch: {actual} != {expected}"
            else:
                assert actual == expected, f"UI default {attr} mismatch: {actual} != {expected}"
        assert abs(ui_settings.alpha_recover_strength - defaults.alpha_recover_strength) < 1e-9
        assert abs(ui_settings.key_vector_despill - defaults.key_vector_despill) < 1e-9
        assert abs(ui_settings.foreground_reference_pull - defaults.foreground_reference_pull) < 1e-9
        assert ui_settings.gpu_acceleration == defaults.gpu_acceleration

        window.transition_unmix.setChecked(False)
        window.alpha_recover.set_value(0.0, emit=False)
        window.key_vector_despill.set_value(0.0, emit=False)
        window.foreground_reference_pull.set_value(0.0, emit=False)
        window.gpu_acceleration.setCurrentText("Off")
        window.apply_preset("High Accuracy")
        ui_settings = window.current_settings()
        assert ui_settings.transition_unmix is bool(defaults.transition_unmix)
        assert abs(ui_settings.alpha_recover_strength - defaults.alpha_recover_strength) < 1e-9
        assert abs(ui_settings.key_vector_despill - defaults.key_vector_despill) < 1e-9
        assert abs(ui_settings.foreground_reference_pull - defaults.foreground_reference_pull) < 1e-9
        assert window.gpu_acceleration.currentText() == defaults.gpu_acceleration == "Auto"
        assert ui_settings.gpu_acceleration == "Auto"
        assert "Auto" in window.gpu_probe_status.text(), "High Accuracy reset should restore Auto GPU status"

        window.full_rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        before_zoom = window.canvas.transform().m11()
        generation = window._preview_generation
        window.transition_unmix.setChecked(False)
        assert window._preview_generation == generation + 1, "transition toggle must schedule preview"
        assert window.canvas.transform().m11() == before_zoom, "transition toggle must not reset viewer zoom"
        assert not window.alpha_recover.isEnabled(), "transition sliders should disable when transition unmix is off"
        window._preview_timer.stop()
        window.transition_unmix.setChecked(True)
        window._preview_timer.stop()
        generation = window._preview_generation
        window.alpha_recover.set_value(0.84)
        assert window._preview_generation == generation + 1, "transition slider must schedule preview"
        assert window.canvas.transform().m11() == before_zoom, "transition slider must not reset viewer zoom"
        window._preview_timer.stop()
        generation = window._preview_generation
        window.gpu_acceleration.setCurrentText("Off")
        window._preview_timer.stop()
        generation = window._preview_generation
        window.gpu_acceleration.setCurrentText("Auto")
        assert window._preview_generation == generation + 1, "GPU acceleration mode change must schedule preview"
        assert "Auto" in window.gpu_probe_status.text(), "GPU status label should reflect selected mode"
        assert window.current_settings().gpu_acceleration == "Auto"
        assert window._message_mentions_gpu_backend("compact CUDA DLL unavailable")
        window._preview_timer.stop()
        window.full_rgb = None
        forbidden = ("Bi" + "RefNet", "Corridor" + "Key", "A" + "I Hint", "Hy" + "brid" + " Bi" + "RefNet")
        for phrase in forbidden:
            assert all(phrase not in mode for mode in app_module.VIEW_MODES), f"removed phrase leaked into view modes: {phrase}"
            assert window.output_mode.findText(phrase) < 0, f"removed phrase leaked into output modes: {phrase}"
            assert phrase not in window.windowTitle(), f"removed phrase leaked into window title: {phrase}"

        assert window.current_settings().mode == "GraphicExact", "default output must use the classical keyer"
        alpha_hint = window._processing_alpha_input(window.current_settings(), (6, 8))
        assert alpha_hint is None, "Classical output mode must not pass imported mattes"

        imported = np.full((6, 8), 255, dtype=np.uint8)
        window.alpha_hint_mask = imported
        window._sync_imported_matte_status("manual.png")
        window.output_mode.setCurrentText("Imported Matte")
        assert window.current_settings().mode == "ImportedMatte", "imported matte output mode must drive imported matte guidance"
        alpha_hint = window._processing_alpha_input(window.current_settings(), (6, 8))
        assert alpha_hint is not None and alpha_hint.shape == (6, 8), "Imported Matte mode must pass the loaded matte"
        assert "Imported matte: loaded" in window.alpha_hint_status.text()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()
        if created_app:
            qt_app.quit()

    after_probe = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_probe == before, f"UI probe must not import heavy runtimes: {after_probe - before}"


def _cyanish_screen_fixture() -> DiagnosticFixture:
    h, w = 360, 520
    x_grad = np.linspace(-18, 20, w, dtype=np.float32).reshape(1, w)
    y_grad = np.linspace(-10, 12, h, dtype=np.float32).reshape(h, 1)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = np.clip(6 + y_grad * 0.10, 0, 255).astype(np.uint8)
    background[:, :, 1] = np.clip(184 + x_grad * 0.55 + y_grad * 0.28, 0, 255).astype(np.uint8)
    background[:, :, 2] = np.clip(204 + x_grad * 0.65 - y_grad * 0.18, 0, 255).astype(np.uint8)
    alpha = _disc_alpha(h, w, 112, feather=9.0)
    foreground = (226, 166, 108)
    known_background, foreground_core, soft_edge = _masks_from_alpha(alpha)
    return DiagnosticFixture(
        name="cyanish_gradient_screen",
        rgb=_composite_rgb(background, foreground, alpha),
        settings=KeySettings(key_color=(0, 190, 210), auto_border_sample=True, edge_refine_radius=5, fringe_band_radius=3),
        notes="v6 diagnostic: cyan-ish key screen with mild uneven lighting.",
        known_background_mask=known_background,
        foreground_core_mask=foreground_core,
        soft_edge_mask=soft_edge,
        expected_alpha=alpha,
        expected_foreground_rgb=foreground,
    )


def _hard_disc_masks(shape: tuple[int, int], radius: float, bg_radius: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    foreground = _disc_alpha(h, w, radius).astype(bool)
    core = _disc_alpha(h, w, max(1.0, radius * 0.70)).astype(bool)
    background = ~_disc_alpha(h, w, bg_radius if bg_radius is not None else radius + 28).astype(bool)
    alpha_u8 = foreground.astype(np.uint8) * 255
    return alpha_u8, background, core


def run_v6_screen_analysis_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    screen_analysis = importlib.import_module("screen_analysis")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing screen_analysis must not import heavy runtimes: {after_import - before}"

    cases: list[tuple[DiagnosticFixture, np.ndarray, np.ndarray, np.ndarray, str]] = []
    green = green_flat_fixture()
    alpha_u8, bg_mask, fg_mask = _hard_disc_masks(green.rgb.shape[:2], 180, 220)
    cases.append((green, alpha_u8, bg_mask, fg_mask, "green"))

    blue = blue_flat_fixture()
    alpha_u8, bg_mask, fg_mask = _hard_disc_masks(blue.rgb.shape[:2], 150, 190)
    cases.append((blue, alpha_u8, bg_mask, fg_mask, "blue"))

    cyan = _cyanish_screen_fixture()
    cases.append((
        cyan,
        np.rint(cyan.expected_alpha * 255.0).astype(np.uint8),
        cyan.known_background_mask.astype(bool),
        cyan.foreground_core_mask.astype(bool),
        "cyan",
    ))

    uneven = uneven_gradient_fixture()
    alpha_u8, bg_mask, fg_mask = _hard_disc_masks(uneven.rgb.shape[:2], 175, 215)
    cases.append((uneven, alpha_u8, bg_mask, fg_mask, "green"))

    for fixture, alpha_u8, bg_mask, fg_mask, family in cases:
        analysis = screen_analysis.analyze_screen(
            fixture.rgb,
            alpha_u8,
            background_mask=bg_mask,
            settings=fixture.settings,
            max_full_res_screen_plate_pixels=0,
            low_res_max_side=96,
        )
        h, w = fixture.rgb.shape[:2]
        for name in (
            "screen_probability",
            "screen_distance",
            "spill_probability",
            "classical_confidence",
            "edge_mask",
            "fringe_mask",
        ):
            arr = getattr(analysis, name)
            assert arr.shape == (h, w), f"{fixture.name}: {name} shape mismatch"
            assert arr.dtype == np.uint8, f"{fixture.name}: {name} must be uint8"
        screen = analysis.screen_color_rgb
        if family == "green":
            assert screen[1] > screen[0] + 90 and screen[1] > screen[2] + 90, f"{fixture.name}: expected green screen, got {screen}"
        elif family == "blue":
            assert screen[2] > screen[0] + 90 and screen[2] > screen[1] + 90, f"{fixture.name}: expected blue screen, got {screen}"
        else:
            assert screen[1] > 130 and screen[2] > 145 and screen[0] < 60, f"{fixture.name}: expected cyan-ish screen, got {screen}"

        bg_prob = float(np.median(analysis.screen_probability[bg_mask]))
        fg_prob = float(np.median(analysis.screen_probability[fg_mask]))
        bg_dist = float(np.median(analysis.screen_distance[bg_mask]))
        fg_dist = float(np.median(analysis.screen_distance[fg_mask]))
        assert bg_prob >= 205.0, f"{fixture.name}: background screen probability too low: {bg_prob:.1f}"
        assert fg_prob <= 120.0, f"{fixture.name}: foreground screen probability too high: {fg_prob:.1f}"
        assert bg_dist <= 85.0, f"{fixture.name}: background screen distance too high: {bg_dist:.1f}"
        assert fg_dist >= 120.0, f"{fixture.name}: foreground screen distance too low: {fg_dist:.1f}"
        assert analysis.classical_confidence[bg_mask].mean() >= 220.0, f"{fixture.name}: confident background should stay high confidence"
        assert np.count_nonzero(analysis.edge_mask) > 0, f"{fixture.name}: edge mask should not be empty"
        if np.any((alpha_u8 > 2) & (alpha_u8 < 253)):
            assert np.count_nonzero(analysis.fringe_mask) > 0, f"{fixture.name}: soft-alpha fringe mask should not be empty"

        plate = analysis.screen_plate_rgb
        assert not plate.is_full_res_retained, f"{fixture.name}: cap=0 must avoid full-resolution plate retention"
        assert plate.low_res_rgb is not None and plate.low_res_rgb.dtype == np.uint8, f"{fixture.name}: low-res plate must be uint8"
        tile = plate.resolve((0, 0, min(32, w), min(32, h)))
        assert tile.shape == (min(32, h), min(32, w), 3) and tile.dtype == np.uint8, f"{fixture.name}: plate resolver tile mismatch"
        debug = analysis.debug_images()
        for key in ("screen_color", "screen_probability", "screen_distance", "screen_plate", "spill_probability", "edge_mask", "fringe_mask"):
            assert key in debug, f"{fixture.name}: missing debug image {key}"

    uneven_analysis = screen_analysis.analyze_screen(
        uneven.rgb,
        cases[-1][1],
        background_mask=cases[-1][2],
        settings=uneven.settings,
        max_full_res_screen_plate_pixels=0,
        low_res_max_side=96,
    )
    h, w = uneven.rgb.shape[:2]
    left = uneven_analysis.screen_plate_rgb.resolve((0, 0, 64, h))[:, :, 1]
    right = uneven_analysis.screen_plate_rgb.resolve((w - 64, 0, w, h))[:, :, 1]
    assert float(np.median(right)) - float(np.median(left)) >= 18.0, "screen plate resolver should preserve uneven green lighting"

    large_rgb = np.zeros((900, 1300, 3), dtype=np.uint8)
    large_rgb[:, :] = (0, 220, 45)
    large_candidates = np.ones(large_rgb.shape[:2], dtype=bool)
    large_plate = screen_analysis.build_screen_plate_rgb(
        large_rgb,
        large_candidates,
        (0, 220, 45),
        max_full_res_pixels=0,
        low_res_max_side=64,
    )
    assert large_plate.full_res_rgb is None, "large/cap-zero screen plate must not retain full HxWx3 RGB"
    assert large_plate.low_res_rgb is not None and max(large_plate.low_res_rgb.shape[:2]) <= 64, "large screen plate must stay low-res bounded"

    # Screen-analysis maps stay standalone; export/low-memory results must not
    # grow retained debug-map fields.
    low_memory = process_key_image(green.rgb, green.settings, include_debug=False)
    for name in ("screen_distance", "spill_probability", "classical_confidence", "screen_plate_rgb"):
        assert not hasattr(low_memory, name), f"low-memory export result should not retain {name} in Phase 6"

    after_tests = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_tests == before, f"screen analysis tests must not import heavy runtimes: {after_tests - before}"


def run_import_compile_tests() -> None:
    sources = [
        Path("app.py"),
        Path("keyer.py"),
        Path("smoke_test.py"),
        Path("gpu_backend.py"),
        Path("gpu_accel.py"),
        Path("gpu_runtime.py"),
        Path("native_toolchain.py"),
        Path("vulkan_runtime.py"),
        Path("screen_analysis.py"),
    ]
    engine_dir = Path("imgkey_engine")
    if engine_dir.exists():
        sources.extend(sorted(engine_dir.glob("*.py")))
    ui_dir = Path("ui")
    if ui_dir.exists():
        sources.extend(sorted(ui_dir.glob("*.py")))
    for source in sources:
        py_compile.compile(source, doraise=True)
    importlib.import_module("app")
    importlib.import_module("keyer")
    importlib.import_module("gpu_backend")
    importlib.import_module("gpu_accel")
    importlib.import_module("gpu_runtime")
    importlib.import_module("native_toolchain")
    importlib.import_module("vulkan_runtime")


def run_removed_surface_tests() -> None:
    missing_paths = [
        Path("a" + "i" + "_assist.py"),
        Path("a" + "i" + "_worker.py"),
        Path("a" + "i" + "_backends"),
        Path("ImgKey-GPU-" + "Bi" + "RefNet.spec"),
        Path("requirements-gpu-" + "bi" + "refnet" + "-cu128.txt"),
    ]
    existing = [str(path) for path in missing_paths if path.exists()]
    assert not existing, f"removed runtime files still exist: {existing}"


def _string_expr_value(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_expr_value(node.left) + _string_expr_value(node.right)
    raise AssertionError(f"unsupported string expression in spec: {ast.dump(node)}")


def _spec_string_list_assignment(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        assert isinstance(node.value, ast.List), f"{name} in {path} must be a list"
        return [_string_expr_value(item) for item in node.value.elts]
    raise AssertionError(f"{name} assignment not found in {path}")


def run_packaging_flavor_tests() -> None:
    default_excludes = set(_spec_string_list_assignment(Path("ImgKey.spec"), "DEFAULT_RUNTIME_EXCLUDES"))
    gpu_excludes = set(_spec_string_list_assignment(Path("ImgKey-GPU.spec"), "GPU_RUNTIME_EXCLUDES"))
    shared_optional_excludes = {
        "torch",
        "torchvision",
        "torchaudio",
        "torchtext",
        "triton",
        "nvidia",
        "trans" + "formers",
        "timm",
        "kornia",
        "einops",
        "accelerate",
        "hugging" + "face_hub",
        "safe" + "tensors",
        "skimage",
        "diff" + "users",
        "peft",
        "tokenizers",
        "sentencepiece",
        "tensorflow",
        "keras",
        "jax",
        "jaxlib",
        "flax",
        "ultralytics",
        "onnx",
        "onnxruntime",
        "onnxruntime_gpu",
        "pymatting",
        "scipy",
        "numba",
        "corridor" + "key",
        "Corridor" + "Key",
    }
    assert {"torch", "nvidia"}.issubset(default_excludes), "primary spec must exclude CUDA/runtime package stacks"
    assert not (shared_optional_excludes - default_excludes), f"default spec missing excludes: {sorted(shared_optional_excludes - default_excludes)}"
    assert {"torch", "nvidia", "cupy", "pycuda", "pyopencl"}.issubset(gpu_excludes), "GPU spec must exclude Python GPU package stacks"
    assert not (shared_optional_excludes - gpu_excludes), f"GPU spec missing excludes: {sorted(shared_optional_excludes - gpu_excludes)}"

    default_spec_text = Path("ImgKey.spec").read_text(encoding="utf-8")
    assert "imgkey_gpu.dll" in default_spec_text, "primary spec must bundle the native D3D12 GPU DLL"
    assert "native/imgkey_gpu/build.ps1 -Clean" in default_spec_text, "primary spec must document the native D3D12 build prerequisite"
    assert "native_gpu_binaries()" in default_spec_text, "primary spec must use explicit native GPU binary collection"
    assert "collect_dynamic_libs" not in default_spec_text, "primary spec must not collect Python GPU package libraries"
    assert "imgkey_cuda.dll" not in default_spec_text, "primary ImgKey.exe must not bundle the legacy CUDA compatibility DLL"
    assert "'gpu_backend'" in default_spec_text and "'gpu_runtime'" in default_spec_text, "primary spec must include backend probe helper modules"
    assert "'torch'" not in default_spec_text.partition("hiddenimports=")[2].partition("]")[0], "primary hidden imports must not include torch"
    assert "name='ImgKey'" in default_spec_text, "primary spec must build ImgKey.exe"

    gpu_spec_text = Path("ImgKey-GPU.spec").read_text(encoding="utf-8")
    assert "datas=[]" in gpu_spec_text, "GPU spec should not bundle extra data files"
    assert "'gpu_accel'" in gpu_spec_text and "'gpu_runtime'" in gpu_spec_text, "legacy GPU spec must include GPU helper modules"
    assert "imgkey_cuda.dll" in gpu_spec_text, "legacy GPU spec must bundle the compact CUDA DLL"
    assert "collect_dynamic_libs" not in gpu_spec_text, "GPU spec must not collect Python CUDA package libraries"
    assert "'torch'" not in gpu_spec_text.partition("hiddenimports=")[2].partition("]")[0], "GPU hidden imports must not include torch"
    assert "Splash(" in gpu_spec_text and "packaging/imgkey_splash.png" in gpu_spec_text, "legacy GPU spec must keep onefile splash/progress"
    assert "name='ImgKey-GPU'" in gpu_spec_text, "legacy GPU spec must build ImgKey-GPU.exe"

    requirement_lines = [
        line.strip()
        for line in Path("requirements-gpu-runtime-cu128.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirement_lines == [], requirement_lines


def run_source_surface_guard() -> None:
    roots = [
        Path("app.py"),
        Path("keyer.py"),
        Path("smoke_test.py"),
        Path("README.md"),
        Path("AGENTS.md"),
        Path("gpu_backend.py"),
        Path("gpu_accel.py"),
        Path("gpu_runtime.py"),
        Path("native_toolchain.py"),
        Path("vulkan_runtime.py"),
        Path("screen_analysis.py"),
        Path("ImgKey.spec"),
        Path("ImgKey-GPU.spec"),
        Path("requirements.txt"),
        Path("requirements-gpu-runtime-cu128.txt"),
    ]
    docs = Path("docs")
    if docs.exists():
        roots.extend(path for path in docs.glob("**/*") if path.is_file())
    engine_dir = Path("imgkey_engine")
    if engine_dir.exists():
        roots.extend(path for path in engine_dir.glob("**/*.py") if path.is_file())
    ui_dir = Path("ui")
    if ui_dir.exists():
        roots.extend(path for path in ui_dir.glob("**/*.py") if path.is_file())
    forbidden = [
        "Bi" + "RefNet",
        "bi" + "ref",
        "Corridor" + "Key",
        "Matting" + " Anything",
        "S" + "AM",
        "U2" + "Net",
        "MOD" + "Net",
        "ViT" + "Matte",
        "Hugging" + " Face",
        "trans" + "formers",
        "safe" + "tensors",
        "A" + "I Hint",
        "Hy" + "brid" + " Bi" + "RefNet",
    ]
    hits: list[tuple[str, str]] = []
    for path in roots:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for phrase in forbidden:
            if phrase in text:
                hits.append((str(path), phrase))
    assert not hits, f"removed product/runtime phrases remain: {hits}"


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    allowed = {
        "--write-diagnostics",
        "--write-edge-repair-diagnostics",
        "--write-algorithm-baseline",
        "--write-transition-unmix-diagnostics",
        "--write-geometric-benchmark",
        "--tune-geometric-defaults",
        "--gpu-parity",
        "--gpu-benchmark",
        "--write-perf-baseline",
    }
    unknown = [arg for arg in args if arg not in allowed]
    if unknown:
        raise SystemExit(
            "usage: python smoke_test.py [--write-diagnostics] [--write-edge-repair-diagnostics] "
            "[--write-algorithm-baseline] [--write-transition-unmix-diagnostics] [--write-geometric-benchmark] "
            "[--tune-geometric-defaults] [--gpu-parity] [--gpu-benchmark] [--write-perf-baseline]; "
            f"unknown: {', '.join(unknown)}"
        )

    writing_algorithm_baseline = "--write-algorithm-baseline" in args

    rgba = run_current_baseline()
    run_v2_numeric_tests()
    run_v4_edge_repair_tests()
    run_transition_unmix_baseline_tests()
    if not writing_algorithm_baseline:
        ensure_algorithm_upgrade_baseline()
        run_phase2_linear_color_tests()
        run_phase3_guided_alpha_tests()
        run_phase4_tile_local_screen_tests()
        run_phase5_crop_render_tests()
        run_phase6_tile_local_nearest_inner_tests()
    run_gpu_runtime_probe_tests()
    run_gpu_accel_backend_tests()
    run_gpu_backend_registry_tests()
    run_app_ui_tests()
    run_v6_screen_analysis_tests()
    run_geometric_benchmark_gate_tests()
    run_removed_surface_tests()
    run_packaging_flavor_tests()
    run_source_surface_guard()
    run_import_compile_tests()
    if "--write-diagnostics" in args:
        write_diagnostic_outputs()
    if "--write-edge-repair-diagnostics" in args:
        write_edge_repair_diagnostics()
    if "--write-algorithm-baseline" in args:
        write_algorithm_upgrade_baseline()
    if "--write-transition-unmix-diagnostics" in args:
        write_transition_unmix_diagnostics()
    if "--write-geometric-benchmark" in args:
        write_geometric_benchmark()
    if "--tune-geometric-defaults" in args:
        write_geometric_tuning_summary()
    if "--gpu-parity" in args:
        run_gpu_parity_tests()
    if "--gpu-benchmark" in args:
        write_gpu_benchmarks()
    if "--write-perf-baseline" in args:
        write_perf_baseline()
    print("smoke ok", rgba.shape)


if __name__ == "__main__":
    main()
