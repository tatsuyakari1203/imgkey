from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import keyer as keyer_module
from ai_assist import BiRefNetAlphaAssist, CorridorKeyPlugin, check_birefnet_availability, check_corridorkey_availability
from keyer import (
    KeyResult,
    KeySettings,
    _MAX_INNER_LABEL_PIXELS,
    _build_nearest_inner_label_map,
    _build_tile_local_nearest_inner_rgb,
    _estimate_screen_tile,
    _guided_filter_gray,
    _linear_f32_to_srgb_u8,
    _srgb_u8_to_linear_f32,
    _tile_local_nearest_inner_radius,
    checkerboard_composite,
    process_chroma_key,
    process_key_image,
)


ARTIFACT_DIR = Path(".artifact") / "smoke-fixtures"
EDGE_ARTIFACT_DIR = Path(".artifact") / "edge-repair-verification"
ALGORITHM_BASELINE_DIR = Path(".artifact") / "algorithm-upgrade-baseline"
HEAVY_OPTIONAL_MODULES = frozenset(
    {
        "accelerate",
        "einops",
        "huggingface_hub",
        "kornia",
        "numba",
        "onnxruntime",
        "onnxruntime_gpu",
        "pymatting",
        "safetensors",
        "scipy",
        "skimage",
        "timm",
        "torch",
        "torchvision",
        "transformers",
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
    diagnostic_only: bool = True


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
    alpha_u8 = np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.zeros((*alpha_u8.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = fg
    rgba[:, :, 3] = alpha_u8
    rgba[alpha_u8 == 0, :3] = 0
    return rgba


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
            for name in ("black", "white", "gray")
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
    return metrics


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
    fixtures = algorithm_upgrade_fixtures(include_large=True)
    all_metrics: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "python smoke_test.py --write-algorithm-baseline",
        "baseline_note": "Phase 1 v4 diagnostic baseline; hardest new fixture thresholds are intentionally not enforced yet.",
        "fixtures": {},
    }
    summary_lines = [
        "# ImgKey v5 classical algorithm baseline",
        "",
        "Generated by `python smoke_test.py --write-algorithm-baseline`.",
        "Hardest v5 fixture checks are diagnostic-only in Phase 1; later phases compare against these hashes/metrics.",
        "",
        "| Fixture | Edge residual max | Core max delta | Soft band px | Transparent RGB zero | Tile/full max |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]

    for fixture in fixtures:
        print(f"baseline fixture {fixture.name}: {fixture.notes}")
        result = process_key_image(fixture.rgb, fixture.settings)
        known_background, foreground_core, soft_edge = _fixture_masks(fixture)
        expected_alpha_u8 = (
            np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
            if fixture.expected_alpha is not None
            else np.zeros(fixture.rgb.shape[:2], dtype=np.uint8)
        )
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
        alpha_diff = int(np.abs(result.alpha.astype(np.int16) - artifact["alpha"].astype(np.int16)).max())
        assert alpha_diff == 0, f"{fixture.name}: Phase 2 must be color-only; alpha max diff vs v4 baseline={alpha_diff}"

        transparent = transparent_rgb_zero(result.rgba)
        assert transparent["ok"], (
            f"{fixture.name}: transparent RGB must stay zero, max={transparent['max_rgb_when_transparent']}"
        )

        soft_edge = artifact["soft_edge_mask"].astype(bool)
        edge_mask = soft_edge if np.any(soft_edge) else (result.fringe_mask > 0 if result.fringe_mask is not None else soft_edge)
        current_residual = edge_key_residual(result.rgba, fixture.settings.key_color, edge_mask)
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
            f"{fixture.name}: alpha_diff={alpha_diff} fringe_max={current_residual['max_positive_excess']}"
            f"<=v4:{baseline_residual['max_positive_excess']} fringe_p95={current_residual['p95_positive_excess']}"
            f"<=v4:{baseline_residual['p95_positive_excess']} core_drift={core_delta} unchanged_drift={unchanged_delta}"
        )

    print("Phase 2 linear-light checks vs v4 baseline:")
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
        assert local_residual["p95_positive_excess"] < no_pull_residual["p95_positive_excess"], (
            f"tile-local pull should improve p95 edge residual: {no_pull_residual} -> {local_residual}"
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
    assert hinted.alpha[island_y, island_x] >= 248, "AI alpha hint should protect foreground details without a model runtime"
    assert hinted.alpha_hint is not None and hinted.alpha_hint[island_y, island_x] == 255, "alpha hint should be returned for UI/debug views"

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
    low_cleanup = process_key_image(control_rgb, replace(control_settings, decontaminate=0.0, despill=0.0))
    high_cleanup = process_key_image(control_rgb, replace(control_settings, decontaminate=1.0, despill=1.0))
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


def run_optional_ai_seam_tests() -> None:
    heavy_modules = HEAVY_OPTIONAL_MODULES
    before = {name for name in heavy_modules if name in sys.modules}
    missing_model = Path(".artifact") / "missing-birefnet-model"

    cap = check_birefnet_availability(model_path=missing_model, adapter="")
    assert not cap.available, "BiRefNet seam must be disabled without local model/adapter configuration"
    assert cap.missing_configuration, "BiRefNet disabled status should explain missing model/adapter configuration"
    assist = BiRefNetAlphaAssist(model_path=missing_model, adapter="")
    try:
        assist.generate_hint(np.zeros((8, 8, 3), dtype=np.uint8))
    except RuntimeError as exc:
        assert "disabled" in str(exc).lower(), "BiRefNet no-dependency failure should be explicit"
    else:  # pragma: no cover - defensive
        raise AssertionError("BiRefNet stub should not run without external dependencies/model/adapter")

    corridor_cap = check_corridorkey_availability(adapter="")
    assert not corridor_cap.available, "CorridorKey plugin must be disabled without an external adapter"
    corridor = CorridorKeyPlugin(adapter="")
    try:
        corridor.process(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))
    except RuntimeError as exc:
        assert "corridorkey" in str(exc).lower(), "CorridorKey no-adapter failure should be explicit"
    else:  # pragma: no cover - defensive
        raise AssertionError("CorridorKey plugin should not run without an external adapter")

    after = {name for name in heavy_modules if name in sys.modules}
    assert after == before, f"AI capability checks must not import heavy runtimes: {after - before}"


def run_birefnet_adapter_manifest_tests() -> None:
    heavy_modules = HEAVY_OPTIONAL_MODULES
    before = {name for name in heavy_modules if name in sys.modules}

    importlib.import_module("ai_backends")
    adapter = importlib.import_module("ai_backends.birefnet_adapter")
    after_import = {name for name in heavy_modules if name in sys.modules}
    assert after_import == before, f"importing BiRefNet adapter must not import heavy runtimes: {after_import - before}"

    manifest = adapter.load_manifest()
    assert manifest["backend"] == "birefnet", "manifest must be BiRefNet-only"
    assert manifest["model_family"] == "BiRefNet", "manifest model family must be BiRefNet"
    assert manifest["selected_model"] == "ZhengPeng7/BiRefNet", "manifest must pin the selected BiRefNet source"
    assert manifest["source"]["repo_id"] == "ZhengPeng7/BiRefNet", "manifest must record exact HF source repo"
    assert manifest["source"].get("revision"), "manifest must record a pinned revision/commit"
    policy = manifest["local_path_policy"]
    assert policy["network_access"] == "forbidden", "BiRefNet runtime must be offline/local-only"
    assert policy["transformers_from_pretrained_args"]["local_files_only"] is True, "manifest must require local_files_only"
    assert policy["transformers_from_pretrained_args"]["trust_remote_code"] is True, "manifest must document remote-code loading"

    required_paths = {entry["path"] for entry in manifest["expected_layout"]["root_required_files"]}
    expected_paths = {"config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors", "README.md"}
    assert expected_paths <= required_paths, f"manifest missing required BiRefNet snapshot files: {expected_paths - required_paths}"
    license_paths = {entry["path"] for entry in manifest["expected_layout"]["license_files"]}
    assert "README.md" in license_paths, "manifest must record license/notice metadata file names"

    bad_paths = {
        "": "empty",
        "https://huggingface.co/ZhengPeng7/BiRefNet": "URL",
        "ZhengPeng7/BiRefNet": "repo IDs",
        str(Path(".artifact") / "missing-birefnet-model"): "does not exist",
    }
    for bad_path, expected_text in bad_paths.items():
        try:
            adapter.validate_model_path(bad_path)
        except adapter.ModelValidationError as exc:
            assert expected_text.lower() in str(exc).lower(), f"unexpected validation error for {bad_path!r}: {exc}"
        else:  # pragma: no cover - defensive
            raise AssertionError(f"invalid BiRefNet path should be rejected: {bad_path!r}")

        result = adapter.generate_alpha_hint(np.zeros((4, 5, 3), dtype=np.uint8), bad_path)
        assert result["ok"] is False and result["alpha_hint"] is None, "invalid paths must fail cleanly without inference"
        assert result["error"]["code"] == "model_validation_failed", "invalid paths should report model validation failure"

    with tempfile.TemporaryDirectory(prefix="imgkey-empty-hf-cache-") as empty_cache:
        try:
            adapter.validate_model_path(empty_cache)
        except adapter.ModelValidationError as exc:
            assert "missing required file" in str(exc).lower(), f"empty cache should fail on manifest layout: {exc}"
        else:  # pragma: no cover - defensive
            raise AssertionError("empty BiRefNet cache directory should be rejected")
        result = adapter.generate_alpha_hint(np.zeros((4, 5, 3), dtype=np.uint8), empty_cache)
        assert result["ok"] is False and result["error"]["code"] == "model_validation_failed", (
            "empty local cache paths must fail cleanly without downloads"
        )

    with tempfile.TemporaryDirectory(prefix="imgkey-birefnet-manifest-") as tmp_dir:
        snapshot = Path(tmp_dir)
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["BiRefNet"],
                    "auto_map": {
                        "AutoConfig": "BiRefNet_config.BiRefNetConfig",
                        "AutoModelForImageSegmentation": "birefnet.BiRefNet",
                    },
                    "bb_pretrained": False,
                }
            ),
            encoding="utf-8",
        )
        (snapshot / "BiRefNet_config.py").write_text("class BiRefNetConfig: pass\n", encoding="utf-8")
        (snapshot / "birefnet.py").write_text("class BiRefNet: pass\n", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"placeholder weights for manifest validation only")
        (snapshot / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
        validation = adapter.validate_model_path(snapshot)
        assert validation["ok"] is True, "validator must consume manifest for local snapshot layout"
        assert validation["config"]["architectures"] == ["BiRefNet"], "validator must check BiRefNet config architecture"

    rgb = adapter.ensure_rgb_u8(np.full((2, 3, 4), 128.8, dtype=np.float32))
    assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8 and rgb.flags.c_contiguous, "RGB helper must return HxWx3 uint8"
    alpha = adapter.ensure_alpha_u8(np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32), (2, 2))
    assert alpha.shape == (2, 2) and alpha.dtype == np.uint8 and alpha[0, 0] == 0 and alpha[1, 0] == 255, (
        "alpha helper must convert float masks to HxW uint8"
    )

    after_validation = {name for name in heavy_modules if name in sys.modules}
    assert after_validation == before, f"BiRefNet manifest/validation tests must not import heavy runtimes: {after_validation - before}"


def run_ai_worker_contract_tests() -> None:
    heavy_modules = HEAVY_OPTIONAL_MODULES
    before = {name for name in heavy_modules if name in sys.modules}

    ai_worker = importlib.import_module("ai_worker")
    after_import = {name for name in heavy_modules if name in sys.modules}
    assert after_import == before, f"importing ai_worker must not import heavy runtimes: {after_import - before}"

    unsupported = ai_worker.run_worker_request({"backend": "not-birefnet"})
    assert unsupported["ok"] is False
    assert unsupported["error"]["code"] == "unsupported_backend"
    assert unsupported["alpha_hint_path"] is None
    assert "supported" in unsupported["message"].lower()

    with tempfile.TemporaryDirectory(prefix="imgkey-ai-worker-contract-") as tmp_dir:
        tmp = Path(tmp_dir)
        input_path = tmp / "source.png"
        Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8), mode="RGB").save(input_path)

        base_request = {
            "backend": "birefnet",
            "input_image_path": str(input_path),
            "model_path": str(tmp / "missing-birefnet-model"),
            "device": "cpu",
            "mode": "global_plus_roi",
            "max_side": 64,
            "tile_size": 64,
            "tile_overlap": 8,
            "precision": "fp32",
            "output_dir": str(tmp / "out"),
            "temp_dir": str(tmp / "temp"),
        }
        invalid_model = ai_worker.run_worker_request(base_request)
        assert invalid_model["ok"] is False
        assert invalid_model["error"]["code"] == "model_validation_failed"
        assert "does not exist" in invalid_model["message"].lower()
        assert invalid_model["alpha_hint_path"] is None
        assert invalid_model["diagnostics_path"] and Path(invalid_model["diagnostics_path"]).is_file()
        invalid_diagnostics = json.loads(Path(invalid_model["diagnostics_path"]).read_text(encoding="utf-8"))
        cleanup = invalid_diagnostics["diagnostics"].get("temp_cleanup")
        assert cleanup and cleanup["removed"] is True, "worker staging temp directory should be removed on validation failure"

        cancel_file = tmp / "cancel-ai-worker"
        cancel_file.write_text("cancel\n", encoding="utf-8")
        cancelled_request = dict(base_request)
        cancelled_request["cancel_file_path"] = str(cancel_file)
        cancelled_request["output_dir"] = str(tmp / "cancelled-out")
        cancelled = ai_worker.run_worker_request(cancelled_request)
        assert cancelled["ok"] is False
        assert cancelled["error"]["code"] == "cancelled"
        assert "cancel" in cancelled["message"].lower()
        assert cancelled["diagnostics_path"] and Path(cancelled["diagnostics_path"]).is_file()

        missing_input_request = dict(base_request)
        missing_input_request["input_image_path"] = str(tmp / "missing-source.png")
        missing_input_request["output_dir"] = str(tmp / "missing-input-out")
        missing_input = ai_worker.run_worker_request(missing_input_request)
        assert missing_input["ok"] is False
        assert missing_input["error"]["code"] == "missing_input"
        assert "does not exist" in missing_input["message"].lower()
        assert missing_input["diagnostics_path"] and Path(missing_input["diagnostics_path"]).is_file()

    cli_request = json.dumps({"backend": "not-birefnet"})
    completed = subprocess.run(
        [sys.executable, "ai_worker.py", "--request", cli_request, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1, f"CLI worker failure should exit 1, got {completed.returncode}: {completed.stderr}"
    cli_response = json.loads(completed.stdout)
    assert cli_response["ok"] is False
    assert cli_response["error"]["code"] == "unsupported_backend"

    with tempfile.TemporaryDirectory(prefix="imgkey-ai-worker-subprocess-") as tmp_dir:
        tmp = Path(tmp_dir)
        input_path = tmp / "source.png"
        Image.fromarray(np.zeros((7, 11, 3), dtype=np.uint8), mode="RGB").save(input_path)
        cli_missing_model_request = json.dumps(
            {
                "backend": "birefnet",
                "input_image_path": str(input_path),
                "model_path": str(tmp / "missing-birefnet-model"),
                "device": "cpu",
                "mode": "global_plus_roi",
                "max_side": 64,
                "tile_size": 64,
                "tile_overlap": 8,
                "precision": "fp32",
                "output_dir": str(tmp / "out"),
                "temp_dir": str(tmp / "temp"),
            }
        )
        completed = subprocess.run(
            [sys.executable, "ai_worker.py", "--request", cli_missing_model_request, "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 1, f"CLI missing-model failure should exit 1, got {completed.returncode}: {completed.stderr}"
        cli_response = json.loads(completed.stdout)
        assert cli_response["ok"] is False
        assert cli_response["error"]["code"] == "model_validation_failed"
        assert cli_response["alpha_hint_path"] is None
        assert cli_response["diagnostics_path"] and Path(cli_response["diagnostics_path"]).is_file()
        diagnostics = json.loads(Path(cli_response["diagnostics_path"]).read_text(encoding="utf-8"))
        cleanup = diagnostics["diagnostics"].get("temp_cleanup")
        assert cleanup and cleanup["removed"] is True, "CLI worker subprocess must clean staging temp directories"
        temp_parent = tmp / "temp"
        leftovers = [path for path in temp_parent.glob("imgkey-ai-worker-*") if path.exists()] if temp_parent.exists() else []
        assert not leftovers, f"CLI worker subprocess left staging directories behind: {leftovers}"

    after_tests = {name for name in heavy_modules if name in sys.modules}
    assert after_tests == before, f"AI worker contract tests must not import heavy runtimes: {after_tests - before}"


def run_gpu_runtime_probe_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert "torch" not in sys.modules, "smoke test start should not have torch imported"
    gpu_runtime = importlib.import_module("gpu_runtime")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing gpu_runtime must not import heavy runtimes: {after_import - before}"

    fake_smi = {
        "available": False,
        "path": None,
        "error": "nvidia-smi was not found on PATH",
        "driver_version": None,
        "cuda_version": None,
        "gpus": [],
    }

    def missing_torch_loader() -> object:
        raise ImportError("No module named 'torch'")

    missing_probe = gpu_runtime.probe_gpu(torch_loader=missing_torch_loader, nvidia_smi_probe=lambda: fake_smi)
    assert missing_probe["status"] == "unavailable"
    assert missing_probe["available"] is False
    assert missing_probe["reason"] == "torch_import_failed"
    assert missing_probe["torch"]["import_success"] is False
    assert "ImportError" in missing_probe["torch"]["import_error"]
    assert missing_probe["cuda"]["is_available"] is False
    assert missing_probe["matmul_smoke"]["ran"] is False
    assert "pytorch" in missing_probe["message"].lower()

    class FakeVersion:
        cuda = None

    class FakeCuda:
        def get_arch_list(self) -> list[str]:
            return ["sm_120"]

        def is_available(self) -> bool:
            return False

        def device_count(self) -> int:
            return 0

    class FakeTorch:
        __version__ = "0.test"
        version = FakeVersion()
        cuda = FakeCuda()

    cpu_probe = gpu_runtime.probe_gpu(torch_loader=lambda: FakeTorch(), nvidia_smi_probe=lambda: fake_smi)
    assert cpu_probe["status"] == "unavailable"
    assert cpu_probe["reason"] == "cuda_unavailable"
    assert cpu_probe["torch"]["import_success"] is True
    assert cpu_probe["torch"]["version"] == "0.test"
    assert cpu_probe["cuda"]["arch_list"] == ["sm_120"]
    assert cpu_probe["matmul_smoke"]["ran"] is False
    assert "cuda" in cpu_probe["message"].lower()

    for probe in (missing_probe, cpu_probe):
        round_tripped = json.loads(json.dumps(probe))
        for key in ("schema_version", "status", "message", "torch", "cuda", "nvidia_smi", "matmul_smoke"):
            assert key in round_tripped, f"gpu runtime probe JSON missing {key}"

    after_probe = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_probe == before, f"fake gpu probe tests must not import heavy runtimes: {after_probe - before}"


def run_app_birefnet_ui_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app_module = importlib.import_module("app")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing app for UI probe must not import heavy runtimes: {after_import - before}"
    assert "BiRefNet Alpha" in app_module.VIEW_MODES, "BiRefNet Alpha view mode must be registered"

    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication

    created_app = QApplication.instance() is None
    qt_app = QApplication.instance() or QApplication(["imgkey-ui-probe"])
    window = app_module.MainWindow()
    try:
        assert window.generate_biref_action.text() == "Generate BiRefNet Hint"
        assert window.cancel_ai_action.text() == "Cancel AI"
        assert window.gpu_status_action.text() == "GPU Status"
        assert window.generate_birefnet_btn.text() == "Generate BiRefNet Hint"
        assert window.cancel_ai_btn.text() == "Cancel AI"
        assert window.gpu_status_btn.text() == "GPU Status"
        assert window.view_combo.findText("BiRefNet Alpha") >= 0
        assert window.output_mode.findText("Classical") >= 0
        assert window.output_mode.findText("Manual AI Hint") >= 0
        assert window.output_mode.findText("Hybrid BiRefNet") >= 0
        assert window.biref_alpha_mask is None
        assert window.alpha_hint_mask is None

        window.biref_alpha_mask = np.full((6, 8), 127, dtype=np.uint8)
        window._sync_birefnet_status("done")
        assert "Hybrid BiRefNet" in window.birefnet_status.text()
        assert window.current_settings().mode == "GraphicExact", "generated BiRefNet hint must not switch classical mode"
        alpha_hint, generated_hint = window._processing_alpha_inputs(window.current_settings(), (6, 8))
        assert alpha_hint is None and generated_hint is None, "Classical output mode must not pass manual or generated hints"

        manual_hint = np.full((6, 8), 255, dtype=np.uint8)
        window.alpha_hint_mask = manual_hint
        window._sync_ai_hint_status("manual.png")
        window.output_mode.setCurrentText("Manual AI Hint")
        assert window.current_settings().mode == "AIHint", "manual alpha hint import path must still drive AIHint mode"
        alpha_hint, generated_hint = window._processing_alpha_inputs(window.current_settings(), (6, 8))
        assert alpha_hint is not None and generated_hint is None, "Manual AI Hint mode must pass only the imported manual hint"
        window.output_mode.setCurrentText("Hybrid BiRefNet")
        assert window.current_settings().mode == "HybridBiRefNet", "Hybrid BiRefNet output mode must select the hybrid keyer"
        alpha_hint, generated_hint = window._processing_alpha_inputs(window.current_settings(), (6, 8))
        assert alpha_hint is None and generated_hint is not None, "Hybrid BiRefNet mode must pass only generated BiRefNet alpha"
        assert window.biref_alpha_mask is not window.alpha_hint_mask, "BiRefNet alpha must stay separate from manual alpha_hint_mask"

        with tempfile.TemporaryDirectory(prefix="imgkey-ui-ai-process-") as tmp_dir:
            tmp = Path(tmp_dir)
            request_path = tmp / "request.json"
            cancel_path = tmp / "cancel.flag"
            temp_input_path = tmp / "source.png"
            request_path.write_text("{}", encoding="utf-8")
            temp_input_path.write_text("temp", encoding="utf-8")
            worker = QProcess(window)
            worker.start(sys.executable, ["-c", "import time; time.sleep(30)"])
            assert worker.waitForStarted(3000), f"dummy AI process did not start: {worker.errorString()}"
            window.ai_worker_process = worker
            window.ai_worker_request_path = request_path
            window.ai_worker_cancel_path = cancel_path
            window.ai_worker_temp_input_path = temp_input_path
            window.cancel_ai()
            assert cancel_path.exists(), "Cancel AI should write a cancel flag for the worker"
            window.close()
            assert worker.state() == QProcess.NotRunning, "closeEvent must terminate the AI subprocess"
            assert window.ai_worker_process is None, "closeEvent must clear the AI process reference"
            assert not request_path.exists(), "closeEvent must clean the worker request file"
            assert not temp_input_path.exists(), "closeEvent must clean temp UI input files"
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()
        if created_app:
            qt_app.quit()

    after_probe = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_probe == before, f"BiRefNet UI probe must not import heavy runtimes: {after_probe - before}"


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

    # New maps are standalone in Phase 6. The existing export/low-memory result
    # must not grow retained debug-map fields before hybrid mode is wired later.
    low_memory = process_key_image(green.rgb, green.settings, include_debug=False)
    for name in ("screen_distance", "spill_probability", "classical_confidence", "screen_plate_rgb"):
        assert not hasattr(low_memory, name), f"low-memory export result should not retain {name} in Phase 6"

    after_tests = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_tests == before, f"screen analysis tests must not import heavy runtimes: {after_tests - before}"


def run_v6_hybrid_trimap_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    hybrid_trimap = importlib.import_module("hybrid_trimap")
    screen_analysis = importlib.import_module("screen_analysis")
    after_import = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_import == before, f"importing hybrid trimap helpers must not import heavy runtimes: {after_import - before}"

    h, w = 56, 56
    alpha = np.zeros((h, w), dtype=np.uint8)
    screen_prob = np.full((h, w), 255, dtype=np.uint8)
    screen_dist = np.zeros((h, w), dtype=np.uint8)
    spill_prob = np.zeros((h, w), dtype=np.uint8)
    confidence = np.full((h, w), 255, dtype=np.uint8)
    biref = np.zeros((h, w), dtype=np.uint8)
    background = np.zeros((h, w), dtype=np.uint8)
    edge = np.zeros((h, w), dtype=np.uint8)
    fringe = np.zeros((h, w), dtype=np.uint8)
    keep = np.zeros((h, w), dtype=np.uint8)
    remove = np.zeros((h, w), dtype=np.uint8)

    bg_pt = (3, 3)
    fg_pt = (10, 10)
    biref_fg_pt = (10, 40)
    conflict_pt = (26, 26)
    keep_conflict_pt = (42, 10)
    remove_pt = (42, 42)
    edge_pt = (18, 8)
    fringe_pt = (8, 28)
    spill_pt = (28, 8)
    keep_spill_pt = (8, 48)

    alpha[fg_pt] = 255
    screen_prob[fg_pt] = 10
    alpha[biref_fg_pt] = 24
    biref[biref_fg_pt] = 232
    screen_prob[biref_fg_pt] = 20

    alpha[conflict_pt] = 255
    biref[conflict_pt] = 128
    screen_prob[conflict_pt] = 250

    alpha[keep_conflict_pt] = 128
    biref[keep_conflict_pt] = 160
    screen_prob[keep_conflict_pt] = 250
    keep[keep_conflict_pt] = 255
    remove[keep_conflict_pt] = 255

    alpha[remove_pt] = 255
    biref[remove_pt] = 255
    screen_prob[remove_pt] = 20
    remove[remove_pt] = 255

    alpha[edge_pt] = 255
    screen_prob[edge_pt] = 20
    edge[edge_pt] = 255

    alpha[fringe_pt] = 120
    screen_prob[fringe_pt] = 80
    fringe[fringe_pt] = 255

    alpha[spill_pt] = 128
    screen_prob[spill_pt] = 60
    spill_prob[spill_pt] = 180

    alpha[keep_spill_pt] = 128
    screen_prob[keep_spill_pt] = 60
    spill_prob[keep_spill_pt] = 220
    keep[keep_spill_pt] = 255

    plate = screen_analysis.ScreenPlateRGB(source_shape=(h, w), fallback_rgb=(0, 220, 50), low_res_rgb=np.full((4, 4, 3), (0, 220, 50), dtype=np.uint8))
    result = hybrid_trimap.build_hybrid_trimap(
        alpha,
        screen_prob,
        screen_dist,
        spill_prob,
        confidence,
        background,
        edge,
        fringe,
        plate,
        biref,
        keep,
        remove,
        spill_threshold=96,
    )

    assert result.known_bg[bg_pt], "high-confidence screen/low-alpha pixel should be known background"
    assert result.safe_bg[bg_pt], "known screen background should be safe_bg"
    assert result.known_fg[fg_pt], "classical opaque non-screen pixel should be known foreground"
    assert result.known_fg[biref_fg_pt], "BiRefNet opaque non-screen pixel should be known foreground"
    assert result.protected_fg[fg_pt] and result.protected_fg[biref_fg_pt], "clean known foreground should be protected"

    assert result.conflict[conflict_pt], "screen/BiRefNet disagreement should be a conflict"
    assert result.unknown[conflict_pt] and result.hard_unknown[conflict_pt], "conflict should become hard unknown"
    assert not result.known_fg[conflict_pt], "conflict must override automatic known foreground"
    assert not result.known_bg[conflict_pt], "conflict should not become known background"

    assert result.known_fg[keep_conflict_pt], "manual keep should be the foreground override that wins over conflict"
    assert not result.known_bg[keep_conflict_pt] and not result.unknown[keep_conflict_pt], "manual keep should be exclusive"
    assert result.manual_keep_core[keep_conflict_pt], "manual keep core should be returned"
    assert not result.manual_remove_effective[keep_conflict_pt], "keep must override remove"

    assert result.known_bg[remove_pt], "manual remove should force background where keep is absent"
    assert not result.known_fg[remove_pt] and not result.unknown[remove_pt], "manual remove should be exclusive"
    assert result.manual_remove_effective[remove_pt], "manual remove effective mask should be returned"

    assert result.unknown[edge_pt] and result.hard_unknown[edge_pt], "strong edge band should become hard unknown"
    assert result.soft_unknown[fringe_pt], "fringe mask should expand soft unknown"
    assert result.unmix_region[fringe_pt], "soft detail region should be available for unmix"
    assert result.spill_region[spill_pt], "mid-alpha high-spill pixel should be spill region"
    assert result.despill_region[spill_pt], "spill region should be despill candidate away from keep/background"
    assert result.unmix_region[spill_pt], "mid-alpha spill pixel should be unmix candidate"
    assert result.spill_region[keep_spill_pt], "manual keep does not hide spill diagnostics"
    assert not result.despill_region[keep_spill_pt], "manual keep must protect from aggressive despill"

    assert not np.any(result.known_bg & result.known_fg), "known_bg/known_fg must be mutually exclusive"
    assert not np.any(result.known_bg & result.unknown), "known_bg/unknown must be mutually exclusive"
    assert not np.any(result.known_fg & result.unknown), "known_fg/unknown must be mutually exclusive"
    assert result.spill_threshold == 96, "spill threshold should be explicit on the result"
    assert result.candidate_alpha.dtype == np.uint8 and np.array_equal(result.candidate_alpha, alpha), "candidate alpha should default to classical alpha"
    assert "has_screen_plate" in result.debug_masks, "screen plate reference should be represented for Phase 8 diagnostics"

    explicit_core = np.zeros((h, w), dtype=np.uint8)
    explicit_core[keep_spill_pt] = 255
    explicit_core_result = hybrid_trimap.build_hybrid_trimap(
        alpha,
        screen_prob,
        screen_dist,
        spill_prob,
        confidence,
        background,
        edge,
        fringe,
        plate,
        biref,
        keep,
        remove,
        manual_keep_core=explicit_core,
    )
    assert explicit_core_result.known_fg[keep_conflict_pt], "raw keep must still win over remove when manual_keep_core is supplied"
    assert not explicit_core_result.manual_remove_effective[keep_conflict_pt], "manual_keep_core must not let remove bypass raw keep"

    after_tests = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_tests == before, f"hybrid trimap tests must not import heavy runtimes: {after_tests - before}"


def run_v6_hybrid_alpha_mode_tests() -> None:
    before = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    hybrid_trimap = importlib.import_module("hybrid_trimap")

    fixture = hair_lines_fixture()
    biref_alpha = np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    classical = process_key_image(fixture.rgb, fixture.settings)
    hybrid_settings = replace(fixture.settings, mode="HybridBiRefNet", guided_alpha_refine=0.0)
    hybrid = process_key_image(fixture.rgb, hybrid_settings, biref_alpha=biref_alpha)

    detail_mask = (fixture.expected_alpha > 0.05) & (fixture.expected_alpha < 0.90)
    classical_detail_mean = float(np.mean(classical.alpha[detail_mask]))
    hybrid_detail_mean = float(np.mean(hybrid.alpha[detail_mask]))
    assert hybrid_detail_mean >= classical_detail_mean + 8.0, (
        f"HybridBiRefNet should retain more thin detail alpha, {classical_detail_mean:.2f}->{hybrid_detail_mean:.2f}"
    )
    border = np.zeros(fixture.rgb.shape[:2], dtype=bool)
    border[:20, :] = True
    border[-20:, :] = True
    border[:, :20] = True
    border[:, -20:] = True
    assert int(hybrid.alpha[border].max()) == 0, "hybrid known/background border must stay fully transparent"
    assert int(hybrid.rgba[hybrid.alpha == 0, :3].max()) == 0, "HybridBiRefNet must keep alpha==0 RGB exactly zero"
    assert not np.array_equal(hybrid.alpha, biref_alpha), "HybridBiRefNet must not use BiRefNet alpha directly as final alpha"

    try:
        process_key_image(fixture.rgb, hybrid_settings, alpha_hint=biref_alpha)
    except ValueError as exc:
        assert "biref_alpha" in str(exc), "missing distinct BiRefNet alpha error should name biref_alpha"
    else:
        raise AssertionError("HybridBiRefNet must not silently consume manual alpha_hint as BiRefNet input")

    noisy_biref = np.full(fixture.rgb.shape[:2], 255, dtype=np.uint8)
    for mode in ("GraphicExact", "ProChroma", "AIHint"):
        settings = replace(fixture.settings, mode=mode)
        manual_hint = biref_alpha if mode == "AIHint" else None
        base = process_key_image(fixture.rgb, settings, alpha_hint=manual_hint)
        with_biref = process_key_image(fixture.rgb, settings, alpha_hint=manual_hint, biref_alpha=noisy_biref)
        assert np.array_equal(base.rgba, with_biref.rgba), f"{mode} output must ignore unrelated biref_alpha input"
        assert np.array_equal(base.alpha, with_biref.alpha), f"{mode} alpha must stay unchanged"

    h, w = 8, 8
    classical_alpha = np.full((h, w), 100, dtype=np.uint8)
    biref = np.full((h, w), 100, dtype=np.uint8)
    screen_prob = np.zeros((h, w), dtype=np.uint8)
    screen_dist = np.zeros((h, w), dtype=np.uint8)
    spill_prob = np.zeros((h, w), dtype=np.uint8)
    background = np.zeros((h, w), dtype=bool)
    edge = np.zeros((h, w), dtype=np.uint8)
    keep = np.zeros((h, w), dtype=np.uint8)
    remove = np.zeros((h, w), dtype=np.uint8)

    bg_pt = (0, 0)
    fg_pt = (0, 1)
    conflict_pt = (0, 2)
    keep_remove_pt = (0, 3)
    remove_pt = (0, 4)
    unknown_pt = (0, 5)

    screen_prob[bg_pt] = 255
    classical_alpha[bg_pt] = 0
    biref[bg_pt] = 0
    classical_alpha[fg_pt] = 20
    biref[fg_pt] = 230
    screen_prob[fg_pt] = 0
    classical_alpha[conflict_pt] = 255
    biref[conflict_pt] = 128
    screen_prob[conflict_pt] = 255
    classical_alpha[keep_remove_pt] = 20
    biref[keep_remove_pt] = 160
    screen_prob[keep_remove_pt] = 255
    keep[keep_remove_pt] = 255
    remove[keep_remove_pt] = 255
    classical_alpha[remove_pt] = 255
    biref[remove_pt] = 255
    remove[remove_pt] = 255
    classical_alpha[unknown_pt] = 32
    biref[unknown_pt] = 160

    trimap = hybrid_trimap.build_hybrid_trimap(
        classical_alpha,
        screen_prob,
        screen_dist,
        spill_prob,
        None,
        background,
        edge,
        None,
        None,
        biref,
        keep,
        remove,
    )
    original_alpha = np.ones((h, w), dtype=np.float32)
    original_alpha[fg_pt] = 0.5
    original_alpha[keep_remove_pt] = 0.25
    merged = keyer_module._merge_hybrid_alpha(
        np.zeros((h, w, 3), dtype=np.uint8),
        classical_alpha,
        biref,
        trimap,
        KeySettings(guided_alpha_refine=0.0),
        original_alpha,
        keep > 127,
        remove > 127,
    )

    expected_conflict_w = (128.0 - 64.0) / (220.0 - 64.0)
    expected_conflict_w = expected_conflict_w * expected_conflict_w * (3.0 - 2.0 * expected_conflict_w)
    expected_conflict = int(round(255.0 * (1.0 - expected_conflict_w) + 128.0 * expected_conflict_w))
    expected_unknown_w = (160.0 - 64.0) / (220.0 - 64.0)
    expected_unknown_w = expected_unknown_w * expected_unknown_w * (3.0 - 2.0 * expected_unknown_w)
    expected_unknown = int(round(32.0 * (1.0 - expected_unknown_w) + 160.0 * expected_unknown_w))

    assert int(merged[bg_pt]) == 0, "known_bg must clamp final alpha to 0"
    assert int(merged[fg_pt]) == 115, "source alpha must cap automatic known-fg max(classical,biref) alpha"
    assert trimap.conflict[conflict_pt] and trimap.unknown[conflict_pt], "conflict must stay unknown before manual overrides"
    assert int(merged[conflict_pt]) == expected_conflict, "conflict/hard-unknown should use unknown blend, not foreground clamp"
    assert int(merged[keep_remove_pt]) == 64, "manual keep must beat remove then remain capped by original source alpha"
    assert int(merged[remove_pt]) == 0, "manual remove must force background where keep is absent"
    assert int(merged[unknown_pt]) == expected_unknown, "unknown blend must use smoothstep(64, 220, biref_alpha)"

    tiled_settings = replace(hybrid_settings, use_tiling=True, tile_size=97, tile_overlap=18, local_screen_model=True)
    tiled = process_key_image(fixture.rgb, tiled_settings, biref_alpha=biref_alpha)
    full = process_key_image(fixture.rgb, replace(tiled_settings, use_tiling=False), biref_alpha=biref_alpha)
    tile_diff = np.abs(tiled.rgba.astype(np.int16) - full.rgba.astype(np.int16))
    assert int(tile_diff[:, :, 3].max()) == 0, f"HybridBiRefNet tiled/full alpha diff must be exact, got {int(tile_diff[:, :, 3].max())}"
    assert int(tile_diff.max()) <= 1, f"HybridBiRefNet tiled/full RGBA seam diff too high: {int(tile_diff.max())}"

    crop = (170, 90, 390, 310)
    x0, y0, x1, y1 = crop
    cropped = process_key_image(fixture.rgb, replace(tiled_settings, full_res_crop=crop), biref_alpha=biref_alpha)
    crop_diff = np.abs(cropped.rgba.astype(np.int16) - full.rgba[y0:y1, x0:x1].astype(np.int16))
    assert int(crop_diff[:, :, 3].max()) == 0, "HybridBiRefNet crop alpha must match full-render crop exactly"
    assert int(crop_diff.max()) <= 1, f"HybridBiRefNet crop RGBA diff too high: {int(crop_diff.max())}"

    skipped = process_key_image(
        fixture.rgb,
        replace(hybrid_settings, guided_alpha_refine=1.0, guided_radius=5, guided_max_pixels=1),
        biref_alpha=biref_alpha,
    )
    assert np.array_equal(hybrid.alpha, skipped.alpha), "HybridBiRefNet guided cap fallback must skip deterministically unchanged"
    assert np.array_equal(hybrid.rgba, skipped.rgba), "HybridBiRefNet guided cap fallback must preserve RGBA"

    refined = process_key_image(
        fixture.rgb,
        replace(hybrid_settings, guided_alpha_refine=0.85, guided_radius=5, guided_eps=1e-3, guided_max_pixels=1_000_000),
        biref_alpha=biref_alpha,
    )
    refine_mask = (hybrid.alpha > 0) & (hybrid.alpha < 255)
    rough_before = alpha_edge_roughness(hybrid.alpha, refine_mask)
    rough_after = alpha_edge_roughness(refined.alpha, refine_mask)
    assert rough_after <= rough_before * 0.75, f"unknown-only hybrid refinement should reduce jagged alpha, {rough_before:.3f}->{rough_after:.3f}"
    assert float(np.mean(refined.alpha[detail_mask])) >= hybrid_detail_mean, "guided hybrid refinement must preserve BiRefNet-retained thin detail"
    assert int(refined.alpha[border].max()) == 0, "guided hybrid refinement must not expand confident background leak"

    after_tests = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_tests == before, f"HybridBiRefNet alpha tests must not import heavy runtimes: {after_tests - before}"
    print(
        "Phase 7 HybridBiRefNet alpha checks: "
        f"detail_mean {classical_detail_mean:.2f}->{hybrid_detail_mean:.2f}; "
        f"tile_rgba_diff={int(tile_diff.max())}; crop_rgba_diff={int(crop_diff.max())}; "
        f"roughness {rough_before:.2f}->{rough_after:.2f}"
    )


def run_v6_hybrid_rgb_cleanup_tests() -> None:
    before_imports = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    fixture = hair_lines_fixture()
    biref_alpha = np.rint(np.clip(fixture.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    cleanup_off_settings = replace(
        fixture.settings,
        mode="HybridBiRefNet",
        guided_alpha_refine=0.0,
        edge_color_repair=0.0,
        unmix_amount=0.0,
        despill=0.0,
        fringe_remove=0.0,
    )
    cleanup_on_settings = replace(fixture.settings, mode="HybridBiRefNet", guided_alpha_refine=0.0)
    before = process_key_image(fixture.rgb, cleanup_off_settings, biref_alpha=biref_alpha)
    after = process_key_image(fixture.rgb, cleanup_on_settings, biref_alpha=biref_alpha)

    assert np.array_equal(before.alpha, after.alpha), "P8 RGB cleanup must not alter final HybridBiRefNet alpha"
    assert int(after.rgba[after.alpha == 0, :3].max()) == 0, "P8 must keep alpha==0 RGB exactly zero"
    assert np.isfinite(after.rgba.astype(np.float32)).all(), "P8 RGBA output must not contain NaN/Inf"

    edge_mask = fixture.soft_edge_mask & (after.alpha > 0)
    composite_residuals: dict[str, tuple[dict[str, int | float], dict[str, int | float]]] = {}
    for name, color in {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "checker": None,
    }.items():
        before_rgb = checkerboard_composite(before.rgba) if color is None else _solid_composite(before.rgba, color)
        after_rgb = checkerboard_composite(after.rgba) if color is None else _solid_composite(after.rgba, color)
        before_residual = rgb_key_residual(before_rgb, fixture.settings.key_color, edge_mask)
        after_residual = rgb_key_residual(after_rgb, fixture.settings.key_color, edge_mask)
        composite_residuals[name] = (before_residual, after_residual)
        assert after_residual["mean_positive_excess"] < before_residual["mean_positive_excess"] * 0.70, (
            f"hybrid cleanup should lower {name} composite halo mean: {before_residual} -> {after_residual}"
        )
        assert after_residual["p95_positive_excess"] <= before_residual["p95_positive_excess"], (
            f"hybrid cleanup should not worsen {name} composite p95 halo: {before_residual} -> {after_residual}"
        )

    core_delta = opaque_foreground_max_delta(fixture.rgb, after.rgba, fixture.foreground_core_mask, fixture.expected_foreground_rgb)
    assert core_delta["max_delta"] <= 32, f"hybrid cleanup changed protected foreground core too much: {core_delta}"

    low_alpha = edge_mask & (after.alpha > 0) & (after.alpha < 38)
    low_alpha_residual = edge_key_residual(after.rgba, fixture.settings.key_color, low_alpha)
    assert low_alpha_residual["max_positive_excess"] <= 96, f"low-alpha hybrid RGB stayed too saturated/noisy: {low_alpha_residual}"
    assert low_alpha_residual["p95_positive_excess"] <= 80, f"low-alpha hybrid RGB p95 stayed too saturated/noisy: {low_alpha_residual}"

    original_alpha = np.ones(fixture.rgb.shape[:2], dtype=np.float32)
    original_alpha[fixture.foreground_core_mask] = 0.42
    keep_mask = fixture.foreground_core_mask.astype(np.uint8) * 255
    capped = process_key_image(
        fixture.rgb,
        cleanup_on_settings,
        original_alpha=original_alpha,
        keep_mask=keep_mask,
        biref_alpha=biref_alpha,
    )
    cap_limit = int(round(255.0 * 0.42)) + 1
    assert int(capped.alpha[fixture.foreground_core_mask].max()) <= cap_limit, "P8 must not raise source-alpha-capped manual-keep pixels"
    assert int(capped.rgba[capped.alpha == 0, :3].max()) == 0, "source-capped known background must keep RGB zero"

    noisy_biref = np.full(fixture.rgb.shape[:2], 255, dtype=np.uint8)
    classical = process_key_image(fixture.rgb, replace(fixture.settings, mode="GraphicExact"))
    classical_with_biref = process_key_image(fixture.rgb, replace(fixture.settings, mode="GraphicExact"), biref_alpha=noisy_biref)
    assert np.array_equal(classical.rgba, classical_with_biref.rgba), "generated BiRefNet hints must not alter classical output"

    gradient = green_gradient_screen_fixture()
    gradient_biref = np.rint(np.clip(gradient.expected_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    gradient_off_settings = replace(
        gradient.settings,
        mode="HybridBiRefNet",
        guided_alpha_refine=0.0,
        edge_color_repair=0.0,
        unmix_amount=0.0,
        despill=0.0,
        fringe_remove=0.0,
    )
    gradient_off = process_key_image(gradient.rgb, gradient_off_settings, biref_alpha=gradient_biref)
    gradient_on = process_key_image(gradient.rgb, replace(gradient.settings, mode="HybridBiRefNet", guided_alpha_refine=0.0), biref_alpha=gradient_biref)
    gradient_edge = gradient.soft_edge_mask & (gradient_on.alpha > 0)
    gradient_before = edge_key_residual(gradient_off.rgba, gradient.settings.key_color, gradient_edge)
    gradient_after = edge_key_residual(gradient_on.rgba, gradient.settings.key_color, gradient_edge)
    assert gradient_after["mean_positive_excess"] < gradient_before["mean_positive_excess"], (
        f"local screen plate cleanup should reduce gradient-screen edge residual: {gradient_before} -> {gradient_after}"
    )

    after_imports = {name for name in HEAVY_OPTIONAL_MODULES if name in sys.modules}
    assert after_imports == before_imports, f"HybridBiRefNet RGB cleanup tests must not import heavy runtimes: {after_imports - before_imports}"
    print(
        "Phase 8 HybridBiRefNet RGB cleanup checks: "
        + "; ".join(
            f"{name} mean {pair[0]['mean_positive_excess']:.2f}->{pair[1]['mean_positive_excess']:.2f}"
            for name, pair in composite_residuals.items()
        )
        + f"; core_delta={core_delta['max_delta']}; low_alpha_p95={low_alpha_residual['p95_positive_excess']:.1f}"
    )


def run_import_compile_tests() -> None:
    for source in (
        "app.py",
        "keyer.py",
        "smoke_test.py",
        "ai_assist.py",
        "gpu_runtime.py",
        "ai_worker.py",
        "screen_analysis.py",
        "hybrid_trimap.py",
        "ai_backends/__init__.py",
        "ai_backends/birefnet_adapter.py",
    ):
        py_compile.compile(source, doraise=True)
    importlib.import_module("app")
    importlib.import_module("keyer")
    importlib.import_module("gpu_runtime")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    allowed = {"--write-diagnostics", "--write-edge-repair-diagnostics", "--write-algorithm-baseline"}
    unknown = [arg for arg in args if arg not in allowed]
    if unknown:
        raise SystemExit(
            "usage: python smoke_test.py [--write-diagnostics] [--write-edge-repair-diagnostics] "
            "[--write-algorithm-baseline]; "
            f"unknown: {', '.join(unknown)}"
        )

    writing_algorithm_baseline = "--write-algorithm-baseline" in args

    rgba = run_current_baseline()
    run_v2_numeric_tests()
    run_v4_edge_repair_tests()
    if not writing_algorithm_baseline:
        run_phase2_linear_color_tests()
        run_phase3_guided_alpha_tests()
        run_phase4_tile_local_screen_tests()
        run_phase5_crop_render_tests()
        run_phase6_tile_local_nearest_inner_tests()
    run_optional_ai_seam_tests()
    run_birefnet_adapter_manifest_tests()
    run_ai_worker_contract_tests()
    run_gpu_runtime_probe_tests()
    run_app_birefnet_ui_tests()
    run_v6_screen_analysis_tests()
    run_v6_hybrid_trimap_tests()
    run_v6_hybrid_alpha_mode_tests()
    run_v6_hybrid_rgb_cleanup_tests()
    run_import_compile_tests()
    if "--write-diagnostics" in args:
        write_diagnostic_outputs()
    if "--write-edge-repair-diagnostics" in args:
        write_edge_repair_diagnostics()
    if "--write-algorithm-baseline" in args:
        write_algorithm_upgrade_baseline()
    print("smoke ok", rgba.shape)


if __name__ == "__main__":
    main()
