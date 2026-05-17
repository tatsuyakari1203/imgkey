from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import py_compile
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ai_assist import BiRefNetAlphaAssist, CorridorKeyPlugin, check_birefnet_availability, check_corridorkey_availability
from keyer import (
    KeyResult,
    KeySettings,
    _MAX_INNER_LABEL_PIXELS,
    _build_nearest_inner_label_map,
    _estimate_screen_tile,
    _guided_filter_gray,
    _linear_f32_to_srgb_u8,
    _srgb_u8_to_linear_f32,
    checkerboard_composite,
    process_chroma_key,
    process_key_image,
)


ARTIFACT_DIR = Path(".artifact") / "smoke-fixtures"
EDGE_ARTIFACT_DIR = Path(".artifact") / "edge-repair-verification"
ALGORITHM_BASELINE_DIR = Path(".artifact") / "algorithm-upgrade-baseline"


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

    forbidden = {"pymatting", "scipy", "numba", "torch", "torchvision", "transformers", "onnxruntime", "onnxruntime_gpu"}
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

    forbidden = {"pymatting", "scipy", "numba", "torch", "torchvision", "transformers", "onnxruntime", "onnxruntime_gpu"}
    imported = forbidden & set(sys.modules)
    assert not imported, f"tile-local screen model must not import heavy optional modules: {sorted(imported)}"

    print("Phase 4 tile-local screen checks:")
    for line in summaries:
        print(f"  {line}")


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

    forbidden = {"pymatting", "scipy", "numba", "torch", "torchvision", "transformers", "onnxruntime", "onnxruntime_gpu"}
    imported = forbidden & set(sys.modules)
    assert not imported, f"default v4 edge repair path must not import heavy optional modules: {sorted(imported)}"


def run_optional_ai_seam_tests() -> None:
    heavy_modules = {"torch", "torchvision", "transformers", "onnxruntime", "onnxruntime_gpu", "pymatting", "scipy", "numba"}
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


def run_import_compile_tests() -> None:
    for source in ("app.py", "keyer.py", "smoke_test.py", "ai_assist.py"):
        py_compile.compile(source, doraise=True)
    importlib.import_module("app")
    importlib.import_module("keyer")


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
    run_optional_ai_seam_tests()
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
