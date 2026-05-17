from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import py_compile
import sys

import numpy as np
from PIL import Image

from ai_assist import BiRefNetAlphaAssist, CorridorKeyPlugin, check_birefnet_availability, check_corridorkey_availability
from keyer import (
    KeyResult,
    KeySettings,
    _MAX_INNER_LABEL_PIXELS,
    _build_nearest_inner_label_map,
    checkerboard_composite,
    process_chroma_key,
    process_key_image,
)


ARTIFACT_DIR = Path(".artifact") / "smoke-fixtures"
EDGE_ARTIFACT_DIR = Path(".artifact") / "edge-repair-verification"


@dataclass(frozen=True)
class DiagnosticFixture:
    name: str
    rgb: np.ndarray
    settings: KeySettings
    notes: str


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
    ]
    if include_large:
        fixtures.append(large_synthetic_fixture())
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
    allowed = {"--write-diagnostics", "--write-edge-repair-diagnostics"}
    unknown = [arg for arg in args if arg not in allowed]
    if unknown:
        raise SystemExit(
            "usage: python smoke_test.py [--write-diagnostics] [--write-edge-repair-diagnostics]; "
            f"unknown: {', '.join(unknown)}"
        )

    rgba = run_current_baseline()
    run_v2_numeric_tests()
    run_v4_edge_repair_tests()
    run_optional_ai_seam_tests()
    run_import_compile_tests()
    if "--write-diagnostics" in args:
        write_diagnostic_outputs()
    if "--write-edge-repair-diagnostics" in args:
        write_edge_repair_diagnostics()
    print("smoke ok", rgba.shape)


if __name__ == "__main__":
    main()
