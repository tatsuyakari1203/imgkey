from __future__ import annotations

from dataclasses import replace

import numpy as np

from keyer import KeySettings

from .preview_controller import resize_alpha_hint_mask


APP_DEFAULT_KEY_MODE = "Blue"
APP_DEFAULT_EDGE_RADIUS = 24
APP_DEFAULT_SETTINGS = KeySettings(
    key_color=(30, 80, 235),
    tolerance=0.26,
    softness=0.02,
    edge_blur=(APP_DEFAULT_EDGE_RADIUS - 1) / 4.0,
    cleanup=0,
    despill=0.80,
    sample_size=10,
    auto_border_sample=True,
    auto_detect_key_color=False,
    clip_background=0.95,
    clip_foreground=0.08,
    matte_gamma=1.60,
    core_strength=0.45,
    edge_refine_radius=APP_DEFAULT_EDGE_RADIUS,
    edge_softness=0.04,
    erode_expand=-4,
    despeckle_min_area=0,
    aggressive_interior_removal=True,
    decontaminate=0.70,
    luminance_restore=0.85,
    fringe_remove=0.85,
    edge_color_repair=0.80,
    inner_color_pull=0.60,
    fringe_band_radius=5,
    luminance_protect=0.85,
    transition_unmix=True,
    alpha_recover_strength=0.90,
    key_vector_despill=0.85,
    foreground_reference_pull=0.75,
    screen_cleanup_strength=1.00,
    screen_cleanup_similarity=8,
    gpu_acceleration="Off",
)


def app_default_settings() -> KeySettings:
    return replace(APP_DEFAULT_SETTINGS)


def current_settings_from_window(window) -> KeySettings:
    key_mode = window.key_mode.currentText()
    output_mode = window.output_mode.currentText() if hasattr(window, "output_mode") else "Classical"
    engine_mode = {
        "Imported Matte": "ImportedMatte",
    }.get(output_mode, "GraphicExact")
    radius = int(window.edge_radius.value())
    return KeySettings(
        key_color=window.settings.key_color,
        tolerance=float(window.screen_tolerance.value()),
        softness=float(window.screen_softness.value()),
        edge_blur=max(0.0, (radius - 1) / 4.0),
        cleanup=0,
        mode=engine_mode,
        sample_size=int(window.sample_size.value()),
        auto_border_sample=key_mode != "Pick",
        auto_detect_key_color=key_mode == "Auto",
        clip_background=float(window.clip_background.value()),
        clip_foreground=float(window.clip_foreground.value()),
        matte_gamma=float(window.matte_gamma.value()),
        core_strength=float(window.core_strength.value()),
        edge_refine_radius=radius,
        edge_softness=float(window.edge_softness.value()),
        erode_expand=int(window.erode_expand.value()),
        despeckle_min_area=int(window.despeckle.value()),
        aggressive_interior_removal=window.policy.currentText() == "Aggressive Interior Removal",
        despill=float(window.despill.value()),
        decontaminate=float(window.decontaminate.value()),
        luminance_restore=float(window.luminance_restore.value()),
        fringe_remove=float(window.fringe_remove.value()),
        edge_color_repair=float(window.edge_color_repair.value()),
        inner_color_pull=float(window.inner_color_pull.value()),
        fringe_band_radius=int(window.fringe_band.value()),
        transition_unmix=bool(window.transition_unmix.isChecked()),
        alpha_recover_strength=float(window.alpha_recover.value()),
        key_vector_despill=float(window.key_vector_despill.value()),
        foreground_reference_pull=float(window.foreground_reference_pull.value()),
        screen_cleanup_strength=(
            APP_DEFAULT_SETTINGS.screen_cleanup_strength
            if window.policy.currentText() == "Aggressive Interior Removal"
            else 0.0
        ),
        screen_cleanup_similarity=APP_DEFAULT_SETTINGS.screen_cleanup_similarity,
        gpu_acceleration=window.gpu_acceleration.currentText() if hasattr(window, "gpu_acceleration") else "Off",
        luminance_protect=float(window.luminance_restore.value()),
        preview_scale=float(window.current_display_scale),
        use_tiling=True,
    )


def processing_alpha_input(
    settings: KeySettings,
    alpha_hint_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if settings.mode == "ImportedMatte":
        return resize_alpha_hint_mask(alpha_hint_mask, shape)
    return None


def preset_control_values(name: str, window) -> dict[object, float | int]:
    presets = {
        "Fast": {
            window.screen_tolerance: 0.20,
            window.screen_softness: 0.07,
            window.clip_background: 0.80,
            window.clip_foreground: 0.16,
            window.edge_radius: 3,
            window.edge_softness: 0.35,
            window.despeckle: 24,
            window.decontaminate: 0.35,
            window.fringe_remove: 0.55,
            window.edge_color_repair: 0.40,
            window.inner_color_pull: 0.20,
            window.fringe_band: 2,
            window.luminance_restore: 0.20,
            window.alpha_recover: APP_DEFAULT_SETTINGS.alpha_recover_strength,
            window.key_vector_despill: APP_DEFAULT_SETTINGS.key_vector_despill,
            window.foreground_reference_pull: APP_DEFAULT_SETTINGS.foreground_reference_pull,
        },
        "Clean": {
            window.screen_tolerance: 0.18,
            window.screen_softness: 0.08,
            window.clip_background: 0.78,
            window.clip_foreground: 0.14,
            window.edge_radius: 6,
            window.edge_softness: 0.55,
            window.despeckle: 48,
            window.decontaminate: 0.50,
            window.fringe_remove: 0.70,
            window.edge_color_repair: 0.55,
            window.inner_color_pull: 0.35,
            window.fringe_band: 3,
            window.luminance_restore: 0.35,
            window.alpha_recover: APP_DEFAULT_SETTINGS.alpha_recover_strength,
            window.key_vector_despill: APP_DEFAULT_SETTINGS.key_vector_despill,
            window.foreground_reference_pull: APP_DEFAULT_SETTINGS.foreground_reference_pull,
        },
        "High Accuracy": {
            window.screen_tolerance: APP_DEFAULT_SETTINGS.tolerance,
            window.screen_softness: APP_DEFAULT_SETTINGS.softness,
            window.clip_background: APP_DEFAULT_SETTINGS.clip_background,
            window.clip_foreground: APP_DEFAULT_SETTINGS.clip_foreground,
            window.matte_gamma: APP_DEFAULT_SETTINGS.matte_gamma,
            window.core_strength: APP_DEFAULT_SETTINGS.core_strength,
            window.despeckle: APP_DEFAULT_SETTINGS.despeckle_min_area,
            window.edge_radius: APP_DEFAULT_SETTINGS.edge_refine_radius,
            window.edge_softness: APP_DEFAULT_SETTINGS.edge_softness,
            window.erode_expand: APP_DEFAULT_SETTINGS.erode_expand,
            window.despill: APP_DEFAULT_SETTINGS.despill,
            window.decontaminate: APP_DEFAULT_SETTINGS.decontaminate,
            window.fringe_remove: APP_DEFAULT_SETTINGS.fringe_remove,
            window.edge_color_repair: APP_DEFAULT_SETTINGS.edge_color_repair,
            window.inner_color_pull: APP_DEFAULT_SETTINGS.inner_color_pull,
            window.fringe_band: APP_DEFAULT_SETTINGS.fringe_band_radius,
            window.luminance_restore: APP_DEFAULT_SETTINGS.luminance_restore,
            window.alpha_recover: APP_DEFAULT_SETTINGS.alpha_recover_strength,
            window.key_vector_despill: APP_DEFAULT_SETTINGS.key_vector_despill,
            window.foreground_reference_pull: APP_DEFAULT_SETTINGS.foreground_reference_pull,
        },
    }
    return presets[name]
