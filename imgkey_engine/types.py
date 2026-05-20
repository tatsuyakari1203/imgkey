from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]

_MAX_INNER_LABEL_PIXELS = 16_000_000
_MIN_TILE_LOCAL_INNER_PIXELS = 8
_MAX_TILE_LOCAL_INNER_LABEL_PIXELS = 8_000_000
_MAX_TILE_LOCAL_NEAREST_INNER_RADIUS = 256
_MAX_ALPHA_RECOVERY_BLOCK_PIXELS = 2_000_000


@dataclass(slots=True)
class KeySettings:
    """Settings for the ImgKey classical keying pipeline.

    The original app still constructs this class with only
    ``key_color/tolerance/softness/edge_blur/cleanup/despill``. Those fields
    remain first-class compatibility controls and feed the v2 pipeline.
    """

    # Original v1 positional/keyword fields, kept in the same order.
    key_color: tuple[int, int, int] = (0, 220, 50)
    tolerance: float = 0.18
    softness: float = 0.075
    edge_blur: float = 1.2
    cleanup: int = 1
    despill: float = 0.70

    # v2 sampling controls.
    mode: str = "GraphicExact"
    sample_size: int = 5
    auto_border_sample: bool = True
    auto_detect_key_color: bool = False
    border_sample_width: int = 24
    # Local screen model builds a full-image uint8 screen map when the image is
    # small enough, and falls back to tile-local read-region estimates for large
    # tiled renders. It does not change matte probability decisions.
    local_screen_model: bool = True
    max_local_screen_model_pixels: int = 12_000_000

    # Matte controls.
    brightness_tolerance: float = 0.34
    clip_background: float = 0.78
    clip_foreground: float = 0.14
    matte_gamma: float = 1.0
    core_strength: float = 0.55
    edge_refine_radius: int = 0
    edge_softness: float = 0.55
    erode_expand: int = 0
    despeckle_min_area: int = 48

    # Connected-background policy. Default preserves disconnected key-colored
    # foreground islands; aggressive mode removes interior high-confidence key.
    aggressive_interior_removal: bool = False
    aggressive_threshold: float = 0.84
    aggressive_min_area: int = 0

    # Optional imported matte. A grayscale matte is merged conservatively into
    # the classical connected-background pipeline as foreground protection and
    # alpha guidance.
    alpha_hint_foreground_threshold: int = 192
    alpha_hint_minimum_alpha: int = 48
    alpha_hint_strength: float = 1.0

    # Color decontamination.
    decontaminate: float = 0.50
    luminance_restore: float = 0.35
    unmix_amount: float = 0.75

    # Export/preview hooks.
    preview_scale: float = 1.0
    full_res_crop: tuple[int, int, int, int] | None = None
    use_tiling: bool = True
    tile_size: int = 2048
    tile_overlap: int = 128

    # v4 edge color reconstruction. App/UI code can continue to drive
    # luminance_restore; luminance_protect is an optional API alias/override.
    fringe_remove: float = 0.75
    edge_color_repair: float = 0.65
    inner_color_pull: float = 0.45
    fringe_band_radius: int = 3
    luminance_protect: float | None = None

    # Optional v5 guided alpha refinement, appended to preserve existing
    # positional compatibility for earlier settings fields.
    guided_alpha_refine: float = 0.0
    guided_radius: int = 8
    guided_eps: float = 1e-3
    guided_max_pixels: int = 2_000_000

    # v7 transition-unmix controls, appended for positional compatibility.
    transition_unmix: bool = True
    alpha_recover_strength: float = 0.85
    foreground_reference_pull: float = 0.65
    key_vector_despill: float = 0.75
    transition_spill_threshold: float = 0.08
    transition_reconstruction_error: float = 0.08
    foreground_reference_radius: int = 96
    foreground_candidate_count: int = 4
    transition_alpha_min: int = 2
    transition_alpha_max: int = 253
    preserve_foreground_luma: float = 0.85

    # v9 screen-residue cleanup. Off by default so connected-background mode can
    # still preserve disconnected key-colored foreground unless the app/profile
    # opts into aggressive cleanup explicitly.
    screen_cleanup_strength: float = 0.0
    screen_cleanup_similarity: int = 8

    # Optional no-AI compact CUDA DLL acceleration. Default is CPU/off so library
    # callers and tests never load the DLL unless they opt in explicitly.
    gpu_acceleration: str = "Off"


@dataclass(slots=True)
class KeyResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground: np.ndarray | None
    background_mask: np.ndarray | None
    edge_mask: np.ndarray | None
    despill_mask: np.ndarray | None
    preview_scale: float = 1.0
    screen_probability: np.ndarray | None = None
    screen_color: tuple[int, int, int] | None = None
    alpha_hint: np.ndarray | None = None
    fringe_mask: np.ndarray | None = None
    repaired_edge: np.ndarray | None = None
    foreground_rgb: np.ndarray | None = None
    gpu_acceleration: dict | None = None


@dataclass(slots=True)
class _GlobalMatte:
    screen_color: tuple[int, int, int]
    screen_probability: np.ndarray
    screen_map: np.ndarray | None
    background_mask: np.ndarray
    edge_mask: np.ndarray
    alpha: np.ndarray
    color_alpha: np.ndarray | None
    alpha_hint: np.ndarray | None
    fringe_mask: np.ndarray
    inner_labels: np.ndarray | None
    inner_label_to_flat: np.ndarray | None
    inner_distance: np.ndarray | None
