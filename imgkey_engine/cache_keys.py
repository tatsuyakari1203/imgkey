from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
from typing import Any, Iterable, Literal

from .types import KeySettings


SettingsCategory = Literal[
    "source",
    "mask",
    "base_matte",
    "transition_alpha",
    "tile_prep",
    "color",
    "backend",
]

ALGORITHM_VERSION = "imgkey-v11-cache-keys-v1"

SOURCE = "source"
MASK = "mask"
BASE_MATTE = "base_matte"
TRANSITION_ALPHA = "transition_alpha"
TILE_PREP = "tile_prep"
COLOR = "color"
BACKEND = "backend"

SETTING_FIELD_CATEGORIES: dict[str, SettingsCategory] = {
    # Source/proxy metadata. Image identity, decode generation, source alpha,
    # and proxy/full-resolution source generation live in source_generation_key.
    "preview_scale": SOURCE,
    # Manual/imported matte controls.
    "mode": MASK,
    "alpha_hint_foreground_threshold": MASK,
    "alpha_hint_minimum_alpha": MASK,
    "alpha_hint_strength": MASK,
    # Base/global matte and foreground/background classification controls.
    "key_color": BASE_MATTE,
    "tolerance": BASE_MATTE,
    "softness": BASE_MATTE,
    "edge_blur": BASE_MATTE,
    "cleanup": BASE_MATTE,
    "sample_size": BASE_MATTE,
    "auto_border_sample": BASE_MATTE,
    "auto_detect_key_color": BASE_MATTE,
    "border_sample_width": BASE_MATTE,
    "brightness_tolerance": BASE_MATTE,
    "clip_background": BASE_MATTE,
    "clip_foreground": BASE_MATTE,
    "matte_gamma": BASE_MATTE,
    "core_strength": BASE_MATTE,
    "edge_refine_radius": BASE_MATTE,
    "edge_softness": BASE_MATTE,
    "erode_expand": BASE_MATTE,
    "despeckle_min_area": BASE_MATTE,
    "aggressive_interior_removal": BASE_MATTE,
    "aggressive_threshold": BASE_MATTE,
    "aggressive_min_area": BASE_MATTE,
    "fringe_band_radius": BASE_MATTE,
    "guided_alpha_refine": BASE_MATTE,
    "guided_radius": BASE_MATTE,
    "guided_eps": BASE_MATTE,
    "guided_max_pixels": BASE_MATTE,
    "screen_cleanup_strength": BASE_MATTE,
    "screen_cleanup_similarity": BASE_MATTE,
    # Transition-alpha recovery controls. RGB-only foreground pull/despill knobs
    # stay in COLOR even though they share the transition repair path.
    "transition_unmix": TRANSITION_ALPHA,
    "alpha_recover_strength": TRANSITION_ALPHA,
    "transition_spill_threshold": TRANSITION_ALPHA,
    "transition_reconstruction_error": TRANSITION_ALPHA,
    "foreground_reference_radius": TRANSITION_ALPHA,
    "foreground_candidate_count": TRANSITION_ALPHA,
    "transition_alpha_min": TRANSITION_ALPHA,
    "transition_alpha_max": TRANSITION_ALPHA,
    # Tile geometry and bounded large-image prep controls.
    "local_screen_model": TILE_PREP,
    "max_local_screen_model_pixels": TILE_PREP,
    "full_res_crop": TILE_PREP,
    "use_tiling": TILE_PREP,
    "tile_size": TILE_PREP,
    "tile_overlap": TILE_PREP,
    # RGB-only repair/decontamination controls.
    "despill": COLOR,
    "decontaminate": COLOR,
    "luminance_restore": COLOR,
    "unmix_amount": COLOR,
    "fringe_remove": COLOR,
    "edge_color_repair": COLOR,
    "inner_color_pull": COLOR,
    "luminance_protect": COLOR,
    "foreground_reference_pull": COLOR,
    "key_vector_despill": COLOR,
    "preserve_foreground_luma": COLOR,
    # Backend selection never participates in source/matte fingerprints.
    "gpu_acceleration": BACKEND,
}

ALL_SETTINGS_CATEGORIES: tuple[SettingsCategory, ...] = (
    SOURCE,
    MASK,
    BASE_MATTE,
    TRANSITION_ALPHA,
    TILE_PREP,
    COLOR,
    BACKEND,
)

BASE_MATTE_FINGERPRINT_CATEGORIES: tuple[SettingsCategory, ...] = (MASK, BASE_MATTE)
MATTE_PIPELINE_FINGERPRINT_CATEGORIES: tuple[SettingsCategory, ...] = (
    MASK,
    BASE_MATTE,
    TRANSITION_ALPHA,
    TILE_PREP,
)
COLOR_FINGERPRINT_CATEGORIES: tuple[SettingsCategory, ...] = (COLOR,)
BACKEND_FINGERPRINT_CATEGORIES: tuple[SettingsCategory, ...] = (BACKEND,)

# Runtime cache fingerprints deliberately split the global matte pipeline more
# narrowly than the Phase 1 classification helpers.  Full-resolution crop is a
# render target, not a global-matte input, so it is normalized out to allow a
# Full Crop preview to seed a later full export.  Tile size is retained for the
# transition-alpha key because the large-image tile-local transition fallback can
# use it when global nearest-inner labels are skipped.
BASE_MATTE_RUNTIME_FIELDS: tuple[str, ...] = (
    "mode",
    "alpha_hint_foreground_threshold",
    "alpha_hint_minimum_alpha",
    "alpha_hint_strength",
    "key_color",
    "tolerance",
    "softness",
    "edge_blur",
    "cleanup",
    "sample_size",
    "auto_border_sample",
    "auto_detect_key_color",
    "border_sample_width",
    "brightness_tolerance",
    "clip_background",
    "clip_foreground",
    "matte_gamma",
    "core_strength",
    "edge_refine_radius",
    "edge_softness",
    "erode_expand",
    "despeckle_min_area",
    "aggressive_interior_removal",
    "aggressive_threshold",
    "aggressive_min_area",
    "fringe_band_radius",
    "guided_alpha_refine",
    "guided_radius",
    "guided_eps",
    "guided_max_pixels",
    "screen_cleanup_strength",
    "screen_cleanup_similarity",
    "local_screen_model",
    "max_local_screen_model_pixels",
)

REFERENCE_PREP_RUNTIME_FIELDS: tuple[str, ...] = (
    "local_screen_model",
    "max_local_screen_model_pixels",
    "foreground_reference_radius",
)

TRANSITION_ALPHA_RUNTIME_FIELDS: tuple[str, ...] = (
    "transition_unmix",
    "alpha_recover_strength",
    "transition_spill_threshold",
    "transition_reconstruction_error",
    "foreground_reference_radius",
    "foreground_candidate_count",
    "transition_alpha_min",
    "transition_alpha_max",
    "tile_size",
    "local_screen_model",
)

COLOR_RENDER_RUNTIME_FIELDS: tuple[str, ...] = (
    "despill",
    "decontaminate",
    "luminance_restore",
    "unmix_amount",
    "fringe_remove",
    "edge_color_repair",
    "inner_color_pull",
    "luminance_protect",
    "foreground_reference_pull",
    "key_vector_despill",
    "preserve_foreground_luma",
    "full_res_crop",
    "use_tiling",
    "tile_size",
    "tile_overlap",
    "gpu_acceleration",
)


def _key_settings_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(KeySettings))


def validate_settings_classification() -> None:
    names = set(_key_settings_field_names())
    classified = set(SETTING_FIELD_CATEGORIES)
    missing = names - classified
    extra = classified - names
    unknown_categories = set(SETTING_FIELD_CATEGORIES.values()) - set(ALL_SETTINGS_CATEGORIES)
    if missing or extra or unknown_categories:
        raise AssertionError(
            "KeySettings cache classification mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)} unknown_categories={sorted(unknown_categories)}"
        )


validate_settings_classification()


def setting_category(field_name: str) -> SettingsCategory:
    return SETTING_FIELD_CATEGORIES[field_name]


def settings_category_fields(category: SettingsCategory) -> tuple[str, ...]:
    return tuple(name for name in _key_settings_field_names() if SETTING_FIELD_CATEGORIES[name] == category)


def classified_settings_fields() -> dict[SettingsCategory, tuple[str, ...]]:
    return {category: settings_category_fields(category) for category in ALL_SETTINGS_CATEGORIES}


def settings_payload(settings: KeySettings, categories: Iterable[SettingsCategory]) -> dict[str, Any]:
    requested = set(categories)
    values = asdict(settings)
    return {
        name: values[name]
        for name in _key_settings_field_names()
        if SETTING_FIELD_CATEGORIES[name] in requested
    }


def settings_fields_payload(settings: KeySettings, field_names: Iterable[str]) -> dict[str, Any]:
    values = asdict(settings)
    payload: dict[str, Any] = {}
    valid = set(_key_settings_field_names())
    for name in field_names:
        if name not in valid:
            raise KeyError(f"Unknown KeySettings field in cache fingerprint: {name}")
        payload[name] = values[name]
    return payload


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def settings_fingerprint(
    settings: KeySettings,
    categories: Iterable[SettingsCategory],
    *,
    algorithm_version: str = ALGORITHM_VERSION,
) -> str:
    category_tuple = tuple(categories)
    return stable_fingerprint(
        {
            "kind": "settings",
            "algorithm_version": algorithm_version,
            "categories": category_tuple,
            "fields": settings_payload(settings, category_tuple),
        }
    )


def base_matte_settings_fingerprint(settings: KeySettings) -> str:
    return settings_fingerprint(settings, BASE_MATTE_FINGERPRINT_CATEGORIES)


def matte_pipeline_settings_fingerprint(settings: KeySettings) -> str:
    return settings_fingerprint(settings, MATTE_PIPELINE_FINGERPRINT_CATEGORIES)


def color_settings_fingerprint(settings: KeySettings) -> str:
    return settings_fingerprint(settings, COLOR_FINGERPRINT_CATEGORIES)


def backend_settings_fingerprint(settings: KeySettings) -> str:
    return settings_fingerprint(settings, BACKEND_FINGERPRINT_CATEGORIES)


def source_generation_key(
    image_identity: str,
    source_generation: int,
    *,
    decode_generation: int = 0,
    original_alpha_generation: int = 0,
    proxy_generation: int = 0,
    resolution: str = "full",
    shape: tuple[int, int] | None = None,
    algorithm_version: str = ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Return the immutable source/proxy generation payload for future caches."""

    return {
        "kind": "source_generation",
        "algorithm_version": algorithm_version,
        "image_identity": str(image_identity),
        "source_generation": int(source_generation),
        "decode_generation": int(decode_generation),
        "original_alpha_generation": int(original_alpha_generation),
        "proxy_generation": int(proxy_generation),
        "resolution": str(resolution),
        "shape": None if shape is None else [int(shape[0]), int(shape[1])],
    }


def source_generation_fingerprint(*args: Any, **kwargs: Any) -> str:
    return stable_fingerprint(source_generation_key(*args, **kwargs))


def mask_generation_key(
    *,
    mask_generation: int = 0,
    keep_generation: int = 0,
    remove_generation: int = 0,
    imported_matte_generation: int = 0,
    alpha_hint_generation: int | None = None,
    algorithm_version: str = ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Return manual/imported matte generations without touching mask arrays."""

    if alpha_hint_generation is None:
        alpha_hint_generation = imported_matte_generation
    return {
        "kind": "mask_generation",
        "algorithm_version": algorithm_version,
        "mask_generation": int(mask_generation),
        "keep_generation": int(keep_generation),
        "remove_generation": int(remove_generation),
        "imported_matte_generation": int(imported_matte_generation),
        "alpha_hint_generation": int(alpha_hint_generation),
    }


def mask_generation_fingerprint(**kwargs: Any) -> str:
    return stable_fingerprint(mask_generation_key(**kwargs))


def base_matte_cache_fingerprint(settings: KeySettings, source_key: dict[str, Any], mask_key: dict[str, Any]) -> str:
    return stable_fingerprint(
        {
            "kind": "base_matte_cache",
            "algorithm_version": ALGORITHM_VERSION,
            "settings": settings_payload(settings, BASE_MATTE_FINGERPRINT_CATEGORIES),
            "source": source_key,
            "mask": mask_key,
        }
    )


def matte_pipeline_cache_fingerprint(settings: KeySettings, source_key: dict[str, Any], mask_key: dict[str, Any]) -> str:
    return stable_fingerprint(
        {
            "kind": "matte_pipeline_cache",
            "algorithm_version": ALGORITHM_VERSION,
            "settings": settings_payload(settings, MATTE_PIPELINE_FINGERPRINT_CATEGORIES),
            "source": source_key,
            "mask": mask_key,
        }
    )


def runtime_base_matte_cache_fingerprint(settings: KeySettings, source_key: dict[str, Any], mask_key: dict[str, Any]) -> str:
    """Fingerprint the base global matte, excluding render crop/color/backend.

    This key includes local-screen cap settings because screen-residue cleanup can
    use the local screen plate to change alpha.  It intentionally excludes
    ``full_res_crop`` so crop previews can seed a full-resolution export.
    """

    return stable_fingerprint(
        {
            "kind": "runtime_base_matte_cache",
            "algorithm_version": ALGORITHM_VERSION,
            "settings": settings_fields_payload(settings, BASE_MATTE_RUNTIME_FIELDS),
            "source": source_key,
            "mask": mask_key,
        }
    )


def reference_prep_cache_fingerprint(settings: KeySettings, base_key: str) -> str:
    """Fingerprint global screen/reference prep derived from a base matte."""

    return stable_fingerprint(
        {
            "kind": "reference_tile_prep_cache",
            "algorithm_version": ALGORITHM_VERSION,
            "base_key": str(base_key),
            "settings": settings_fields_payload(settings, REFERENCE_PREP_RUNTIME_FIELDS),
        }
    )


def transition_alpha_cache_fingerprint(settings: KeySettings, base_key: str, reference_key: str) -> str:
    """Fingerprint recovered alpha derived from base matte + reference prep."""

    return stable_fingerprint(
        {
            "kind": "transition_alpha_cache",
            "algorithm_version": ALGORITHM_VERSION,
            "base_key": str(base_key),
            "reference_key": str(reference_key),
            "settings": settings_fields_payload(settings, TRANSITION_ALPHA_RUNTIME_FIELDS),
        }
    )


def color_render_cache_fingerprint(settings: KeySettings, transition_key: str, *, render_shape: tuple[int, int]) -> str:
    """Fingerprint optional rendered RGBA outputs.

    Phase 2 defines the color cache contract but keeps publication conservative;
    callers can use this helper once retaining rendered crops is desirable.
    """

    return stable_fingerprint(
        {
            "kind": "color_render_cache",
            "algorithm_version": ALGORITHM_VERSION,
            "transition_key": str(transition_key),
            "render_shape": [int(render_shape[0]), int(render_shape[1])],
            "settings": settings_fields_payload(settings, COLOR_RENDER_RUNTIME_FIELDS),
        }
    )
