from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import numpy as np

from .cache_keys import stable_fingerprint


def _readonly_array(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


def _readonly_copy(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def readonly_array(array: np.ndarray | None, *, copy: bool = False) -> np.ndarray | None:
    """Return a read-only ndarray for cache storage.

    Cache records own immutable-by-contract arrays.  Callers that expose arrays
    through ``KeyResult`` must copy before returning them to UI/library code.
    """

    return _readonly_copy(array) if copy else _readonly_array(array)


def mutable_result_copy(array: np.ndarray | None) -> np.ndarray | None:
    """Copy a cached/internal array before exposing it through public results."""

    if array is None:
        return None
    return np.array(array, copy=True)


@dataclass(slots=True)
class ProcessingGenerations:
    """UI/controller-owned generation counters for conservative invalidation.

    The cache never infers mutation from ndarray identity.  Controllers increment
    these counters on image load/decode, proxy rebuild, source-alpha changes,
    keep/remove mask edits, imported matte load/clear/update, and mask resets.
    Settings changes are handled by cache fingerprints rather than counters.
    """

    source_generation: int = 0
    decode_generation: int = 0
    original_alpha_generation: int = 0
    proxy_generation: int = 0
    mask_generation: int = 0
    keep_generation: int = 0
    remove_generation: int = 0
    imported_matte_generation: int = 0
    alpha_hint_generation: int = 0

    def bump_source(self) -> None:
        self.source_generation += 1
        self.decode_generation += 1
        self.original_alpha_generation += 1
        self.proxy_generation += 1
        self.mask_generation += 1
        self.keep_generation += 1
        self.remove_generation += 1
        self.imported_matte_generation += 1
        self.alpha_hint_generation += 1

    def bump_proxy(self) -> None:
        self.proxy_generation += 1

    def bump_keep(self) -> None:
        self.keep_generation += 1
        self.mask_generation += 1

    def bump_remove(self) -> None:
        self.remove_generation += 1
        self.mask_generation += 1

    def bump_imported_matte(self) -> None:
        self.imported_matte_generation += 1
        self.alpha_hint_generation += 1
        self.mask_generation += 1

    def bump_masks(self) -> None:
        self.mask_generation += 1
        self.keep_generation += 1
        self.remove_generation += 1
        self.imported_matte_generation += 1
        self.alpha_hint_generation += 1


@dataclass(frozen=True, slots=True)
class ProcessCacheContext:
    """Cache identity supplied by preview/export controllers.

    ``source_key`` must include resolution (``full`` or ``proxy``) and shape.
    Full export cache lookup therefore cannot match a proxy preview matte.
    ``mask_key`` carries manual/imported matte generations; mask arrays are not
    hashed in the hot path.
    """

    source_key: dict[str, Any]
    mask_key: dict[str, Any]

    @property
    def resolution(self) -> str:
        return str(self.source_key.get("resolution") or "full")

    @property
    def shape(self) -> tuple[int, int] | None:
        value = self.source_key.get("shape")
        if value is None:
            return None
        return int(value[0]), int(value[1])

    @property
    def source_fingerprint(self) -> str:
        return stable_fingerprint(self.source_key)

    @property
    def mask_fingerprint(self) -> str:
        return stable_fingerprint(self.mask_key)

    @property
    def active_generation_token(self) -> str:
        source = dict(self.source_key)
        source.pop("resolution", None)
        source.pop("shape", None)
        source.pop("proxy_generation", None)
        return stable_fingerprint({"source": source, "mask": self.mask_key})


@dataclass(frozen=True, slots=True)
class SourceCacheRecord:
    key: str
    source_key: dict[str, Any]
    mask_key: dict[str, Any]
    resolution: str
    shape: tuple[int, int]
    original_alpha_present: bool


@dataclass(frozen=True, slots=True)
class BaseMatteRecord:
    key: str
    source_fingerprint: str
    mask_fingerprint: str
    resolution: str
    shape: tuple[int, int]
    screen_color: tuple[int, int, int]
    screen_probability: np.ndarray
    background_mask: np.ndarray
    edge_mask: np.ndarray
    alpha: np.ndarray
    alpha_hint: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ReferencePrepRecord:
    key: str
    base_key: str
    screen_map: np.ndarray | None
    fringe_mask: np.ndarray
    inner_labels: np.ndarray | None = None
    inner_label_to_flat: np.ndarray | None = None
    inner_distance: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class TransitionAlphaRecord:
    key: str
    base_key: str
    reference_key: str
    alpha: np.ndarray
    color_alpha: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ColorRenderRecord:
    key: str
    transition_key: str
    rgba: np.ndarray
    despill_mask: np.ndarray | None = None


@dataclass(slots=True)
class CacheRunInfo:
    enabled: bool = False
    resolution: str | None = None
    base_matte: str = "disabled"
    reference_prep: str = "disabled"
    transition_alpha: str = "disabled"
    color_render: str = "disabled"
    cache_hit: str | bool = False
    cache_miss_reason: str | None = "cache_disabled"
    committed: bool = False
    staged_records: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "resolution": self.resolution,
            "base_matte": self.base_matte,
            "reference_prep": self.reference_prep,
            "transition_alpha": self.transition_alpha,
            "color_render": self.color_render,
            "cache_hit": self.cache_hit,
            "cache_miss_reason": self.cache_miss_reason,
            "committed": bool(self.committed),
            "staged_records": int(self.staged_records),
            "details": dict(self.details),
        }


class ProcessCache:
    """Active-image matte/transition cache with staged publication.

    The cache keeps source/mask identity metadata plus one active record per
    heavyweight layer by default.  Old full-size arrays are evicted when source,
    source alpha, masks, imported matte generations, or matte-affecting settings
    change.  Preview workers publish through ``ProcessCacheTransaction`` only
    after the UI accepts the result, so cancelled/stale previews never expose
    partial cache records.
    """

    def __init__(self, *, max_records_per_layer: int = 1) -> None:
        self.max_records_per_layer = max(1, int(max_records_per_layer))
        self._lock = RLock()
        self._active_generation_token: str | None = None
        self.source_records: dict[str, SourceCacheRecord] = {}
        self.base_matte_records: dict[str, BaseMatteRecord] = {}
        self.reference_prep_records: dict[str, ReferencePrepRecord] = {}
        self.transition_alpha_records: dict[str, TransitionAlphaRecord] = {}
        self.color_render_records: dict[str, ColorRenderRecord] = {}

    def clear(self) -> None:
        with self._lock:
            self.source_records.clear()
            self.base_matte_records.clear()
            self.reference_prep_records.clear()
            self.transition_alpha_records.clear()
            self.color_render_records.clear()
            self._active_generation_token = None

    def begin(self, context: ProcessCacheContext) -> "ProcessCacheTransaction":
        with self._lock:
            token = context.active_generation_token
            if self._active_generation_token != token:
                self.source_records.clear()
                self.base_matte_records.clear()
                self.reference_prep_records.clear()
                self.transition_alpha_records.clear()
                self.color_render_records.clear()
                self._active_generation_token = token
        return ProcessCacheTransaction(self, context)

    def _remember_source(self, record: SourceCacheRecord) -> None:
        self.source_records[record.key] = record
        self._trim(self.source_records)

    def _store_base(self, record: BaseMatteRecord) -> None:
        self.base_matte_records[record.key] = record
        for key, ref in list(self.reference_prep_records.items()):
            if ref.base_key != record.key:
                self.reference_prep_records.pop(key, None)
        for key, transition in list(self.transition_alpha_records.items()):
            if transition.base_key != record.key:
                self.transition_alpha_records.pop(key, None)
        self._trim(self.base_matte_records)

    def _store_reference(self, record: ReferencePrepRecord) -> None:
        self.reference_prep_records[record.key] = record
        for key, transition in list(self.transition_alpha_records.items()):
            if transition.reference_key != record.key:
                self.transition_alpha_records.pop(key, None)
        self._trim(self.reference_prep_records)

    def _store_transition(self, record: TransitionAlphaRecord) -> None:
        self.transition_alpha_records[record.key] = record
        self._trim(self.transition_alpha_records)

    def _store_color(self, record: ColorRenderRecord) -> None:
        self.color_render_records[record.key] = record
        self._trim(self.color_render_records)

    def _trim(self, records: dict[str, Any]) -> None:
        while len(records) > self.max_records_per_layer:
            records.pop(next(iter(records)))

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sources": len(self.source_records),
                "base_mattes": len(self.base_matte_records),
                "reference_preps": len(self.reference_prep_records),
                "transition_alphas": len(self.transition_alpha_records),
                "color_renders": len(self.color_render_records),
            }

    def matte_status(self, *, base_key: str, reference_key: str, transition_key: str) -> dict[str, bool]:
        """Return lightweight cache presence for preview UX/status labels."""

        with self._lock:
            return {
                "base_matte": str(base_key) in self.base_matte_records,
                "reference_prep": str(reference_key) in self.reference_prep_records,
                "transition_alpha": str(transition_key) in self.transition_alpha_records,
            }


class ProcessCacheTransaction:
    """Read-through/staged cache writer for one keyer invocation."""

    def __init__(self, cache: ProcessCache, context: ProcessCacheContext) -> None:
        self.cache = cache
        self.context = context
        self.info = CacheRunInfo(enabled=True, resolution=context.resolution, cache_miss_reason=None)
        self._staged_sources: dict[str, SourceCacheRecord] = {}
        self._staged_base: dict[str, BaseMatteRecord] = {}
        self._staged_reference: dict[str, ReferencePrepRecord] = {}
        self._staged_transition: dict[str, TransitionAlphaRecord] = {}
        self._staged_color: dict[str, ColorRenderRecord] = {}
        self._closed = False

    def remember_source(self, original_alpha_present: bool) -> None:
        shape = self.context.shape or (0, 0)
        key = self.context.source_fingerprint
        self._staged_sources[key] = SourceCacheRecord(
            key=key,
            source_key=dict(self.context.source_key),
            mask_key=dict(self.context.mask_key),
            resolution=self.context.resolution,
            shape=shape,
            original_alpha_present=bool(original_alpha_present),
        )

    def get_base(self, key: str) -> BaseMatteRecord | None:
        if key in self._staged_base:
            return self._staged_base[key]
        with self.cache._lock:
            return self.cache.base_matte_records.get(key)

    def get_reference(self, key: str) -> ReferencePrepRecord | None:
        if key in self._staged_reference:
            return self._staged_reference[key]
        with self.cache._lock:
            return self.cache.reference_prep_records.get(key)

    def get_transition(self, key: str) -> TransitionAlphaRecord | None:
        if key in self._staged_transition:
            return self._staged_transition[key]
        with self.cache._lock:
            return self.cache.transition_alpha_records.get(key)

    def stage_base(self, record: BaseMatteRecord) -> None:
        self._staged_base[record.key] = record
        self._refresh_staged_count()

    def stage_reference(self, record: ReferencePrepRecord) -> None:
        self._staged_reference[record.key] = record
        self._refresh_staged_count()

    def stage_transition(self, record: TransitionAlphaRecord) -> None:
        self._staged_transition[record.key] = record
        self._refresh_staged_count()

    def stage_color(self, record: ColorRenderRecord) -> None:
        self._staged_color[record.key] = record
        self._refresh_staged_count()

    def commit(self) -> None:
        if self._closed:
            return
        with self.cache._lock:
            for record in self._staged_sources.values():
                self.cache._remember_source(record)
            for record in self._staged_base.values():
                self.cache._store_base(record)
            for record in self._staged_reference.values():
                self.cache._store_reference(record)
            for record in self._staged_transition.values():
                self.cache._store_transition(record)
            for record in self._staged_color.values():
                self.cache._store_color(record)
        self.info.committed = True
        self._closed = True

    def discard(self) -> None:
        self._staged_sources.clear()
        self._staged_base.clear()
        self._staged_reference.clear()
        self._staged_transition.clear()
        self._staged_color.clear()
        self._refresh_staged_count()
        self._closed = True

    def _refresh_staged_count(self) -> None:
        self.info.staged_records = (
            len(self._staged_sources)
            + len(self._staged_base)
            + len(self._staged_reference)
            + len(self._staged_transition)
            + len(self._staged_color)
        )
