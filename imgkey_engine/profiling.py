from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Any, Iterator


_ACTIVE_PROFILER: ContextVar["PipelineProfiler | None"] = ContextVar("imgkey_active_profiler", default=None)


def _round_ms(value: float) -> float:
    return float(round(float(value), 3))


@dataclass(slots=True)
class StageTiming:
    calls: int = 0
    total_ms: float = 0.0
    samples_ms: list[float] = field(default_factory=list)

    def add(self, elapsed_ms: float) -> None:
        elapsed = float(elapsed_ms)
        self.calls += 1
        self.total_ms += elapsed
        self.samples_ms.append(elapsed)

    def as_dict(self) -> dict[str, Any]:
        samples = list(self.samples_ms)
        if not samples:
            return {
                "calls": int(self.calls),
                "total_ms": _round_ms(self.total_ms),
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "samples_ms": [],
            }
        ordered = sorted(samples)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median = ordered[mid]
        else:
            median = (ordered[mid - 1] + ordered[mid]) / 2.0
        return {
            "calls": int(self.calls),
            "total_ms": _round_ms(sum(samples)),
            "mean_ms": _round_ms(sum(samples) / max(1, len(samples))),
            "median_ms": _round_ms(median),
            "min_ms": _round_ms(min(samples)),
            "max_ms": _round_ms(max(samples)),
            "samples_ms": [_round_ms(sample) for sample in samples],
        }


@dataclass(slots=True)
class PipelineProfiler:
    """Small reusable profiler for ImgKey pipeline stages.

    The profiler is intentionally passive: when no profiler is installed through
    ``profile_scope`` the instrumentation helpers below are near no-ops and do
    not change keying behavior. Reports aggregate by stable stage names so smoke
    tests, CLI profiling, and later UI/export controllers can share the same
    timing vocabulary.
    """

    timings: dict[str, StageTiming] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, elapsed_ms: float) -> None:
        if stage not in self.timings:
            self.timings[stage] = StageTiming()
        self.timings[stage].add(elapsed_ms)

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] = int(self.counters.get(name, 0)) + int(amount)

    def snapshot(self) -> dict[str, Any]:
        return {
            "timings": {stage: timing.as_dict() for stage, timing in sorted(self.timings.items())},
            "counters": {name: int(value) for name, value in sorted(self.counters.items())},
            "metadata": dict(self.metadata),
        }


def active_profiler() -> PipelineProfiler | None:
    return _ACTIVE_PROFILER.get()


@contextmanager
def profile_scope(profiler: PipelineProfiler) -> Iterator[PipelineProfiler]:
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


@contextmanager
def time_block(stage: str) -> Iterator[None]:
    profiler = active_profiler()
    if profiler is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profiler.record(stage, (time.perf_counter() - start) * 1000.0)


def record_timing(stage: str, elapsed_ms: float | None) -> None:
    profiler = active_profiler()
    if profiler is not None and elapsed_ms is not None:
        profiler.record(stage, float(elapsed_ms))


def record_count(name: str, amount: int = 1) -> None:
    profiler = active_profiler()
    if profiler is not None:
        profiler.count(name, amount)
