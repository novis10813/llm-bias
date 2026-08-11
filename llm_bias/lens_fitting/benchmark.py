"""Deterministic benchmark accounting and fail-closed 48-hour gate."""
from __future__ import annotations

from dataclasses import dataclass, asdict

MAX_GATE_SECONDS = 48 * 60 * 60

@dataclass(frozen=True)
class BenchmarkEstimate:
    prompts_measured: int
    seconds_per_prompt: float
    fit_prompts: int = 128
    evaluation_seconds: float = 0.0
    promotion_seconds: float = 0.0
    pilot_seconds: float = 0.0

    @property
    def fitting_seconds(self) -> float:
        return self.fit_prompts * self.seconds_per_prompt

    @property
    def total_seconds(self) -> float:
        return self.fitting_seconds + self.evaluation_seconds + self.promotion_seconds + self.pilot_seconds

    def as_dict(self) -> dict[str, float | int | bool]:
        value = asdict(self)
        value.update({"fitting_seconds": self.fitting_seconds, "total_seconds": self.total_seconds, "within_48h": self.total_seconds <= MAX_GATE_SECONDS})
        return value


def estimate_from_benchmark(*, elapsed_seconds: float, prompts: int, evaluation_seconds: float = 0.0, promotion_seconds: float = 0.0, pilot_seconds: float = 0.0) -> BenchmarkEstimate:
    if prompts < 1 or elapsed_seconds < 0:
        raise ValueError("benchmark prompts must be positive and elapsed time non-negative")
    return BenchmarkEstimate(prompts, elapsed_seconds / prompts, evaluation_seconds=evaluation_seconds, promotion_seconds=promotion_seconds, pilot_seconds=pilot_seconds)


def gate_benchmark(estimate: BenchmarkEstimate, *, finite: bool, resume_ok: bool, stable: bool) -> dict[str, object]:
    allowed = bool(finite and resume_ok and stable and estimate.total_seconds <= MAX_GATE_SECONDS)
    return {"status": "passed" if allowed else "failed", "fail_closed": not allowed, "reasons": [name for name, ok in (("finite_jacobians", finite), ("checkpoint_resume", resume_ok), ("stable_resources", stable), ("under_48_hours", estimate.total_seconds <= MAX_GATE_SECONDS)) if not ok], "estimate": estimate.as_dict()}
