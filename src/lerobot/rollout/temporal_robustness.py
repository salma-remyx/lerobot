# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporal-robustness scoring for speed-swept policy rollouts.

Adapted from *"Does Imitation Learning Preserve Temporal Robustness in
Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution
Speeds"* (arXiv:2609.01453). The paper's finding is that equal task success at
*nominal* execution speed does not imply a learner preserves that success as the
task is executed faster: a scripted expert and an ACT learner both reach 100%
success at nominal speed, yet the learner degrades far more steeply as the
speedup factor grows (e.g. −34 to −48 percentage points at the maximum
demonstrated speed, versus −16 for the expert).

This module ports the paper's *quantitative core* — running the same policy
across a sweep of task-execution speed factors and reporting how much success
degrades relative to nominal — while intentionally leaving out the paper's
ParcelStow simulation, scripted expert, and ACT training pipeline. Any lerobot
policy is swept through the existing rollout stack by
:class:`~lerobot.rollout.strategies.speed_sweep.SpeedSweepStrategy`, and this
module scores the per-speed outcomes it collects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# The demonstrated / trained cadence. A speed factor of 1.0 replays the policy at
# the fps it was recorded at; larger factors compress the trajectory in time.
NOMINAL_SPEED_FACTOR = 1.0


def scaled_fps(base_fps: float, speed_factor: float) -> float:
    """Return the control-loop fps that executes the task at ``speed_factor``.

    The rollout loop holds each policy action for ``1 / fps`` seconds, so running
    the loop at ``base_fps * speed_factor`` replays the same demonstrated
    trajectory ``speed_factor`` times faster (or slower, for factors < 1). This
    is the single knob the paper sweeps, expressed in the repo's own cadence
    units — no policy or environment change is required.
    """
    if base_fps <= 0:
        raise ValueError(f"base_fps must be > 0, got {base_fps}")
    if speed_factor <= 0:
        raise ValueError(f"speed_factor must be > 0, got {speed_factor}")
    return base_fps * speed_factor


@dataclass(frozen=True)
class SpeedResult:
    """Aggregate success at one task-execution speed."""

    speed_factor: float
    successes: int
    trials: int

    def __post_init__(self) -> None:
        if self.speed_factor <= 0:
            raise ValueError(f"speed_factor must be > 0, got {self.speed_factor}")
        if self.trials < 0:
            raise ValueError(f"trials must be >= 0, got {self.trials}")
        if not 0 <= self.successes <= self.trials:
            raise ValueError(
                f"successes must be in [0, trials]; got successes={self.successes}, trials={self.trials}"
            )

    @property
    def success_rate(self) -> float:
        """Fraction of trials that succeeded, in ``[0, 1]`` (0 when no trials)."""
        return self.successes / self.trials if self.trials else 0.0


@dataclass(frozen=True)
class TemporalRobustnessReport:
    """Per-speed success and its degradation relative to the nominal speed.

    Built by :func:`build_temporal_robustness_report`. ``results`` is ordered by
    ascending ``speed_factor`` and always contains an entry for ``nominal_factor``.
    """

    results: tuple[SpeedResult, ...]
    nominal_factor: float = NOMINAL_SPEED_FACTOR

    def _result_for(self, speed_factor: float) -> SpeedResult:
        for result in self.results:
            if math.isclose(result.speed_factor, speed_factor):
                return result
        raise KeyError(f"No result recorded for speed_factor={speed_factor}")

    @property
    def speed_factors(self) -> tuple[float, ...]:
        return tuple(result.speed_factor for result in self.results)

    @property
    def max_speed_factor(self) -> float:
        return max(result.speed_factor for result in self.results)

    @property
    def nominal_success_rate(self) -> float:
        return self._result_for(self.nominal_factor).success_rate

    def success_rate_at(self, speed_factor: float) -> float:
        return self._result_for(speed_factor).success_rate

    def degradation_pp(self, speed_factor: float) -> float:
        """Drop in success at ``speed_factor`` versus nominal, in percentage points.

        Positive means the policy lost success relative to nominal speed; a small
        or negative value means it held up (the paper's "temporal robustness").
        """
        drop = self.nominal_success_rate - self.success_rate_at(speed_factor)
        return drop * 100.0

    def max_speed_degradation_pp(self) -> float:
        """Success drop at the fastest evaluated speed — the paper's headline number."""
        return self.degradation_pp(self.max_speed_factor)

    def retention_at_max(self) -> float:
        """Fraction of nominal success retained at the fastest speed.

        ``NaN`` when the policy never succeeds at nominal speed (retention is
        undefined — there is no baseline performance to preserve).
        """
        nominal = self.nominal_success_rate
        if nominal <= 0:
            return math.nan
        return self.success_rate_at(self.max_speed_factor) / nominal

    def summary(self) -> str:
        """Human-readable table of success and degradation across the sweep."""
        lines = [
            "Temporal robustness (success vs task-execution speed)",
            f"  nominal speed factor: {self.nominal_factor:g}",
            "  factor |  success | rate   | Δ vs nominal",
            "  -------+----------+--------+-------------",
        ]
        for result in self.results:
            degradation = self.degradation_pp(result.speed_factor)
            lines.append(
                f"  {result.speed_factor:>6.2f} | {result.successes:>4d}/{result.trials:<3d} | "
                f"{result.success_rate * 100:>5.1f}% | {degradation:>+6.1f} pp"
            )
        retention = self.retention_at_max()
        retention_str = "n/a" if math.isnan(retention) else f"{retention * 100:.1f}%"
        lines.append(
            f"  max-speed degradation: {self.max_speed_degradation_pp():+.1f} pp "
            f"(retains {retention_str} of nominal success)"
        )
        return "\n".join(lines)


def build_temporal_robustness_report(
    outcomes: Mapping[float, Sequence[bool]],
    nominal_factor: float = NOMINAL_SPEED_FACTOR,
) -> TemporalRobustnessReport:
    """Aggregate per-episode success labels into a :class:`TemporalRobustnessReport`.

    Args:
        outcomes: Maps each evaluated speed factor to the per-episode success
            labels collected at that speed (``True`` = task success).
        nominal_factor: The baseline speed every other speed is compared against;
            must be present in ``outcomes`` so degradation has a reference.

    Raises:
        ValueError: if ``outcomes`` is empty or lacks ``nominal_factor``.
    """
    if not outcomes:
        raise ValueError("outcomes is empty: nothing to score")
    if not any(math.isclose(factor, nominal_factor) for factor in outcomes):
        raise ValueError(
            f"nominal_factor={nominal_factor} not present in outcomes {sorted(outcomes)}; "
            "the nominal speed is required as the degradation baseline"
        )

    results = []
    for factor in sorted(outcomes):
        labels = list(outcomes[factor])
        successes = sum(1 for label in labels if label)
        results.append(SpeedResult(speed_factor=factor, successes=successes, trials=len(labels)))
    return TemporalRobustnessReport(results=tuple(results), nominal_factor=nominal_factor)
