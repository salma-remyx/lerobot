#!/usr/bin/env python

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

"""Tests for temporal-robustness scoring and the speed-sweep rollout strategy.

Covers the paper-derived metric (arXiv:2609.01453) in isolation and its wiring
into the existing rollout strategy factory / config validation surface.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.temporal_robustness import (  # noqa: E402
    SpeedResult,
    build_temporal_robustness_report,
    scaled_fps,
)

# ---------------------------------------------------------------------------
# scaled_fps — the paper's single speed knob, in the repo's cadence units
# ---------------------------------------------------------------------------


def test_scaled_fps_multiplies_cadence():
    assert scaled_fps(30.0, 1.0) == 30.0
    assert scaled_fps(30.0, 2.0) == 60.0
    assert scaled_fps(30.0, 0.5) == 15.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_scaled_fps_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        scaled_fps(bad, 1.0)
    with pytest.raises(ValueError):
        scaled_fps(30.0, bad)


# ---------------------------------------------------------------------------
# SpeedResult
# ---------------------------------------------------------------------------


def test_speed_result_success_rate():
    assert SpeedResult(1.0, successes=3, trials=4).success_rate == 0.75
    # No trials -> 0.0, not a ZeroDivisionError.
    assert SpeedResult(1.0, successes=0, trials=0).success_rate == 0.0


def test_speed_result_validates_bounds():
    with pytest.raises(ValueError):
        SpeedResult(1.0, successes=5, trials=4)


# ---------------------------------------------------------------------------
# TemporalRobustnessReport — the paper's degradation numbers
# ---------------------------------------------------------------------------


def test_report_degradation_and_retention():
    # Learner mirrors the paper's shape: 100% at nominal, steep drop at max speed.
    outcomes = {
        1.0: [True, True, True, True],
        1.5: [True, True, True, False],
        2.0: [True, False, False, False],
    }
    report = build_temporal_robustness_report(outcomes, nominal_factor=1.0)

    assert report.nominal_success_rate == 1.0
    assert report.max_speed_factor == 2.0
    assert report.success_rate_at(2.0) == 0.25

    # Nominal -> max is a 75 percentage-point drop; nominal itself does not degrade.
    assert report.degradation_pp(1.0) == pytest.approx(0.0)
    assert report.degradation_pp(1.5) == pytest.approx(25.0)
    assert report.max_speed_degradation_pp() == pytest.approx(75.0)
    assert report.retention_at_max() == pytest.approx(0.25)


def test_report_retention_is_nan_without_nominal_success():
    outcomes = {1.0: [False, False], 2.0: [False, False]}
    report = build_temporal_robustness_report(outcomes)
    assert math.isnan(report.retention_at_max())


def test_report_summary_is_renderable():
    report = build_temporal_robustness_report({1.0: [True], 2.0: [False]})
    text = report.summary()
    assert "Temporal robustness" in text
    assert "Δ vs nominal" in text


def test_build_report_requires_nominal_baseline():
    with pytest.raises(ValueError, match="nominal"):
        build_temporal_robustness_report({2.0: [True]}, nominal_factor=1.0)
    with pytest.raises(ValueError, match="empty"):
        build_temporal_robustness_report({})


# ---------------------------------------------------------------------------
# Integration: the strategy is dispatched by the existing factory / config layer
# ---------------------------------------------------------------------------


def test_speed_sweep_config_registers_and_validates():
    from lerobot.rollout import SpeedSweepStrategyConfig

    cfg = SpeedSweepStrategyConfig()
    # Registered under the shared draccus ChoiceRegistry used by --strategy.type.
    assert cfg.type == "speed_sweep"
    # Pure evaluation: it must not pull in any dataset requirement.
    assert cfg.dataset_mode == "none"

    with pytest.raises(ValueError, match="speed_factors"):
        SpeedSweepStrategyConfig(speed_factors=[])
    with pytest.raises(ValueError, match="episodes_per_speed"):
        SpeedSweepStrategyConfig(episodes_per_speed=0)


def test_factory_dispatches_speed_sweep():
    from lerobot.rollout import SpeedSweepStrategyConfig, create_strategy
    from lerobot.rollout.strategies import SpeedSweepStrategy

    strategy = create_strategy(SpeedSweepStrategyConfig())
    assert isinstance(strategy, SpeedSweepStrategy)


def test_strategy_schedule_puts_nominal_first_and_includes_it():
    from lerobot.rollout import SpeedSweepStrategyConfig, create_strategy

    # Nominal (1.0) omitted from the sweep list; it must still be evaluated first.
    strategy = create_strategy(SpeedSweepStrategyConfig(speed_factors=[2.0, 1.5], nominal_factor=1.0))
    assert strategy._speed_schedule() == [1.0, 1.5, 2.0]


def test_strategy_build_report_from_collected_outcomes():
    from lerobot.rollout import SpeedSweepStrategyConfig, create_strategy

    strategy = create_strategy(SpeedSweepStrategyConfig())
    # Simulate the outcomes the run loop accumulates from operator verdicts.
    strategy._outcomes = {1.0: [True, True], 2.0: [True, False]}
    report = strategy.build_report()
    assert report is not None
    assert report.max_speed_degradation_pp() == pytest.approx(50.0)


def test_strategy_build_report_none_without_nominal_labels():
    from lerobot.rollout import SpeedSweepStrategyConfig, create_strategy

    strategy = create_strategy(SpeedSweepStrategyConfig())
    strategy._outcomes = {2.0: [True, False]}  # nominal never labelled
    assert strategy.build_report() is None
