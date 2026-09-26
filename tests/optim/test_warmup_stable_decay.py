# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import math

import pytest
from torch.optim.lr_scheduler import LambdaLR

# Import through the existing (non-new) scheduler module to exercise the registry
# wiring, not just the standalone helper.
from lerobot.optim.schedulers import LRSchedulerConfig, WarmupStableDecaySchedulerConfig


def _lrs_over_run(optimizer, config, num_training_steps):
    scheduler = config.build(optimizer, num_training_steps=num_training_steps)
    lrs = []
    for _ in range(num_training_steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return scheduler, lrs


def test_registered_in_choice_registry():
    # The build() contract is reachable through the shared draccus registry.
    assert LRSchedulerConfig.get_choice_class("warmup_stable_decay") is WarmupStableDecaySchedulerConfig


def test_build_returns_lambda_lr(optimizer):
    config = WarmupStableDecaySchedulerConfig(num_warmup_steps=10, num_decay_steps=20)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)


def test_warmup_stable_decay_phases(optimizer):
    base_lr = optimizer.param_groups[0]["lr"]
    config = WarmupStableDecaySchedulerConfig(num_warmup_steps=10, num_decay_steps=20, decay_type="linear")
    _, lrs = _lrs_over_run(optimizer, config, num_training_steps=100)

    # Warmup ramps up from ~0 toward the peak.
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[9] == pytest.approx(base_lr * 0.9)

    # Stable phase holds the peak LR (decay starts at step 80 = 100 - 20).
    for step in (10, 40, 79):
        assert lrs[step] == pytest.approx(base_lr)

    # Decay phase anneals below the peak and reaches (near) zero at the end.
    assert lrs[90] < base_lr
    assert lrs[-1] == pytest.approx(0.0, abs=base_lr * 0.06)


def test_min_lr_ratio_floor(optimizer):
    base_lr = optimizer.param_groups[0]["lr"]
    config = WarmupStableDecaySchedulerConfig(
        num_warmup_steps=5, num_decay_steps=20, decay_type="linear", min_lr_ratio=0.1
    )
    _, lrs = _lrs_over_run(optimizer, config, num_training_steps=50)
    # Final LR floors at min_lr_ratio * peak rather than zero.
    assert lrs[-1] == pytest.approx(base_lr * 0.1, abs=base_lr * 0.02)


def test_duration_agnostic_stable_lr(optimizer):
    """Core paper property: extending the run only lengthens the stable phase.

    The cooldown always occupies the trailing ``num_decay_steps`` steps, so the
    peak LR right before decay is identical regardless of total duration.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    config = WarmupStableDecaySchedulerConfig(num_warmup_steps=10, num_decay_steps=20)

    _, short = _lrs_over_run(optimizer, config, num_training_steps=60)
    # Reset LR before reusing the optimizer for a longer run.
    optimizer.param_groups[0]["lr"] = base_lr
    _, long = _lrs_over_run(optimizer, config, num_training_steps=200)

    # Last stable step before cooldown holds the peak in both runs.
    assert short[39] == pytest.approx(base_lr)  # decay_start = 60 - 20 = 40
    assert long[179] == pytest.approx(base_lr)  # decay_start = 200 - 20 = 180


def test_invalid_decay_type_rejected():
    with pytest.raises(ValueError):
        WarmupStableDecaySchedulerConfig(num_warmup_steps=5, num_decay_steps=10, decay_type="nope")


def test_one_sqrt_decays_faster_than_linear_early(optimizer):
    base_lr = optimizer.param_groups[0]["lr"]
    sqrt_cfg = WarmupStableDecaySchedulerConfig(num_warmup_steps=5, num_decay_steps=40, decay_type="1-sqrt")
    lin_cfg = WarmupStableDecaySchedulerConfig(num_warmup_steps=5, num_decay_steps=40, decay_type="linear")

    _, sqrt_lrs = _lrs_over_run(optimizer, sqrt_cfg, num_training_steps=60)
    optimizer.param_groups[0]["lr"] = base_lr
    _, lin_lrs = _lrs_over_run(optimizer, lin_cfg, num_training_steps=60)

    # Just after cooldown starts (step 21), 1-sqrt has dropped more than linear.
    assert sqrt_lrs[21] < lin_lrs[21]
    assert not math.isnan(sqrt_lrs[21])
