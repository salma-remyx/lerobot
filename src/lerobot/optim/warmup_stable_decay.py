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
"""Warmup-Stable-Decay (WSD) learning-rate multipliers.

Adapted from *Scaling Laws and Compute-Optimal Training Beyond Fixed Training
Durations* (Hägele et al., 2024, https://arxiv.org/abs/2405.18392) and the
reference implementation at ``epfml/schedules-and-scaling`` (MIT licensed).

The WSD schedule replaces the cosine schedule's fixed-horizon coupling: instead
of tying the decay curve to the *total* number of steps, it keeps a constant
"stable" learning rate and only anneals over the *last* ``num_decay_steps``
steps. Because the stable phase carries the peak LR for as long as needed, the
same model can be trained for any number of steps and cooled down at the end
without re-tuning the schedule — the property the paper relies on to fit scaling
laws across training durations from a single run.

This module only computes the per-step multiplier (relative to the peak LR set
on the optimizer). The scheduler config that wires it into the LeRobot registry
lives in :mod:`lerobot.optim.schedulers`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

# Cooldown shapes evaluated in the paper. ``1-sqrt`` was reported as the best
# performing cooldown and is therefore the default.
DECAY_TYPES = ("1-sqrt", "linear", "cosine")


def _cooldown_factor(progress: float, decay_type: str) -> float:
    """Return the decay factor in ``[0, 1]`` for a cooldown ``progress`` in ``[0, 1]``.

    ``progress`` is 0 at the start of the cooldown (factor 1.0, peak LR) and 1 at
    the end (factor 0.0, floor LR).
    """
    progress = min(1.0, max(0.0, progress))
    if decay_type == "1-sqrt":
        return 1.0 - math.sqrt(progress)
    if decay_type == "linear":
        return 1.0 - progress
    if decay_type == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown decay_type {decay_type!r}; expected one of {DECAY_TYPES}.")


def make_warmup_stable_decay_lambda(
    num_training_steps: int,
    num_warmup_steps: int,
    num_decay_steps: int,
    decay_type: str = "1-sqrt",
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Build the ``LambdaLR`` multiplier function for a WSD schedule.

    The returned callable maps a training step to a multiplier applied to the
    optimizer's base (peak) learning rate:

    - **Warmup** (``[0, num_warmup_steps)``): linear ramp from 0 to 1.
    - **Stable** (``[num_warmup_steps, decay_start)``): constant 1.0.
    - **Decay** (``[decay_start, num_training_steps)``): anneal from 1.0 down to
      ``min_lr_ratio`` following ``decay_type``, where
      ``decay_start = num_training_steps - num_decay_steps``.

    If the run is too short to hold both warmup and the requested cooldown, the
    cooldown is clamped so it never starts before warmup finishes; the schedule
    stays well-defined for arbitrary ``num_training_steps``.

    Args:
        num_training_steps: Total number of training steps for this run.
        num_warmup_steps: Length of the linear warmup phase.
        num_decay_steps: Length of the trailing cooldown phase.
        decay_type: One of ``"1-sqrt"``, ``"linear"``, ``"cosine"``.
        min_lr_ratio: Floor LR as a fraction of the peak LR (0 anneals to zero).

    Returns:
        A function ``lr_lambda(current_step) -> float``.
    """
    if decay_type not in DECAY_TYPES:
        raise ValueError(f"Unknown decay_type {decay_type!r}; expected one of {DECAY_TYPES}.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")

    warmup_steps = max(0, num_warmup_steps)
    # Never let the cooldown eat into (or precede) the warmup phase.
    decay_steps = max(1, min(num_decay_steps, num_training_steps - warmup_steps))
    decay_start = max(warmup_steps, num_training_steps - decay_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step < decay_start:
            return 1.0
        progress = float(current_step - decay_start) / float(decay_steps)
        factor = _cooldown_factor(progress, decay_type)
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return lr_lambda
