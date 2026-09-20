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
"""Adaptive policy synchronization for the online-RL actor-learner.

The learner periodically pushes fresh policy weights to the actors over the
parameters queue. The default cadence is a fixed wall-clock interval
(``policy_parameters_push_frequency``): a push happens every ``N`` seconds
regardless of whether the policy has actually moved. That wastes bandwidth when
learning has plateaued and, conversely, leaves actors running a stale policy for
up to ``N`` seconds when learning is fast.

This module implements *adaptive policy synchronization*: the decision of *when*
to push is driven by how much the policy has diverged since the last push rather
than by the clock alone. When the policy moves quickly the actors are refreshed
early; when it barely changes, redundant pushes are skipped. A hard upper bound
(the existing fixed cadence) still guarantees actors never exceed a bounded
staleness, and a lower bound rate-limits the (cheap) divergence checks so they
never run on every optimization step.

The state_dict -> queue contract downstream code depends on is untouched: this
only changes *whether* :func:`push_actor_policy_to_queue` is called on a given
learner iteration.

Adapted from "High-Throughput Distributed Reinforcement Learning via Adaptive
Policy Synchronization" (Baheri, arXiv:2507.10990). The paper's ClusterEnv /
DETACH distributed-environment framework and its learner-agnostic transport are
intentionally out of scope — LeRobot already owns its gRPC actor-learner
transport — so only the adaptive-synchronization mechanism is ported here, using
a parameter-free relative-L2 divergence proxy as the staleness signal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

StateDict = Mapping[str, Any]


def relative_state_divergence(current: StateDict, reference: StateDict) -> float:
    """Relative L2 movement between two (possibly nested) weight dicts.

    Returns ``sum_i ||current_i - reference_i|| / sum_i ||reference_i||`` over all
    matching tensors, walking nested dicts (e.g. ``{"policy": {...}}``). Tensors
    that are missing on either side or whose shapes disagree are skipped so that
    an architecture change never raises here. Returns ``0.0`` when the reference
    has no comparable tensors (nothing to diverge from).

    This is a deliberately parameter-free proxy for the paper's staleness signal:
    it needs no learned estimator and is cheap enough to evaluate between pushes.
    """
    numerator = 0.0
    denominator = 0.0

    def _accumulate(cur: Any, ref: Any) -> None:
        nonlocal numerator, denominator
        if isinstance(cur, Mapping) and isinstance(ref, Mapping):
            for key, cur_val in cur.items():
                if key in ref:
                    _accumulate(cur_val, ref[key])
        elif isinstance(cur, torch.Tensor) and isinstance(ref, torch.Tensor):
            if cur.shape != ref.shape:
                return
            cur_f = cur.detach().to(dtype=torch.float32)
            ref_f = ref.detach().to(dtype=torch.float32)
            numerator += float(torch.linalg.vector_norm(cur_f - ref_f))
            denominator += float(torch.linalg.vector_norm(ref_f))

    _accumulate(current, reference)

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _clone_snapshot(state_dict: StateDict) -> dict[str, Any]:
    """Detached CPU clone of a (nested) weight dict, safe to keep as a reference."""

    def _clone(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: _clone(val) for key, val in value.items()}
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", copy=True)
        return value

    return _clone(state_dict)


class AdaptivePolicySyncController:
    """Decide when to push policy weights to actors, based on divergence.

    The controller keeps a snapshot of the last-pushed weights and, on each
    learner iteration, answers :meth:`should_push`:

    - If less than ``min_interval_s`` has elapsed since the last push, skip
      (rate-limits the divergence computation; never fires on every step).
    - If at least ``max_interval_s`` has elapsed, push unconditionally. This is
      the existing fixed cadence and bounds worst-case actor staleness.
    - Otherwise push iff the relative divergence from the last-pushed weights is
      at least ``divergence_threshold``.

    Setting ``divergence_threshold=0.0`` and ``min_interval_s=0.0`` reproduces the
    original "push whenever possible up to the cadence" behavior, so this is a
    strict generalization of the previous fixed-cadence loop.

    Args:
        max_interval_s: Hard upper bound between pushes (the previous fixed
            cadence, ``policy_parameters_push_frequency``).
        min_interval_s: Lower bound between pushes / divergence checks. Clamped
            to ``[0, max_interval_s]``.
        divergence_threshold: Relative L2 movement that triggers an early push.
    """

    def __init__(
        self,
        max_interval_s: float,
        min_interval_s: float = 0.0,
        divergence_threshold: float = 0.0,
    ) -> None:
        if max_interval_s <= 0:
            raise ValueError(f"max_interval_s must be positive, got {max_interval_s}")
        if divergence_threshold < 0:
            raise ValueError(f"divergence_threshold must be non-negative, got {divergence_threshold}")

        self.max_interval_s = float(max_interval_s)
        self.min_interval_s = float(min(max(min_interval_s, 0.0), max_interval_s))
        self.divergence_threshold = float(divergence_threshold)

        self._reference: dict[str, Any] | None = None
        self._last_push_time: float | None = None
        self.last_divergence: float | None = None

    def initialize(self, weights: StateDict, now: float) -> None:
        """Seed the controller with the initial (already-pushed) weights."""
        self._reference = _clone_snapshot(weights)
        self._last_push_time = now

    def should_push(self, now: float, weights_fn: Callable[[], StateDict]) -> bool:
        """Whether to push on this iteration.

        ``weights_fn`` is a zero-arg callable returning the current weights. It is
        only invoked when a divergence check is actually needed (i.e. not on the
        cheap min/max-interval fast paths), so callers can pass an expensive
        ``get_weights`` without paying for it every step.
        """
        if self._last_push_time is None:
            return True

        elapsed = now - self._last_push_time
        if elapsed < self.min_interval_s:
            return False
        if elapsed >= self.max_interval_s:
            return True

        reference = self._reference
        if reference is None:
            return True

        self.last_divergence = relative_state_divergence(weights_fn(), reference)
        return self.last_divergence >= self.divergence_threshold

    def record_push(self, weights: StateDict, now: float) -> None:
        """Record that ``weights`` were just pushed at time ``now``."""
        self._reference = _clone_snapshot(weights)
        self._last_push_time = now
