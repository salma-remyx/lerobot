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
"""Tests for adaptive policy synchronization and its wiring into the learner."""

import pytest
import torch

from lerobot.rl.adaptive_sync import (
    AdaptivePolicySyncController,
    relative_state_divergence,
)

# Import the (non-new) call-site module to exercise the actual wiring edit.
from lerobot.rl.learner import push_actor_policy_to_queue
from lerobot.transport.utils import bytes_to_state_dict


class _FakeAlgorithm:
    """Minimal stand-in for an RLAlgorithm exposing mutable ``get_weights``."""

    def __init__(self, weights):
        self._weights = weights

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights


class _StubQueue:
    """Captures what the learner pushes, standing in for the parameters queue."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _weights(scale: float):
    return {"policy": {"fc.weight": torch.full((4,), scale), "fc.bias": torch.zeros(2)}}


def test_relative_divergence_zero_for_identical_weights():
    ref = _weights(1.0)
    assert relative_state_divergence(_weights(1.0), ref) == 0.0


def test_relative_divergence_scales_with_movement():
    ref = _weights(1.0)
    small = relative_state_divergence(_weights(1.1), ref)
    large = relative_state_divergence(_weights(2.0), ref)
    assert 0.0 < small < large


def test_relative_divergence_ignores_missing_and_mismatched():
    ref = {"policy": {"a": torch.ones(3)}}
    # Missing key on current side -> nothing comparable -> 0.0
    assert relative_state_divergence({"policy": {}}, ref) == 0.0
    # Shape mismatch is skipped rather than raising
    assert relative_state_divergence({"policy": {"a": torch.ones(5)}}, ref) == 0.0


def test_min_interval_rate_limits_pushes():
    controller = AdaptivePolicySyncController(
        max_interval_s=10.0, min_interval_s=2.0, divergence_threshold=0.0
    )
    controller.initialize(_weights(1.0), now=100.0)
    calls = []

    def weights_fn():
        calls.append(1)
        return _weights(5.0)

    # Below the min interval: no push and the (expensive) weights_fn is never called.
    assert controller.should_push(now=101.0, weights_fn=weights_fn) is False
    assert calls == []


def test_max_interval_forces_push_without_divergence():
    controller = AdaptivePolicySyncController(
        max_interval_s=4.0, min_interval_s=0.0, divergence_threshold=100.0
    )
    controller.initialize(_weights(1.0), now=0.0)
    # Weights unchanged (divergence 0) but the hard cadence has elapsed -> push anyway.
    assert controller.should_push(now=5.0, weights_fn=lambda: _weights(1.0)) is True


def test_divergence_threshold_triggers_early_push():
    controller = AdaptivePolicySyncController(
        max_interval_s=100.0, min_interval_s=0.0, divergence_threshold=0.25
    )
    controller.initialize(_weights(1.0), now=0.0)

    # Small movement stays below threshold.
    assert controller.should_push(now=1.0, weights_fn=lambda: _weights(1.1)) is False
    # Large movement crosses it.
    assert controller.should_push(now=1.0, weights_fn=lambda: _weights(2.0)) is True


def test_record_push_resets_reference():
    controller = AdaptivePolicySyncController(
        max_interval_s=100.0, min_interval_s=0.0, divergence_threshold=0.25
    )
    controller.initialize(_weights(1.0), now=0.0)
    controller.record_push(_weights(2.0), now=1.0)
    # Divergence is now measured against the freshly recorded 2.0 weights.
    assert controller.should_push(now=2.0, weights_fn=lambda: _weights(2.0)) is False


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        AdaptivePolicySyncController(max_interval_s=0.0)
    with pytest.raises(ValueError):
        AdaptivePolicySyncController(max_interval_s=1.0, divergence_threshold=-1.0)


def test_controller_drives_learner_push_wiring():
    """End-to-end wiring: controller decision -> push_actor_policy_to_queue -> queue."""
    algorithm = _FakeAlgorithm(_weights(1.0))
    controller = AdaptivePolicySyncController(
        max_interval_s=10.0, min_interval_s=0.0, divergence_threshold=0.25
    )
    controller.initialize(algorithm.get_weights(), now=0.0)
    queue = _StubQueue()

    # This mirrors the learner main loop: no meaningful change -> no push.
    if controller.should_push(now=1.0, weights_fn=algorithm.get_weights):
        weights = algorithm.get_weights()
        push_actor_policy_to_queue(parameters_queue=queue, algorithm=algorithm, weights=weights)
        controller.record_push(weights, now=1.0)
    assert queue.items == []

    # The policy diverges -> the loop pushes the current weights onto the queue.
    algorithm.set_weights(_weights(3.0))
    if controller.should_push(now=2.0, weights_fn=algorithm.get_weights):
        weights = algorithm.get_weights()
        push_actor_policy_to_queue(parameters_queue=queue, algorithm=algorithm, weights=weights)
        controller.record_push(weights, now=2.0)

    assert len(queue.items) == 1
    pushed = bytes_to_state_dict(queue.items[0])
    assert torch.allclose(pushed["policy"]["fc.weight"], torch.full((4,), 3.0))
