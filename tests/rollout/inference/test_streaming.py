#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the streaming inference engine (confidence-gated action reuse).

Adapted from FlashVLA (https://arxiv.org/abs/2608.27384): the engine should emit
one decoded chunk per inference and reuse its buffered actions while the
predicted trajectory stays stable, so a run of ticks costs far fewer policy calls
than ticks.  The wiring is exercised through the public factory
(``create_inference_engine``) — a non-new module — so the tests prove the backend
is reachable from ``--inference.type=flashvla``, not just importable.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.utils.constants import ACTION  # noqa: E402

ORDERED_KEYS = ["j0.pos", "j1.pos"]
DATASET_FEATURES = {ACTION: {"names": ORDERED_KEYS}}


class _Identity:
    """Stand-in for a processor pipeline: passes batches through untouched."""

    def __call__(self, x):
        return x

    def reset(self):
        pass


class _ChunkPolicy:
    """Minimal chunking policy: returns a preset ``[1, T, A]`` chunk per call."""

    def __init__(self, chunks: list[torch.Tensor]) -> None:
        self._chunks = chunks
        self.calls = 0
        self.config = MagicMock(use_amp=False)

    def predict_action_chunk(self, batch, **kwargs):
        chunk = self._chunks[min(self.calls, len(self._chunks) - 1)]
        self.calls += 1
        return chunk.unsqueeze(0)

    def reset(self):
        pass

    def supports_text_generation(self):
        return False


def _make_engine(chunks, **overrides):
    from lerobot.rollout.inference import StreamingInferenceEngine

    kwargs = dict(
        policy=_ChunkPolicy(chunks),
        preprocessor=_Identity(),
        postprocessor=_Identity(),
        dataset_features=DATASET_FEATURES,
        ordered_action_keys=ORDERED_KEYS,
        task="pick",
        device="cpu",
        robot_type="mock",
    )
    kwargs.update(overrides)
    return StreamingInferenceEngine(**kwargs)


def _obs():
    # prepare_observation_for_inference expects numpy arrays keyed by feature name.
    return {"observation.state": np.zeros(2, dtype=np.float32)}


# ---------------------------------------------------------------------------
# Reuse gate (pure logic)
# ---------------------------------------------------------------------------


def test_reuse_gate_confidence_bounds_and_monotonicity():
    from lerobot.rollout.inference import StreamingReuseGate

    gate = StreamingReuseGate(min_horizon=1, max_horizon=10, overlap=4)

    # No previous chunk -> no drift evidence -> full confidence.
    assert gate.confidence(None, torch.zeros(4, 2), consumed=0) == 1.0

    prev = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    # A chunk that continues the previous prediction exactly (consumed=2 aligns
    # prev[2:] with new[:]) is maximally confident.
    identical = prev[2:]
    assert gate.confidence(prev, identical, consumed=2) == pytest.approx(1.0)

    # A chunk that disagrees over the overlap is less confident.
    drifted = identical + 5.0
    assert gate.confidence(prev, drifted, consumed=2) < 0.9


def test_reuse_gate_horizon_scales_with_confidence():
    from lerobot.rollout.inference import StreamingReuseGate

    gate = StreamingReuseGate(min_horizon=2, max_horizon=12, overlap=4)
    assert gate.horizon_for(1.0, chunk_len=20) == 12
    assert gate.horizon_for(0.0, chunk_len=20) == 2
    # Horizon can never exceed the actual chunk length.
    assert gate.horizon_for(1.0, chunk_len=5) == 5


# ---------------------------------------------------------------------------
# Factory wiring (call site)
# ---------------------------------------------------------------------------


def test_create_inference_engine_flashvla():
    from lerobot.rollout import (
        StreamingInferenceConfig,
        StreamingInferenceEngine,
        create_inference_engine,
    )

    assert StreamingInferenceConfig().type == "flashvla"

    engine = create_inference_engine(
        StreamingInferenceConfig(min_horizon=1, max_horizon=8, overlap=4),
        policy=_ChunkPolicy([torch.zeros(4, 2)]),
        preprocessor=_Identity(),
        postprocessor=_Identity(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features=DATASET_FEATURES,
        ordered_action_keys=ORDERED_KEYS,
        task="pick",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, StreamingInferenceEngine)


# ---------------------------------------------------------------------------
# Reuse behavior (integration through the engine)
# ---------------------------------------------------------------------------


def test_stable_trajectory_reuses_actions_and_cuts_inferences():
    # A steady trajectory: every decode returns the same constant chunk, so the
    # previous chunk's un-executed tail matches the fresh chunk's head exactly.
    # The gate reads that agreement as full confidence and keeps reuse at the
    # max horizon, so a run of ticks costs far fewer decodes than ticks.
    chunk = torch.full((6, 2), 0.5)
    engine = _make_engine([chunk], min_horizon=1, max_horizon=4, overlap=4)
    engine.start()

    actions = [engine.get_action(_obs()) for _ in range(12)]

    assert all(a is not None for a in actions)
    # 12 ticks served by far fewer than 12 policy calls: reuse happened.
    assert engine._policy.calls < 12
    assert engine.stats["inferences"] == engine._policy.calls
    assert engine.stats["reuse_ratio"] > 0.0
    assert engine.stats["actions_per_inference"] > 1.0


def test_task_change_drops_buffer_and_forces_decode():
    chunk = torch.ones(6, 2)
    engine = _make_engine([chunk], min_horizon=1, max_horizon=6, overlap=4)
    engine.start()

    engine.get_action(_obs())  # first decode fills the buffer
    calls_after_first = engine._policy.calls
    assert engine._buffer  # actions are buffered for reuse

    engine.set_task("place")  # queued actions are now stale
    engine.get_action(_obs())

    # The stale buffer was dropped and a fresh chunk decoded under the new task.
    assert engine._policy.calls == calls_after_first + 1
    assert engine.dispatched_task == "place"


def test_reset_clears_streaming_state():
    engine = _make_engine([torch.ones(6, 2)], max_horizon=6)
    engine.start()
    engine.get_action(_obs())
    assert engine._buffer

    engine.reset()
    assert engine._buffer == []
    assert engine._prev_chunk is None


def test_non_chunking_policy_is_rejected():
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.rollout.inference import StreamingInferenceEngine

    # A policy that inherits the base predict_action_chunk (i.e. does not override
    # it) must be rejected up front rather than failing deep inside a decode.
    class _NonChunkingPolicy:
        predict_action_chunk = PreTrainedPolicy.predict_action_chunk

    with pytest.raises(ValueError, match="chunking policy"):
        StreamingInferenceEngine(
            policy=_NonChunkingPolicy(),
            preprocessor=_Identity(),
            postprocessor=_Identity(),
            dataset_features=DATASET_FEATURES,
            ordered_action_keys=ORDERED_KEYS,
            task="pick",
            device="cpu",
            robot_type="mock",
        )
