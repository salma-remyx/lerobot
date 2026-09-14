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

"""Unit tests for the FARM failure readout (arXiv:2609.11445).

These exercise the integration contract the readout relies on: the *existing*
``ActionConditionedVideoPredictor`` produces the frozen predictive states, and
the readout decodes a scalar failure risk from them — exactly the path
``VLAJEPAModel.predict_failure_score`` wires together internally.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.vla_jepa.failure_readout import FailureReadout, causal_trajectory_risk
# ActionConditionedVideoPredictor is the existing (non-new) world-model predictor the
# readout consumes in production, via VLAJEPAModel.predict_failure_score.
from lerobot.policies.vla_jepa.world_model import ActionConditionedVideoPredictor

_ACTION_EMBED_DIM = 8


def _make_predictor(embed_dim: int, tokens_per_frame: int) -> ActionConditionedVideoPredictor:
    return ActionConditionedVideoPredictor(
        num_frames=1,
        img_size=(1, tokens_per_frame),
        patch_size=1,
        tubelet_size=1,
        embed_dim=embed_dim,
        action_embed_dim=_ACTION_EMBED_DIM,
        predictor_embed_dim=24,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        num_action_tokens_per_step=2,
    )


@pytest.mark.parametrize("batch,num_steps,tokens_per_frame,embed_dim", [(1, 2, 1, 8), (3, 3, 4, 16)])
def test_readout_decodes_scores_from_predictor_states(
    batch: int, num_steps: int, tokens_per_frame: int, embed_dim: int
) -> None:
    """Predictor output -> readout is the exact wiring of predict_failure_score."""
    predictor = _make_predictor(embed_dim=embed_dim, tokens_per_frame=tokens_per_frame)
    readout = FailureReadout(state_dim=embed_dim)

    frame_tokens = torch.randn(batch, num_steps * tokens_per_frame, embed_dim)
    action_tokens = torch.randn(batch, num_steps * 2, _ACTION_EMBED_DIM)
    predicted_states = predictor(frame_tokens, action_tokens)
    gt_states = torch.randn_like(predicted_states)

    scores = readout.failure_score(predicted_states, gt_states)
    assert tuple(scores.shape) == (batch,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()

    logits = readout(predicted_states, gt_states)
    assert tuple(logits.shape) == (batch,)


def test_readout_is_a_tiny_head() -> None:
    """FARM's premise: the readout is a tiny head (~34k params), not a second model."""
    readout = FailureReadout(state_dim=512)
    n_params = sum(p.numel() for p in readout.parameters())
    assert 20_000 < n_params < 50_000


def test_causal_trajectory_risk_cummax_is_monotone() -> None:
    step_scores = torch.tensor([0.1, 0.4, 0.2, 0.9, 0.3])
    risk = causal_trajectory_risk(step_scores, mode="cummax")
    assert tuple(risk.shape) == step_scores.shape
    assert torch.equal(risk, torch.tensor([0.1, 0.4, 0.4, 0.9, 0.9]))
    # Causal + monotone: risk never decreases as the trajectory progresses.
    assert (risk[1:] >= risk[:-1]).all()


def test_causal_trajectory_risk_mean_and_batched() -> None:
    step_scores = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    risk = causal_trajectory_risk(step_scores, mode="mean")
    assert torch.allclose(risk, torch.tensor([[0.0, 0.5], [0.5, 0.5]]))


def test_causal_trajectory_risk_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        causal_trajectory_risk(torch.zeros(3), mode="bogus")
