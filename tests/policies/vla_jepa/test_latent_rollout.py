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

"""Tests for the SG-JEPA autoregressive latent-rollout loss.

These exercise the rollout loss against the real
:class:`~lerobot.policies.vla_jepa.world_model.ActionConditionedVideoPredictor`
(a non-new module) to prove the wiring, rather than a stub.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.vla_jepa.latent_rollout import semigroup_rollout_loss
from lerobot.policies.vla_jepa.world_model import ActionConditionedVideoPredictor

_ACTION_EMBED_DIM = 8


def _make_predictor(
    embed_dim: int = 8,
    predictor_embed_dim: int = 24,
    num_action_tokens: int = 2,
    tokens_per_frame: int = 1,
) -> ActionConditionedVideoPredictor:
    return ActionConditionedVideoPredictor(
        num_frames=1,
        img_size=(1, tokens_per_frame),
        patch_size=1,
        tubelet_size=1,
        embed_dim=embed_dim,
        action_embed_dim=_ACTION_EMBED_DIM,
        predictor_embed_dim=predictor_embed_dim,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        num_action_tokens_per_step=num_action_tokens,
    )


@pytest.mark.parametrize(
    "batch,num_steps,tokens_per_frame,embed_dim",
    [
        (1, 2, 1, 8),
        (2, 3, 4, 8),
        (4, 5, 2, 16),
    ],
)
def test_rollout_loss_scalar_and_finite(
    batch: int, num_steps: int, tokens_per_frame: int, embed_dim: int
) -> None:
    predictor = _make_predictor(embed_dim=embed_dim, tokens_per_frame=tokens_per_frame)
    actions_per_frame = 2
    init = torch.randn(batch, tokens_per_frame, embed_dim)
    target = torch.randn(batch, tokens_per_frame * num_steps, embed_dim)
    actions = torch.randn(batch, actions_per_frame * num_steps, _ACTION_EMBED_DIM)

    loss = semigroup_rollout_loss(
        predictor,
        init,
        target,
        actions,
        tokens_per_frame=tokens_per_frame,
        actions_per_frame=actions_per_frame,
        num_steps=num_steps,
    )
    assert loss.shape == ()
    assert torch.isfinite(loss).all()


def test_rollout_loss_reduction_none_is_per_sample() -> None:
    batch, num_steps, tokens_per_frame, embed_dim = 3, 2, 4, 8
    predictor = _make_predictor(embed_dim=embed_dim, tokens_per_frame=tokens_per_frame)
    actions_per_frame = 2
    init = torch.randn(batch, tokens_per_frame, embed_dim)
    target = torch.randn(batch, tokens_per_frame * num_steps, embed_dim)
    actions = torch.randn(batch, actions_per_frame * num_steps, _ACTION_EMBED_DIM)

    per_sample = semigroup_rollout_loss(
        predictor,
        init,
        target,
        actions,
        tokens_per_frame=tokens_per_frame,
        actions_per_frame=actions_per_frame,
        num_steps=num_steps,
        reduction="none",
    )
    assert per_sample.shape == (batch,)
    assert torch.isfinite(per_sample).all()
    # The scalar reduction is the batch-mean of the per-sample losses.
    scalar = semigroup_rollout_loss(
        predictor,
        init,
        target,
        actions,
        tokens_per_frame=tokens_per_frame,
        actions_per_frame=actions_per_frame,
        num_steps=num_steps,
    )
    assert torch.allclose(per_sample.mean(), scalar, atol=1e-6)


def test_rollout_backprops_through_whole_chain() -> None:
    """Multi-step rollout gradients must reach the predictor (semigroup consistency)."""
    predictor = _make_predictor(tokens_per_frame=2)
    actions_per_frame = 2
    init = torch.randn(2, 2, 8)
    target = torch.randn(2, 2 * 3, 8)
    actions = torch.randn(2, actions_per_frame * 3, _ACTION_EMBED_DIM)

    loss = semigroup_rollout_loss(
        predictor,
        init,
        target,
        actions,
        tokens_per_frame=2,
        actions_per_frame=actions_per_frame,
        num_steps=3,
    )
    loss.backward()
    grads = [p.grad for p in predictor.parameters() if p.grad is not None]
    assert grads, "rollout loss produced no predictor gradients"
    assert any(g.abs().sum() > 0 for g in grads)


def test_rollout_differs_from_single_step_teacher_forcing() -> None:
    """Composing the predictor on its own output is a distinct signal from one teacher-forced pass."""
    torch.manual_seed(0)
    predictor = _make_predictor(tokens_per_frame=2)
    actions_per_frame = 2
    num_steps = 3
    init = torch.randn(2, 2, 8)
    target = torch.randn(2, 2 * num_steps, 8)
    actions = torch.randn(2, actions_per_frame * num_steps, _ACTION_EMBED_DIM)

    rollout = semigroup_rollout_loss(
        predictor,
        init,
        target,
        actions,
        tokens_per_frame=2,
        actions_per_frame=actions_per_frame,
        num_steps=num_steps,
    )

    # Single-step teacher forcing: every predicted frame is conditioned on the *ground-truth*
    # previous frame, computed in one frame-causal pass (the pre-existing world-model objective).
    teacher_context = torch.cat([init, target[:, : 2 * (num_steps - 1), :]], dim=1)
    with torch.no_grad():
        single_pass = predictor(teacher_context, actions)
    single_step = F.l1_loss(single_pass, target)

    assert not torch.allclose(rollout, single_step)


def test_rollout_clamps_to_available_frames() -> None:
    predictor = _make_predictor(tokens_per_frame=1)
    actions_per_frame = 2
    # Only 2 ground-truth frames available; asking for 5 steps must clamp, not error.
    init = torch.randn(1, 1, 8)
    target = torch.randn(1, 1 * 2, 8)
    actions = torch.randn(1, actions_per_frame * 2, _ACTION_EMBED_DIM)

    loss = semigroup_rollout_loss(
        predictor,
        init,
        target,
        actions,
        tokens_per_frame=1,
        actions_per_frame=actions_per_frame,
        num_steps=5,
    )
    assert torch.isfinite(loss).all()
