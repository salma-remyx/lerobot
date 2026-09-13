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

"""Autoregressive latent-rollout loss for the VLA-JEPA world model.

Adapted from *Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics
Generalization* (https://arxiv.org/abs/2609.10464). SG-JEPA trains an
action-conditioned latent predictor with an *autoregressive rollout* rather than
a single teacher-forced step: the one-step operator is composed ``k`` times on
its own predicted latents and matched against the ``k``-step-ahead encoder
latents. Because the multi-step error is back-propagated through the whole
rollout chain, the model is trained against the error accumulation that a
single-step L1 objective never sees ("semigroup consistency": applying the
dynamics operator ``k`` times equals the ``k``-step operator).

This module keeps that core mechanism at full fidelity on the repo's existing
:class:`~lerobot.policies.vla_jepa.world_model.ActionConditionedVideoPredictor`.
The paper's gravitational-field physics parameter is supplied through the
predictor's existing ``action_tokens`` conditioning path (no new data shape),
and the paper's separate benchmark / independent diffusion-policy evaluation is
left to downstream work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def semigroup_rollout_loss(
    predictor: nn.Module,
    init_states: Tensor,
    target_states: Tensor,
    action_tokens: Tensor,
    tokens_per_frame: int,
    actions_per_frame: int,
    num_steps: int,
    reduction: str = "mean",
) -> Tensor:
    """Autoregressive multi-step rollout loss for a frame-causal latent predictor.

    Starting from the first frame's latent, the predictor is applied ``num_steps``
    times, each time re-feeding its own predicted latent as the newest context
    frame, and each predicted latent is matched (L1) against the ground-truth
    encoder latent at that horizon. The loss is averaged over rollout steps.

    Gradients flow through the entire rollout chain (predictions are *not*
    detached between steps), which is what turns the objective from single-step
    prediction into semigroup consistency.

    Args:
        predictor: A frame-causal action-conditioned predictor with the
            ``ActionConditionedVideoPredictor`` interface: given ``frame_tokens``
            ``[B, F * tokens_per_frame, D]`` and ``action_tokens``
            ``[B, F * actions_per_frame, A]`` it returns per-frame next-latent
            predictions ``[B, F * tokens_per_frame, D]``.
        init_states: First-frame latent ``z_0`` of shape
            ``[B, tokens_per_frame, D]``.
        target_states: Ground-truth future latents ``z_1 .. z_K`` of shape
            ``[B, tokens_per_frame * K, D]`` (``K >= num_steps``).
        action_tokens: Per-step action conditioning ``a_0 .. a_{K-1}`` of shape
            ``[B, actions_per_frame * K, A]``.
        tokens_per_frame: Number of latent tokens per temporal frame.
        actions_per_frame: Number of action tokens per temporal step.
        num_steps: Number of autoregressive rollout steps to perform.
        reduction: ``"mean"`` returns a scalar; ``"none"`` returns a per-sample
            ``(B,)`` loss (mean over tokens/feature then over steps) for
            per-sample weighting.

    Returns:
        Scalar tensor for ``reduction="mean"`` or ``(B,)`` for ``reduction="none"``.
    """
    if num_steps < 1:
        raise ValueError(f"`num_steps` must be >= 1, got {num_steps}.")

    max_target_steps = target_states.shape[1] // tokens_per_frame
    max_action_steps = action_tokens.shape[1] // actions_per_frame
    usable_steps = min(num_steps, max_target_steps, max_action_steps)
    if usable_steps < 1:
        raise ValueError(
            "Not enough ground-truth frames or action tokens for a rollout step "
            f"(target frames={max_target_steps}, action steps={max_action_steps})."
        )

    context = init_states
    step_losses: list[Tensor] = []
    for step in range(usable_steps):
        n_ctx_frames = step + 1
        actions = action_tokens[:, : n_ctx_frames * actions_per_frame, :]
        predicted = predictor(context, actions)
        # Frame-causal predictor: the last frame's output is the next-latent prediction.
        next_latent = predicted[:, -tokens_per_frame:, :]
        target = target_states[:, step * tokens_per_frame : (step + 1) * tokens_per_frame, :]

        elementwise = F.l1_loss(next_latent, target, reduction="none")
        if reduction == "none":
            step_losses.append(elementwise.flatten(start_dim=1).mean(dim=1))
        else:
            step_losses.append(elementwise.mean())

        # Re-feed the prediction (with gradient) as the newest context frame.
        context = torch.cat([context, next_latent], dim=1)

    return torch.stack(step_losses, dim=0).mean(dim=0)
