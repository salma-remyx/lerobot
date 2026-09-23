#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""Residual Semantic Steering (RSS) grounding for language-conditioned VLA training.

Adapted from *Stable Language Guidance for Vision-Language-Action Models*
(arXiv:2601.04052). VLA policies are prone to "modality collapse": strong visual
priors overwhelm the sparse language signal, so the policy overfits to a specific
instruction phrasing and effectively ignores the instruction. The paper counters
this with a probabilistic regularizer that steers the action distribution by the
*residual* attributable to language.

This module implements that core mechanism with a parameter-free proxy that fits
the existing SmolVLA flow-matching path: instead of the paper's learned estimator,
it ablates language by masking the language tokens (a null-conditioned forward) and
measures how much of the predicted action velocity is attributable to the
instruction. A hinge penalty steers the policy so that at least ``margin`` of the
velocity magnitude is language-driven. The penalty is opt-in (disabled when the
config weight is 0) and adds one extra forward pass when enabled.
"""

from __future__ import annotations

import torch


def language_grounding_penalty(
    model,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    actions,
    *,
    margin: float,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the Residual Semantic Steering penalty for one training batch.

    The language-conditioned velocity is compared against a language-ablated
    ("null") velocity obtained by zeroing the language attention mask. The share of
    the conditioned velocity magnitude that is attributable to language (the residual
    ratio) is hinged against ``margin``: samples whose predictions barely change when
    language is removed incur a positive penalty, steering the policy to keep language
    influential.

    Args:
        model: A ``VLAFlowMatching`` instance exposing ``forward(..., return_velocity=True)``
            plus ``sample_noise`` / ``sample_time`` helpers.
        images, img_masks, lang_tokens, lang_masks, state, actions: The same tensors
            already prepared for the main flow-matching loss.
        margin: Target minimum fraction (0..1) of the action velocity magnitude that
            should be driven by the language instruction.
        noise, time: Optional shared flow-matching noise / timestep. When omitted they
            are sampled once and reused across both conditionings so the only
            difference between the two forward passes is the language signal.

    Returns:
        A ``(penalty, metrics)`` tuple. ``penalty`` is a per-sample tensor of shape
        ``(batch_size,)`` suitable for either mean or per-sample reduction; ``metrics``
        holds scalar diagnostics for logging.
    """
    if noise is None:
        noise = model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = model.sample_time(actions.shape[0], actions.device)

    # Language-conditioned velocity.
    _, v_cond = model.forward(
        images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, return_velocity=True
    )
    # Language-ablated velocity: masking every language token removes the instruction
    # from attention while keeping vision/state/time identical (a null-conditioned pass).
    null_masks = torch.zeros_like(lang_masks)
    _, v_null = model.forward(
        images, img_masks, lang_tokens, null_masks, state, actions, noise, time, return_velocity=True
    )

    # Residual of the action velocity attributable to language, per sample.
    residual = (v_cond - v_null).flatten(start_dim=1).norm(dim=-1)
    cond_norm = v_cond.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-6)
    steer_ratio = residual / cond_norm  # fraction of the velocity driven by language, per sample

    # Hinge: only penalize samples below the target margin; once language is influential
    # enough the term is zero and training reverts to the standard flow-matching loss.
    penalty = torch.relu(margin - steer_ratio)

    metrics = {
        "lang_steer_ratio": steer_ratio.mean().item(),
        "lang_grounding_penalty": penalty.mean().item(),
    }
    return penalty, metrics
