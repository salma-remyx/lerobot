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

"""World-model feature-alignment distillation for VLA-JEPA.

Adapted from "Think Like a World Model, Act Like a VLA: Distilling World-Model
Representations into Compact Robot Policies" (arXiv:2609.24682).

The paper's insight is that a world model's grounding lives in its *internal features*, not in
the act of generating the future. Rolling a world model forward costs seconds per decision and
is unusable in the control loop, but the grounding can be inherited without the generative
machinery: run the frozen world model over the training frames once, cache its features, and
add a single feature-alignment term that teaches the compact policy to agree with them. The
alignment projector is used only during training and never at inference, so the deployed policy
is identical to the undistilled baseline and every gain is attributable to the representation
rather than to added capacity or test-time compute.

This module provides that training-only projector and its alignment loss. The frozen
world-model features it aligns against are produced by the V-JEPA encoder already living on
``VLAJEPAModel`` (run under ``no_grad``); reusing that in-graph frozen encoder as the teacher is
the target-native stand-in for the paper's separate on-disk feature cache.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class WorldModelFeatureAligner(nn.Module):
    """Projects policy tokens into the world model's feature space and scores their agreement.

    A lightweight MLP maps the pooled policy representation to the frozen world model's feature
    dimension; the loss is a cosine distance (``1 - cosine_similarity``) between the L2-normalized
    projection and the pooled world-model features. Cosine distance is scale-invariant, so the
    objective aligns the *direction* of the representation and does not chase the world model's
    feature magnitude, which carries no grounding.

    Args:
        policy_dim: Hidden size of the policy tokens handed in (e.g. the Qwen hidden size).
        world_model_dim: Feature size of the pooled frozen world-model features.
        hidden_dim: Width of the projector's hidden layer.
    """

    def __init__(self, policy_dim: int, world_model_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(policy_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, world_model_dim),
        )

    def forward(self, policy_tokens: Tensor, target_features: Tensor, reduction: str = "mean") -> Tensor:
        """Feature-alignment loss between pooled policy tokens and frozen world-model features.

        Args:
            policy_tokens: ``[B, N, policy_dim]`` student tokens (e.g. the embodied-action tokens
                that condition the action head, so the grounding flows into the inference path).
            target_features: ``[B, world_model_dim]`` pooled frozen world-model features.
            reduction: ``"mean"`` for a scalar loss, ``"none"`` for a per-sample ``(B,)`` loss so
                the caller can reweight samples (matches the RA-BC ``reduction`` contract of the
                other VLA-JEPA loss terms).

        Returns:
            Scalar loss (``reduction="mean"``) or per-sample loss ``(B,)`` (``reduction="none"``).
        """
        pooled = policy_tokens.float().mean(dim=1)  # [B, policy_dim]
        projected = F.normalize(self.projector(pooled), dim=-1)  # [B, world_model_dim]
        target = F.normalize(target_features.float(), dim=-1)  # [B, world_model_dim]
        per_sample = 1.0 - (projected * target).sum(dim=-1)  # cosine distance, [B]
        if reduction == "none":
            return per_sample
        return per_sample.mean()
