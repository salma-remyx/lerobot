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

"""Value-guided action selection over behaviour-cloning draws.

Parameter-free inference-time mechanism from Q-Planning ("Beyond Imitation:
Self-Improving Robot Policies via Off-Policy Q-Planning",
https://arxiv.org/abs/2608.21204). Given a set of candidate actions sampled
from a (frozen) behaviour-cloning policy and the Q-value the critic assigns to
each, collapse the draws into a single action via a Q-weighted average.

The key asymmetry Q-Planning exploits: the Q-function estimates *value* rather
than imitating actions, so it can re-rank the BC policy's own proposals at test
time without touching the BC weights. This module only implements the
collapsing step (a pure tensor op); drawing the candidates and scoring them
with the critic stays with the algorithm that owns those networks.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["q_weighted_action"]


def q_weighted_action(action_candidates: Tensor, q_values: Tensor, beta: float = 1.0) -> Tensor:
    """Collapse BC action draws into one action via a Q-weighted average.

    Implements the single-step selection rule of Q-Planning: weight each
    candidate by ``softmax(Q / beta)`` over the sample dimension and return the
    resulting convex combination. ``beta`` is a temperature that interpolates
    between two familiar extremes:

    - ``beta -> 0`` sharpens the softmax onto the highest-Q draw, recovering the
      greedy **Best-of-N** selection. ``beta <= 0`` is treated as this limit
      exactly (a hard ``argmax``), which also avoids a division by zero.
    - large ``beta`` flattens the weights toward uniform, recovering the plain
      (unweighted) BC sample mean.

    Args:
        action_candidates: Sampled actions, shape ``(num_samples, batch, action_dim)``.
        q_values: Critic value for each candidate, shape ``(num_samples, batch)``.
        beta: Softmax temperature over the candidate dimension.

    Returns:
        The selected action, shape ``(batch, action_dim)``.
    """
    if action_candidates.ndim != 3:
        raise ValueError(
            f"action_candidates must be (num_samples, batch, action_dim); got {tuple(action_candidates.shape)}"
        )
    if q_values.shape != action_candidates.shape[:2]:
        raise ValueError(
            f"q_values must be (num_samples, batch)={tuple(action_candidates.shape[:2])}; "
            f"got {tuple(q_values.shape)}"
        )

    if beta <= 0.0:
        # Greedy Best-of-N: pick the single highest-Q draw per batch element.
        best = q_values.argmax(dim=0)  # (batch,)
        index = best.view(1, -1, 1).expand(1, -1, action_candidates.shape[-1])
        return action_candidates.gather(0, index).squeeze(0)

    # Softmax over the sample dimension, with the usual max-subtraction for
    # numerical stability. Weights broadcast across the action dimension.
    weights = torch.softmax(q_values / beta, dim=0)  # (num_samples, batch)
    return (weights.unsqueeze(-1) * action_candidates).sum(dim=0)
