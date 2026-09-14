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

"""Failure-aware readout over frozen VLA-JEPA world-model predictive states.

Adapted from FARM — *Reading Failure Signals from the Internal Predictive
States of a Frozen Robotic World Model* (arXiv:2609.11445). The paper's claim
is that the internal predictive states of a *frozen* robotic world model
already carry directly decodable failure information: a tiny supervised readout
(~34k parameters) trained on those states produces step-wise failure scores and
a causal trajectory risk, with the predictive backbone left untouched.

This module ports that readout for LeRobot's VLA-JEPA policy. It consumes the
frozen ``ActionConditionedVideoPredictor`` output (``predicted_states``)
together with its shift-by-one target (``gt_states``) and decodes a scalar
failure probability per sample — the same I/O contract the paper describes
(predictor latent states / prediction residuals in, failure-risk signal out).

Scoped to the readout itself: training the readout on labelled rollouts and the
paper's separate benchmark suite belong to a downstream PR.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FailureReadout(nn.Module):
    """Lightweight supervised readout decoding failure risk from frozen states.

    The head reads two pooled ``state_dim`` summaries of the frozen predictor
    output: the predictive state itself (FARM's core signal) and the absolute
    prediction residual ``|predicted - target|``. It maps their concatenation
    through a two-layer MLP to a single failure logit per sample. Only this head
    is trained; the world-model backbone stays frozen.

    Args:
        state_dim: Feature dimension of the world-model predictive states
            (``ActionConditionedVideoPredictor`` output width).
        hidden_dim: Width of the readout's hidden layer. The default keeps the
            head near the paper's ~34k-parameter budget for typical
            ``state_dim`` values.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def features(self, predicted_states: Tensor, gt_states: Tensor) -> Tensor:
        """Pool ``[B, N, D]`` states into the ``[B, 2 * state_dim]`` readout input."""
        residual = (predicted_states - gt_states).abs().mean(dim=1)
        pooled_state = predicted_states.mean(dim=1)
        return torch.cat([pooled_state, residual], dim=-1)

    def forward(self, predicted_states: Tensor, gt_states: Tensor) -> Tensor:
        """Return per-sample failure logits with shape ``[B]``."""
        feats = self.features(predicted_states.float(), gt_states.float())
        return self.net(feats).squeeze(-1)

    def failure_score(self, predicted_states: Tensor, gt_states: Tensor) -> Tensor:
        """Return per-sample failure probabilities in ``[0, 1]`` with shape ``[B]``."""
        return torch.sigmoid(self.forward(predicted_states, gt_states))


def causal_trajectory_risk(step_scores: Tensor, mode: str = "cummax") -> Tensor:
    """Aggregate step-wise failure scores into a causal trajectory risk.

    The risk at step ``t`` depends only on steps ``<= t`` (causal, matching an
    online monitor that has not yet seen the future). ``cummax`` yields a
    monotone non-decreasing risk that latches onto the first alarming step;
    ``mean`` yields the running average.

    Args:
        step_scores: Per-step failure probabilities, shape ``[T]`` or ``[B, T]``.
        mode: ``"cummax"`` or ``"mean"``.

    Returns:
        Trajectory risk with the same shape as ``step_scores``.
    """
    squeeze = step_scores.ndim == 1
    if squeeze:
        step_scores = step_scores.unsqueeze(0)

    if mode == "cummax":
        risk = torch.cummax(step_scores, dim=1).values
    elif mode == "mean":
        counts = torch.arange(1, step_scores.shape[1] + 1, device=step_scores.device)
        risk = torch.cumsum(step_scores, dim=1) / counts
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'cummax' or 'mean')")

    return risk.squeeze(0) if squeeze else risk
