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

"""Geometry-aware weighting for the LingBot-VA future-prediction proxy task.

Adapted from the *predictive dynamics modeling* contribution of "From Foundation to
Application: Improving VLA Models in Practice" (LingBot-VLA 2.0, arXiv:2607.06403).
The paper formulates future prediction as a proxy task and enriches it with a
depth-estimation model that supplies *geometric cues* for temporal reasoning.

The LingBot-VA policy already trains a future-video-latent stream (the semantic /
video prior), so the missing ingredient here is the geometric signal. Rather than
bundling a learned depth estimator (a ~GB frozen model and a second forward pass per
step), this module supplies a **parameter-free geometric-saliency proxy** computed
directly on the VAE target latents: the spatial-gradient magnitude of the target
highlights depth discontinuities / object boundaries — the same structure a depth map
surfaces — and is used to up-weight the future-prediction loss in those regions.

The weighting is opt-in via ``LingBotVAConfig.geometric_dynamics_weight`` and is
mean-preserving per frame, so enabling it redistributes supervision toward
geometrically-informative regions without changing the overall loss scale.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def geometric_saliency_weights(target_latents: Tensor, strength: float, eps: float = 1e-6) -> Tensor:
    """Per-location geometric-saliency weights for the future-latent prediction loss.

    Computes the spatial-gradient magnitude of ``target_latents`` (a geometry proxy for the
    depth-discontinuity cues used in LingBot-VLA 2.0), normalizes it to unit spatial mean per
    ``(batch, frame)``, and blends it with a uniform map by ``strength``.

    Args:
        target_latents: Future VAE latents of shape ``[B, C, F, H, W]`` (the flow-matching
            target). Gradients are aggregated over the channel dimension.
        strength: Blend factor in ``[0, 1]``. ``0`` returns an all-ones map (no reweighting);
            ``1`` uses the full saliency map. Values ``> 1`` sharpen the emphasis further.
        eps: Numerical floor for the per-frame normalization.

    Returns:
        A weight tensor of shape ``[B, 1, F, H, W]`` broadcastable over channels, with unit
        spatial mean per ``(batch, frame)``. High-gradient (geometrically salient) locations get
        weights ``> 1`` and flat regions get weights ``< 1``.
    """
    if strength <= 0.0:
        return target_latents.new_ones((target_latents.shape[0], 1, *target_latents.shape[2:]))

    latents = target_latents.detach().float()
    # First-order spatial differences along height and width, padded back to the full grid.
    dh = F.pad(latents[..., 1:, :] - latents[..., :-1, :], (0, 0, 0, 1))
    dw = F.pad(latents[..., :, 1:] - latents[..., :, :-1], (0, 1))
    grad = (dh.pow(2) + dw.pow(2)).mean(dim=1, keepdim=True).sqrt()  # [B, 1, F, H, W]

    spatial_mean = grad.mean(dim=(-2, -1), keepdim=True)
    saliency = grad / (spatial_mean + eps)  # unit spatial mean per (B, F)
    weights = (1.0 - strength) + strength * saliency
    return weights.clamp_min(0.0).to(target_latents.dtype)


def apply_geometric_weighting(latent_error: Tensor, target_latents: Tensor, strength: float) -> Tensor:
    """Reweight the per-element future-prediction error by geometric saliency.

    Thin call-site helper wrapping :func:`geometric_saliency_weights`. Returns ``latent_error``
    unchanged when ``strength <= 0`` so the default configuration reproduces the original loss.

    Args:
        latent_error: Unreduced future-latent error, shape ``[B, C, F, H, W]``.
        target_latents: Flow-matching targets, shape ``[B, C, F, H, W]``.
        strength: Geometric-emphasis strength (see :func:`geometric_saliency_weights`).
    """
    if strength <= 0.0:
        return latent_error
    weights = geometric_saliency_weights(target_latents, strength).to(latent_error.dtype)
    return latent_error * weights
