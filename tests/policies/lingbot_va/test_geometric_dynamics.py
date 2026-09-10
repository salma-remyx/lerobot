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

"""Tests for the geometry-aware future-prediction weighting (LingBot-VLA 2.0).

Covers the parameter-free geometric-saliency proxy and the config wiring that turns it
on. The config field is imported from the (non-new) ``configuration_lingbot_va`` module so
the wiring edit — the ``geometric_dynamics_weight`` knob that ``_flow_matching_loss`` reads —
is exercised through the public policy config.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.geometric_dynamics import (
    apply_geometric_weighting,
    geometric_saliency_weights,
)


def make_config(**overrides) -> LingBotVAConfig:
    kwargs = {"device": "cpu"}
    kwargs.update(overrides)
    return LingBotVAConfig(**kwargs)


def _edge_latents() -> torch.Tensor:
    """A [B, C, F, H, W] target with a sharp vertical edge (right half hot)."""
    latents = torch.zeros(2, 3, 4, 8, 8)
    latents[..., 4:] = 1.0
    return latents


def test_config_default_is_disabled() -> None:
    # Default must reproduce the original loss: geometric weighting off.
    assert make_config().geometric_dynamics_weight == 0.0


def test_config_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="geometric_dynamics_weight"):
        make_config(geometric_dynamics_weight=-0.5)


def test_config_positive_weight_roundtrips() -> None:
    cfg = make_config(geometric_dynamics_weight=0.75)
    assert cfg.geometric_dynamics_weight == 0.75


def test_zero_strength_is_uniform() -> None:
    weights = geometric_saliency_weights(_edge_latents(), strength=0.0)
    assert weights.shape == (2, 1, 4, 8, 8)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_weights_are_mean_preserving_per_frame() -> None:
    weights = geometric_saliency_weights(_edge_latents(), strength=1.0)
    per_frame_mean = weights.mean(dim=(-2, -1))
    torch.testing.assert_close(per_frame_mean, torch.ones_like(per_frame_mean), atol=1e-5, rtol=1e-4)


def test_saliency_emphasizes_geometric_edges() -> None:
    weights = geometric_saliency_weights(_edge_latents(), strength=1.0)[0, 0, 0]
    # Column 3 straddles the 0->1 discontinuity; interior flat columns carry no gradient.
    assert weights[0, 3] > weights[0, 0]
    assert weights[0, 3] > weights[0, 6]


def test_apply_geometric_weighting_noop_when_disabled() -> None:
    error = torch.rand(2, 3, 4, 8, 8)
    weighted = apply_geometric_weighting(error, _edge_latents(), strength=0.0)
    assert weighted is error


def test_apply_geometric_weighting_redistributes_error() -> None:
    error = torch.ones(2, 3, 4, 8, 8)
    weighted = apply_geometric_weighting(error, _edge_latents(), strength=1.0)
    # Uniform error: the weight map is mean-preserving, so total mass is (approximately) conserved
    # while edge locations are emphasized relative to flat interior regions.
    torch.testing.assert_close(weighted.mean(), error.mean(), atol=1e-5, rtol=1e-4)
    assert weighted[0, 0, 0, 0, 3] > weighted[0, 0, 0, 0, 0]
