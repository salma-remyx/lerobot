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

"""Tests for world-model feature-alignment distillation (arXiv:2609.24682).

Covers the standalone projector/loss module and its wiring into ``VLAJEPAModel.forward`` /
``VLAJEPAPolicy.forward`` via the fake-backbone fixture. The last test pins the paper's central
claim in this codebase: the alignment projector is training-only and never touches the inference
path, so the deployed policy is byte-for-byte the undistilled baseline.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.filterwarnings(
    "ignore:In CPU autocast, but the target dtype is not supported:UserWarning"
)

from conftest import make_config, make_inference_batch, make_train_batch, set_seed_all  # noqa: E402
from lerobot.policies.vla_jepa.feature_alignment import WorldModelFeatureAligner  # noqa: E402
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # noqa: E402


def _alignment_config():
    """A CPU test config with feature-alignment enabled and a tiny projector."""
    config = make_config()
    config.enable_feature_alignment = True
    config.feature_alignment_hidden_dim = 8
    return config


# ---------------------------------------------------------------------------
# Standalone projector + loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_aligner_loss_shape_and_range(reduction: str) -> None:
    set_seed_all(0)
    aligner = WorldModelFeatureAligner(policy_dim=16, world_model_dim=8, hidden_dim=8)
    policy_tokens = torch.randn(4, 3, 16)
    target = torch.randn(4, 8)

    loss = aligner(policy_tokens, target, reduction=reduction)

    if reduction == "none":
        assert tuple(loss.shape) == (4,)
    else:
        assert loss.shape == ()
    # Cosine distance (1 - cos) lives in [0, 2].
    assert torch.all(loss >= 0.0) and torch.all(loss <= 2.0)
    assert torch.isfinite(loss).all()


def test_aligner_zero_loss_when_projection_matches_target() -> None:
    """Identical (normalized) directions give ~0 cosine distance; opposite gives ~2."""
    aligner = WorldModelFeatureAligner(policy_dim=8, world_model_dim=8, hidden_dim=8)
    tokens = torch.ones(1, 1, 8)
    projected = torch.nn.functional.normalize(aligner.projector(tokens.mean(dim=1)), dim=-1)

    aligned = aligner(tokens, projected, reduction="mean")
    opposed = aligner(tokens, -projected, reduction="mean")
    assert aligned.item() == pytest.approx(0.0, abs=1e-5)
    assert opposed.item() == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Wiring into the policy forward pass
# ---------------------------------------------------------------------------


def test_forward_includes_align_loss_and_backprops(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(_alignment_config())
    policy.train()

    loss, logs = policy.forward(make_train_batch())

    assert policy.model.feature_aligner is not None
    assert "align_loss" in logs
    assert logs["align_loss"] >= 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert any(
        p.grad is not None for p in policy.model.feature_aligner.parameters() if p.requires_grad
    ), "feature-alignment projector received no gradient"


def test_align_loss_absent_when_disabled(patch_vla_jepa_external_models: None) -> None:
    """Default config keeps the aligner off, so the loss dict/logs are unchanged."""
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.train()

    _, logs = policy.forward(make_train_batch())
    assert policy.model.feature_aligner is None
    assert "align_loss" not in logs
    assert set(logs) == {"action_loss", "wm_loss", "loss"}


@torch.no_grad()
def test_projector_is_discarded_at_inference(patch_vla_jepa_external_models: None) -> None:
    """The paper's core claim: the deployed policy is identical to the baseline.

    Mutating the alignment projector must not change predicted actions, proving it stays out of
    the inference path even though it is present on the module.
    """
    set_seed_all(42)
    policy = VLAJEPAPolicy(_alignment_config())
    policy.eval()
    batch = make_inference_batch()

    set_seed_all(123)
    before = policy.predict_action_chunk(batch)

    # Blow up the projector weights; a policy that used them at inference would move.
    for param in policy.model.feature_aligner.parameters():
        param.mul_(1000.0)

    set_seed_all(123)
    after = policy.predict_action_chunk(batch)

    assert torch.allclose(before, after, atol=1e-6)
