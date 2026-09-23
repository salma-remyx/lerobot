#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the Residual Semantic Steering language-grounding regularizer.

These exercise the wiring between the SmolVLA config knob (a non-new module) and the
``language_grounding_penalty`` proxy, using a lightweight stub that mimics the
``VLAFlowMatching`` velocity contract so no VLM weights / CUDA are required.
"""

import torch

# Import from a non-new module: the config that carries the new opt-in knobs.
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.language_grounding import language_grounding_penalty


class _StubFlowMatching:
    """Minimal stand-in for ``VLAFlowMatching`` for the grounding regularizer.

    The predicted velocity is a constant vision-driven term plus a language-driven
    term scaled by ``lang_gain``. Zeroing the language mask (the null-conditioned pass
    the regularizer performs) removes the language term, so ``lang_gain`` controls
    exactly how much of the velocity is attributable to language.
    """

    def __init__(self, lang_gain: float):
        self.lang_gain = lang_gain

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, bsize, device):
        return torch.full((bsize,), 0.5, device=device)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, return_velocity=False
    ):
        b, t, d = actions.shape
        vision = torch.ones(b, t, d)
        # Attended language tokens (mask == 1) contribute; a zeroed mask contributes nothing.
        lang_signal = lang_masks.float().sum(dim=1).view(b, 1, 1)
        v_t = vision + self.lang_gain * lang_signal * torch.ones(b, t, d)
        losses = v_t**2
        if return_velocity:
            return losses, v_t
        return losses


def _dummy_inputs(batch_size=2, chunk=4, action_dim=3, lang_len=5):
    actions = torch.zeros(batch_size, chunk, action_dim)
    lang_masks = torch.ones(batch_size, lang_len, dtype=torch.bool)
    return {
        "images": None,
        "img_masks": None,
        "lang_tokens": torch.zeros(batch_size, lang_len, dtype=torch.long),
        "lang_masks": lang_masks,
        "state": None,
        "actions": actions,
    }


def test_config_exposes_disabled_grounding_by_default():
    """The new knobs exist on the existing config and default to a no-op."""
    config = SmolVLAConfig()
    assert config.lang_grounding_weight == 0.0
    assert 0.0 < config.lang_grounding_margin < 1.0


def test_collapsed_model_is_penalized():
    """A model whose velocity ignores language (lang_gain=0) gets steered (penalty > 0)."""
    margin = SmolVLAConfig().lang_grounding_margin
    inputs = _dummy_inputs()
    penalty, metrics = language_grounding_penalty(
        _StubFlowMatching(lang_gain=0.0),
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        inputs["actions"],
        margin=margin,
    )
    # Velocity is identical with/without language -> zero steering ratio -> penalty == margin.
    assert torch.allclose(penalty, torch.full_like(penalty, margin))
    assert metrics["lang_steer_ratio"] == 0.0
    assert metrics["lang_grounding_penalty"] > 0.0


def test_language_sensitive_model_is_not_penalized():
    """A model whose velocity strongly depends on language incurs no penalty."""
    margin = SmolVLAConfig().lang_grounding_margin
    inputs = _dummy_inputs()
    penalty, metrics = language_grounding_penalty(
        _StubFlowMatching(lang_gain=1.0),
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        inputs["actions"],
        margin=margin,
    )
    assert metrics["lang_steer_ratio"] > margin
    assert torch.allclose(penalty, torch.zeros_like(penalty))


def test_penalty_is_per_sample_and_differentiable():
    """Penalty is shape (batch,) and carries gradient so it can steer training."""
    inputs = _dummy_inputs(batch_size=3)
    weight = torch.nn.Parameter(torch.ones(1))

    class _GradStub(_StubFlowMatching):
        def forward(
            self,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            return_velocity=False,
        ):
            b, t, d = actions.shape
            lang_signal = lang_masks.float().sum(dim=1).view(b, 1, 1)
            v_t = torch.ones(b, t, d) + weight * self.lang_gain * lang_signal
            losses = v_t**2
            if return_velocity:
                return losses, v_t
            return losses

    penalty, _ = language_grounding_penalty(
        _GradStub(lang_gain=0.05),
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        inputs["actions"],
        margin=0.5,
    )
    assert penalty.shape == (3,)
    assert penalty.requires_grad
    penalty.sum().backward()
    assert weight.grad is not None
