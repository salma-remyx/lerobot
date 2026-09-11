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
"""Tests for Q-Planning value-guided action selection.

Covers both the pure collapsing rule (:func:`q_weighted_action`) and its wiring
into ``SACAlgorithm.select_action_q_weighted`` — the integration path that draws
candidates from the frozen actor and re-ranks them with the critic ensemble.
"""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy  # noqa: E402
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig  # noqa: E402
from lerobot.rl.algorithms.value_guided_selection import q_weighted_action  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


def _make_algorithm(state_dim: int = 10, action_dim: int = 6) -> SACAlgorithm:
    config = GaussianActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
    )
    config.validate_features()
    policy = GaussianActorPolicy(config=config)
    policy.eval()
    algorithm = SACAlgorithm(policy=policy, config=SACAlgorithmConfig.from_policy_config(config))
    return algorithm


# ===========================================================================
# Pure collapsing rule
# ===========================================================================


def test_q_weighted_average_is_convex_combination():
    """Softmax weighting must return a point inside the candidates' bounding box."""
    candidates = torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]])  # (3, 1, 2)
    q_values = torch.tensor([[0.5], [0.5], [0.5]])  # equal -> unweighted mean
    out = q_weighted_action(candidates, q_values, beta=1.0)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 1.0]]))


def test_low_beta_recovers_best_of_n():
    """beta <= 0 (and the beta -> 0 limit) selects the single highest-Q draw."""
    candidates = torch.tensor([[[0.0]], [[9.0]], [[3.0]]])  # (3, 1, 1)
    q_values = torch.tensor([[0.1], [5.0], [1.0]])  # candidate 1 dominates
    hard = q_weighted_action(candidates, q_values, beta=0.0)
    soft = q_weighted_action(candidates, q_values, beta=1e-4)
    assert torch.allclose(hard, torch.tensor([[9.0]]))
    assert torch.allclose(soft, torch.tensor([[9.0]]), atol=1e-3)


def test_large_beta_approaches_unweighted_mean():
    candidates = torch.tensor([[[0.0]], [[6.0]]])  # (2, 1, 1)
    q_values = torch.tensor([[10.0], [0.0]])
    out = q_weighted_action(candidates, q_values, beta=1e6)
    assert torch.allclose(out, torch.tensor([[3.0]]), atol=1e-3)


def test_shape_validation_raises():
    with pytest.raises(ValueError):
        q_weighted_action(torch.zeros(3, 2), torch.zeros(3), beta=1.0)
    with pytest.raises(ValueError):
        q_weighted_action(torch.zeros(3, 2, 4), torch.zeros(2, 3), beta=1.0)


# ===========================================================================
# Integration with SACAlgorithm (frozen BC draws + critic re-ranking)
# ===========================================================================


def test_select_action_q_weighted_shape():
    algorithm = _make_algorithm(state_dim=10, action_dim=6)
    obs = {OBS_STATE: torch.randn(4, 10)}
    action = algorithm.select_action_q_weighted(obs, num_action_samples=8, beta=1.0)
    assert action.shape == (4, 6)
    assert torch.isfinite(action).all()


def test_single_sample_passes_bc_draw_through():
    """With one draw the weight is 1, so the result is exactly the BC action.

    Seeding both the integrated call and a bare actor draw proves the wiring
    forwards the frozen policy's sample untouched (no critic influence for N=1).
    """
    algorithm = _make_algorithm(state_dim=10, action_dim=6)
    obs = {OBS_STATE: torch.randn(3, 10)}

    set_seed(0)
    out = algorithm.select_action_q_weighted(obs, num_action_samples=1, beta=1.0)

    set_seed(0)
    obs_features, _ = algorithm.get_observation_features(obs, obs)
    expected, _, _ = algorithm.policy.actor(obs, obs_features)

    assert torch.allclose(out, expected, atol=1e-6)


def test_does_not_modify_policy_weights():
    """Q-Planning inference re-ranks draws; it must leave the BC weights frozen."""
    algorithm = _make_algorithm()
    before = {n: p.clone() for n, p in algorithm.policy.actor.named_parameters()}

    obs = {OBS_STATE: torch.randn(4, 10)}
    algorithm.select_action_q_weighted(obs, num_action_samples=8, beta=1.0)

    for n, p in algorithm.policy.actor.named_parameters():
        assert torch.equal(p, before[n]), f"actor param '{n}' changed during inference"
