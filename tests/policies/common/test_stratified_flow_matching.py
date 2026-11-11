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

"""Tests for the variance-reduced flow-matching samplers.

Two properties matter for training: every draw keeps the marginal distribution of the
i.i.d. sampler it replaces (so the expected loss is unchanged), and the draws are actually
correlated in the variance-reducing direction (one per quantile bin, +/- noise pairs).
The last two tests exercise the policy wiring, not just the samplers.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.common.flow_matching import sample_beta, sample_time_beta  # noqa: E402
from lerobot.policies.common.stratified_flow_matching import (  # noqa: E402
    beta_quantile,
    flow_variance_from_probe,
    noise_compute_objective,
    relative_noise_compute_cost,
    sample_noise_antithetic,
    sample_time_beta_batch_stratified,
    sample_time_beta_stratified,
    select_num_flow_samples,
)
from lerobot.policies.eo1.configuration_eo1 import EO1Config  # noqa: E402
from lerobot.policies.eo1.modeling_eo1 import EO1VisionFlowMatchingModel  # noqa: E402
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching  # noqa: E402

# openpi convention, shared by pi0/pi05/smolvla/eo1.
OPENPI_TIME_SAMPLING = {"alpha": 1.5, "beta": 1.0, "scale": 0.999, "offset": 0.001}


def test_beta_quantile_matches_beta_marginal():
    u = torch.rand(200_000)
    quantile_samples = beta_quantile(u, alpha=1.5, beta=1.0)
    torch.manual_seed(0)
    beta_samples = sample_beta(1.5, 1.0, 200_000, "cpu")

    # Same mean, and the same quantile at a few probability levels (Kolmogorov-style check).
    assert quantile_samples.mean().item() == pytest.approx(beta_samples.mean().item(), abs=2e-3)
    for level in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert quantile_samples.quantile(level).item() == pytest.approx(
            beta_samples.quantile(level).item(), abs=2e-3
        )
    # Beta(1.5, 1) has the closed form F^-1(u) = u ** (2/3).
    assert beta_quantile(torch.tensor(0.125), alpha=1.5, beta=1.0).item() == pytest.approx(0.125 ** (2 / 3))
    # MolmoAct2's shape, Beta(1, 1.5): F^-1(u) = 1 - (1 - u) ** (2/3).
    assert beta_quantile(torch.tensor(0.5), alpha=1.0, beta=1.5).item() == pytest.approx(1 - 0.5 ** (2 / 3))


def test_beta_quantile_rejects_unsupported_shapes():
    with pytest.raises(ValueError, match="beta=1 or alpha=1"):
        beta_quantile(torch.tensor(0.5), alpha=1.2, beta=1.3)
    with pytest.raises(ValueError, match="u in"):
        beta_quantile(torch.tensor(1.5), alpha=1.5, beta=1.0)


def _bin_edges(num_bins: int, *, alpha: float, beta: float, scale: float, offset: float) -> torch.Tensor:
    """Quantile-bin edges of a Beta time distribution, mapped through its affine."""
    edges = beta_quantile(torch.arange(num_bins + 1, dtype=torch.float32) / num_bins, alpha=alpha, beta=beta)
    return edges * scale + offset


def test_sample_time_beta_stratified_one_time_per_bin():
    torch.manual_seed(0)
    bsize, num_samples = 32, 8
    time = sample_time_beta_stratified(bsize, num_samples, "cpu", **OPENPI_TIME_SAMPLING)

    assert time.shape == (bsize, num_samples) and time.dtype == torch.float32
    edges = _bin_edges(num_samples, **OPENPI_TIME_SAMPLING)
    for k in range(num_samples):
        # Sample k of every observation sits in quantile bin k.
        assert torch.all((time[:, k] >= edges[k]) & (time[:, k] <= edges[k + 1] + 1e-6))


def test_sample_time_beta_batch_stratified_one_time_per_batch_element():
    torch.manual_seed(1)
    bsize = 64
    time = sample_time_beta_batch_stratified(bsize, "cpu", **OPENPI_TIME_SAMPLING)

    assert time.shape == (bsize,) and time.dtype == torch.float32
    edges = _bin_edges(bsize, **OPENPI_TIME_SAMPLING)
    for i in range(bsize):
        assert edges[i] <= time[i] <= edges[i + 1] + 1e-6


@pytest.mark.parametrize("shape,sample_axis", [((3, 64, 4, 2), 1), ((128, 3, 2), 0), ((3, 64, 4), -1)])
def test_sample_noise_antithetic_pairs_and_marginal(shape, sample_axis):
    torch.manual_seed(2)
    noise = sample_noise_antithetic(shape, "cpu", sample_axis=sample_axis)

    assert tuple(noise.shape) == shape and noise.dtype == torch.float32
    axis = sample_axis if sample_axis >= 0 else sample_axis + len(shape)
    num_samples = shape[axis]
    for k in range(0, num_samples - 1, 2):
        plus = noise.select(axis, k)
        minus = noise.select(axis, k + 1)
        assert torch.equal(minus, -plus)
    # Marginals stay standard normal.
    assert noise.mean().item() == pytest.approx(0.0, abs=0.05)
    assert noise.std().item() == pytest.approx(1.0, abs=0.05)


def test_stratified_times_cut_variance_of_the_mean_loss_estimate():
    # A flow-matching loss is strongly t-dependent, so the batch-mean estimate of it is
    # what stratification makes cheaper: same expectation, much lower variance.
    torch.manual_seed(5)
    bsize, num_draws = 32, 2000

    def loss(time):
        return (0.7 - time) ** 2

    iid = torch.stack(
        [loss(sample_time_beta(bsize, "cpu", **OPENPI_TIME_SAMPLING)).mean() for _ in range(num_draws)]
    )
    stratified = torch.stack(
        [
            loss(sample_time_beta_batch_stratified(bsize, "cpu", **OPENPI_TIME_SAMPLING)).mean()
            for _ in range(num_draws)
        ]
    )

    assert stratified.mean().item() == pytest.approx(iid.mean().item(), rel=0.05)
    assert stratified.var().item() < 0.1 * iid.var().item()


def test_antithetic_noise_cancels_odd_components_exactly():
    # The mechanism the paper relies on: whatever a flow sample contributes that is odd in
    # its noise draw sums to zero over a +/- pair.
    torch.manual_seed(6)
    noise = sample_noise_antithetic((16, 8, 4, 2), "cpu", sample_axis=1)
    odd = (noise**3).mean(dim=1) + 3.0 * noise.sum(dim=1)
    assert torch.all(odd == 0)


def test_sample_noise_antithetic_odd_sample_count_keeps_last_unpaired():
    torch.manual_seed(3)
    noise = sample_noise_antithetic((2, 5, 3), "cpu", sample_axis=1)

    assert tuple(noise.shape) == (2, 5, 3)
    for k in range(0, 4, 2):
        assert torch.equal(noise[:, k + 1], -noise[:, k])
    # The unpaired tail is a fresh draw, not a repeat or a negation of anything.
    assert not torch.equal(noise[:, 4], noise[:, 3])
    assert not torch.equal(noise[:, 4], -noise[:, 3])


def test_sample_noise_antithetic_validates_axis():
    with pytest.raises(ValueError, match="sample_axis"):
        sample_noise_antithetic((2, 0, 3), "cpu", sample_axis=1)


def test_noise_compute_model_reproduces_paper_constants():
    # Section 5.4 of arXiv:2611.14432: c_p = 21.4 ms, c_k = 0.30 ms, V_f / V_oa = 1.43.
    costs = {"prefix_cost": 21.4, "sample_cost": 0.30, "flow_variance": 1.43, "other_variance": 1.0}
    k_star = select_num_flow_samples(**costs)
    assert k_star == 10

    # J(K) is minimized at K*, and the paper's flat region [8, 14] is within 1.5%.
    assert noise_compute_objective(k_star, **costs) == pytest.approx(
        min(noise_compute_objective(k, **costs) for k in range(1, 30))
    )
    for k in (8, 12, 14):
        assert relative_noise_compute_cost(k, **costs) == pytest.approx(1.0, abs=0.015)
    # Far from K* the cost grows: single-sample flow matching is clearly worse here.
    assert relative_noise_compute_cost(1, **costs) > 1.4


def test_select_num_flow_samples_edges():
    # No flow variance to reduce -> single-sample training is already optimal.
    assert (
        select_num_flow_samples(prefix_cost=21.4, sample_cost=0.3, flow_variance=0.0, other_variance=1.0) == 1
    )
    with pytest.raises(ValueError, match="positive"):
        select_num_flow_samples(prefix_cost=0.0, sample_cost=0.3, flow_variance=1.0, other_variance=1.0)
    with pytest.raises(ValueError, match="other_variance"):
        select_num_flow_samples(prefix_cost=1.0, sample_cost=0.3, flow_variance=1.0, other_variance=0.0)


def test_flow_variance_from_probe_recovers_single_draw_variance():
    # A probe at K=15 whose flow-only resampling measures V_f / 15.
    assert flow_variance_from_probe(15, 0.1) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="probe_num_samples"):
        flow_variance_from_probe(0, 0.1)


def _openpi_model(model_cls, config_cls, **config_kwargs):
    """Build a policy model's sampler without loading a backbone: `sample_time` only reads config."""
    config = config_cls(**config_kwargs)
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = config
    return model


def _time_kwargs(config) -> dict:
    """The four openpi time-sampling parameters of a config (smolvla hardcodes them)."""
    return {
        "alpha": getattr(config, "time_sampling_beta_alpha", 1.5),
        "beta": getattr(config, "time_sampling_beta_beta", 1.0),
        "scale": getattr(config, "time_sampling_scale", 0.999),
        "offset": getattr(config, "time_sampling_offset", 0.001),
    }


@pytest.mark.parametrize(
    ("model_cls", "config_cls"),
    [
        (PI0Pytorch, PI0Config),
        (PI05Pytorch, PI05Config),
        (VLAFlowMatching, SmolVLAConfig),
        (EO1VisionFlowMatchingModel, EO1Config),
    ],
)
def test_openpi_models_sample_stratified_times_when_enabled(model_cls, config_cls):
    bsize = 64
    default_model = _openpi_model(model_cls, config_cls)
    stratified_model = _openpi_model(model_cls, config_cls, time_sampling_stratified=True)

    torch.manual_seed(4)
    default_time = default_model.sample_time(bsize, "cpu")
    torch.manual_seed(4)
    stratified_time = stratified_model.sample_time(bsize, "cpu")

    assert default_time.shape == stratified_time.shape == (bsize,)

    # Flag off: the sampler is exactly the i.i.d. one released checkpoints were trained with.
    kwargs = _time_kwargs(default_model.config)
    torch.manual_seed(4)
    expected_default = sample_time_beta(bsize, "cpu", **kwargs)
    assert torch.equal(default_time, expected_default)

    # Flag on: element i lands in Beta quantile bin i, so the batch covers the time axis.
    edges = _bin_edges(bsize, **kwargs)
    assert torch.all((stratified_time >= edges[:-1]) & (stratified_time <= edges[1:] + 1e-6))


def test_molmoact2_flow_tensors_use_stratified_times_and_antithetic_noise():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        num_flow_timesteps=4,
        mask_action_dim_padding=False,
        flow_matching_cutoff=1.0,
        flow_matching_time_offset=0.001,
        flow_matching_time_scale=0.999,
        flow_matching_beta_alpha=1.0,
        flow_matching_beta_beta=1.5,
        flow_matching_stratified_times=True,
        flow_matching_antithetic_noise=True,
    )
    actions = torch.zeros(2, 3, 4)

    # Stratified times: each observation's num_flow_timesteps samples cover the Beta(1, 1.5)
    # quantile bins once (noise pinned to zero so the returned timesteps are the only draws).
    _, timesteps, _, _ = policy._prepare_flow_matching_tensors(
        actions=actions, action_dim_is_pad=None, noise=torch.zeros(2, 4, 3, 4)
    )
    assert timesteps.shape == (2, 4)
    edges = _bin_edges(4, alpha=1.0, beta=1.5, scale=0.999, offset=0.001)
    for k in range(4):
        assert torch.all((timesteps[:, k] >= edges[k]) & (timesteps[:, k] <= edges[k + 1] + 1e-6))

    # Antithetic noise: with t fixed at 0.5 and zero actions, xt = 0.5 * noise, so the
    # returned xt exposes the +/- pairing along the flow-sample axis.
    _, _, xt, _ = policy._prepare_flow_matching_tensors(
        actions=actions,
        action_dim_is_pad=None,
        timesteps=torch.full((2, 4), 0.5),
    )
    noise = xt / 0.5
    for k in range(0, 4, 2):
        assert torch.equal(noise[:, k + 1], -noise[:, k])


def test_molmoact2_flow_tensors_default_to_iid_draws():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        num_flow_timesteps=4,
        mask_action_dim_padding=False,
        flow_matching_cutoff=1.0,
        flow_matching_time_offset=0.001,
        flow_matching_time_scale=0.999,
        flow_matching_beta_alpha=1.0,
        flow_matching_beta_beta=1.5,
        flow_matching_stratified_times=False,
        flow_matching_antithetic_noise=False,
    )
    _, timesteps, _, _ = policy._prepare_flow_matching_tensors(
        actions=torch.zeros(2, 3, 4), action_dim_is_pad=None, noise=torch.zeros(2, 4, 3, 4)
    )
    # i.i.d. draws know nothing about bins: the per-observation samples are not sorted.
    assert not torch.all(timesteps[:, 1:] >= timesteps[:, :-1])
