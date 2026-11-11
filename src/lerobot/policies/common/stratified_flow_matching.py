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

"""Variance-reduced flow-matching sampling: stratified times, antithetic noise, and the
noise-vs-compute choice of how many flow samples to draw.

Adapted from "One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR
Challenge" (arXiv:2611.14432), Section 5. The training loss only needs the *marginal*
distribution of each flow time and noise draw to be right, not their independence, so
classic Monte-Carlo variance reduction applies at zero cost:

- Stratified times: sample ``k`` of a group lands in quantile bin ``k`` of the Beta
  time distribution, ``t_k = F^-1((k + U_k) / K)``. A uniformly chosen sample is exactly
  Beta-distributed, so the expected loss is unchanged; what disappears is the chance of
  a whole group clustering at one end of the time axis.
- Antithetic noise: adjacent samples of the same observation share one noise draw with
  opposite signs, so the part of the per-sample gradient that is odd in the noise
  cancels within each pair. ``-eps`` has the same marginal as ``eps``.

Two axes are exposed. ``sample_time_beta_stratified`` / ``sample_noise_antithetic``
stratify the ``K`` flow samples of one observation, which is where both tricks apply and
where the paper measures them (roughly: stratification halves the flow term of the
gradient variance, antithetic pairs remove another 17%). ``sample_time_beta_batch_stratified``
applies the time stratification across the *batch* axis of policies that draw a single
(time, noise) pair per observation: the batch-mean loss stays unbiased because the bins
partition the time axis, so it is a valid free variance reduction there too. Antithetic
noise has no such batch-axis guarantee — pairing needs the two samples to share their
observation and action chunk — so it is only offered on the per-observation axis.

``select_num_flow_samples`` picks ``K`` from the paper's noise-vs-compute model
``J(K) = (c_p + c_k K)(V_oa + V_f / K)`` (gradient variance times step time, with the
batch size cancelling): ``K* = sqrt(c_p / c_k * V_f / V_oa)``. Its inputs are two timing
runs and one gradient-noise probe, so ``K`` can be chosen without an ablation sweep.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def beta_quantile(u: Tensor, *, alpha: float, beta: float) -> Tensor:
    """Quantile function (inverse CDF) of ``Beta(alpha, beta)`` evaluated at ``u``.

    Closed forms for the two one-parameter shapes every flow-matching policy in this
    repo uses; ``Beta(1.5, 1)`` gives ``u ** (2 / 3)``, the openpi convention.
    """
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"Beta parameters must be positive, got alpha={alpha}, beta={beta}")
    if float(u.min()) < 0.0 or float(u.max()) > 1.0:
        raise ValueError("beta_quantile expects u in [0, 1]")
    if beta == 1.0:
        return u.pow(1.0 / alpha)
    if alpha == 1.0:
        return 1.0 - (1.0 - u).pow(1.0 / beta)
    raise ValueError(
        f"Stratified Beta times are only implemented for beta=1 or alpha=1 (closed-form inverse "
        f"CDF), got alpha={alpha}, beta={beta}."
    )


def sample_time_beta_stratified(
    bsize: int, num_samples: int, device, *, alpha: float, beta: float, scale: float, offset: float
) -> Tensor:
    """Stratified Beta-distributed flow times, one draw of ``num_samples`` times per observation.

    Returns a float32 tensor of shape ``(bsize, num_samples)`` where sample ``k`` of each
    observation is drawn from quantile bin ``k`` of ``Beta(alpha, beta)``, then mapped
    through the openpi affine ``time = sample * scale + offset``. Each row covers the
    time axis evenly; each individual column keeps the Beta marginal.
    """
    bins = _stratified_bins((bsize, num_samples), device)
    time = beta_quantile(bins, alpha=alpha, beta=beta) * scale + offset
    return time.to(dtype=torch.float32, device=device)


def sample_time_beta_batch_stratified(
    bsize: int, device, *, alpha: float, beta: float, scale: float, offset: float
) -> Tensor:
    """One stratified flow time per batch element, stratified across the batch.

    Same construction as :func:`sample_time_beta_stratified` with the batch axis playing
    the role of the sample axis: element ``i`` draws from bin ``i``, so a batch of ``B``
    elements covers the time axis in ``B`` bins. The batch-mean training loss stays
    unbiased because the bins partition the time axis; the variance of that mean drops.
    """
    bins = _stratified_bins((bsize,), device)
    time = beta_quantile(bins, alpha=alpha, beta=beta) * scale + offset
    return time.to(dtype=torch.float32, device=device)


def sample_noise_antithetic(shape, device, *, sample_axis: int = 1) -> Tensor:
    """Standard-normal float32 flow noise arranged in +/- pairs along ``sample_axis``.

    Sample ``2j`` and ``2j+1`` (adjacent time bins) share one draw with opposite signs,
    which leaves every marginal standard normal while cancelling the noise-odd component
    of their average. An odd sample count leaves the last draw unpaired. Pairs must share
    their observation and action chunk, i.e. ``sample_axis`` must index flow samples of
    the same observation, not the batch.
    """
    shape = tuple(shape)
    if shape and sample_axis < 0:
        sample_axis += len(shape)
    if not 0 <= sample_axis < len(shape):
        raise ValueError(f"sample_axis {sample_axis} is out of range for shape {shape}")
    num_samples = shape[sample_axis]
    if num_samples < 1:
        raise ValueError(f"expected at least one sample along sample_axis, got shape {shape}")

    half = (num_samples + 1) // 2
    base_shape = shape[:sample_axis] + (half,) + shape[sample_axis + 1 :]
    base = torch.normal(mean=0.0, std=1.0, size=base_shape, dtype=torch.float32, device=device)
    pairs = torch.stack((base, -base), dim=sample_axis + 1).flatten(sample_axis, sample_axis + 1)
    return pairs.narrow(sample_axis, 0, num_samples)


def noise_compute_objective(
    num_samples: int, *, prefix_cost: float, sample_cost: float, flow_variance: float, other_variance: float
) -> float:
    """Noise-vs-compute objective ``J(K) = (c_p + c_k K)(V_oa + V_f / K)``.

    Proportional to gradient variance times step time in the noise-dominated regime, so
    lower is better. ``prefix_cost`` is the per-observation cost of everything shared by
    the flow samples (VLM prefix, data loading, optimizer), ``sample_cost`` the cost of
    one extra pass of the action expert, ``flow_variance`` the single-draw flow term
    ``V_f`` and ``other_variance`` the observation+augmentation term ``V_oa``.
    """
    _validate_cost_model(prefix_cost, sample_cost, flow_variance, other_variance)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    return (prefix_cost + sample_cost * num_samples) * (other_variance + flow_variance / num_samples)


def select_num_flow_samples(
    *, prefix_cost: float, sample_cost: float, flow_variance: float, other_variance: float
) -> int:
    """Number of flow samples ``K* = sqrt(c_p / c_k * V_f / V_oa)`` that minimizes ``J(K)``.

    The batch size cancels out of ``J``, so this is independent of it: pick ``K`` here,
    then set the batch size from memory. Floors at 1, which reduces to single-sample
    flow matching.
    """
    _validate_cost_model(prefix_cost, sample_cost, flow_variance, other_variance)
    variance_ratio = (prefix_cost / sample_cost) * (flow_variance / other_variance)
    return max(1, round(math.sqrt(variance_ratio)))


def relative_noise_compute_cost(
    num_samples: int, *, prefix_cost: float, sample_cost: float, flow_variance: float, other_variance: float
) -> float:
    """``J(num_samples) / J(K*)``: how close a candidate ``K`` is to the optimum.

    ``J`` is flat around ``K*`` — e.g. the paper's constants give ``K* ~ 10`` with
    anything in ``[8, 14]`` within 1.5% — so this tells you when a rounder ``K`` (say the
    power of two a kernel is tuned for) costs nothing.
    """
    optimum = select_num_flow_samples(
        prefix_cost=prefix_cost,
        sample_cost=sample_cost,
        flow_variance=flow_variance,
        other_variance=other_variance,
    )
    best = noise_compute_objective(
        optimum,
        prefix_cost=prefix_cost,
        sample_cost=sample_cost,
        flow_variance=flow_variance,
        other_variance=other_variance,
    )
    cost = noise_compute_objective(
        num_samples,
        prefix_cost=prefix_cost,
        sample_cost=sample_cost,
        flow_variance=flow_variance,
        other_variance=other_variance,
    )
    return cost / best


def flow_variance_from_probe(probe_num_samples: int, probe_variance: float) -> float:
    """Single-draw flow variance ``V_f`` from a gradient-noise probe at fixed ``K``.

    The flow term of a ``K``-sample average is ``V_f / K``, so a probe that resamples
    only the flow draws measures ``V_f / K_probe``: multiply it back out. Combine with the
    observation/augmentation term of the same probe to get the ratio
    :func:`select_num_flow_samples` needs.
    """
    if probe_num_samples < 1:
        raise ValueError(f"probe_num_samples must be >= 1, got {probe_num_samples}")
    if probe_variance < 0.0:
        raise ValueError(f"probe_variance must be non-negative, got {probe_variance}")
    return probe_num_samples * probe_variance


def _stratified_bins(size, device) -> Tensor:
    """Uniform draws ``U`` mapped onto strata ``(arange(K) + U) / K``, one stratum per sample."""
    u = torch.rand(size, device=device, dtype=torch.float32)
    num_samples = size[-1]
    strata = torch.arange(num_samples, device=device, dtype=torch.float32)
    return (u + strata) / num_samples


def _validate_cost_model(prefix_cost: float, sample_cost: float, flow_variance: float, other_variance: float):
    if prefix_cost <= 0.0 or sample_cost <= 0.0:
        raise ValueError(f"costs must be positive, got prefix_cost={prefix_cost}, sample_cost={sample_cost}")
    if flow_variance < 0.0:
        raise ValueError(f"flow_variance must be non-negative, got {flow_variance}")
    if other_variance <= 0.0:
        raise ValueError(f"other_variance must be positive, got {other_variance}")
