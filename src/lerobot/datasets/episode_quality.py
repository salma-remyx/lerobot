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

"""Episode-level supervision-quality scoring for post-training data curation.

Adapted from *RoboDrop: Curating VLA Post-Training Data via Local Gradient
Compatibility* (https://arxiv.org/abs/2609.10021). RoboDrop audits supervision
by measuring, during a one-epoch warm-up run, how compatible each candidate
sample's gradient is with the gradients of task-semantic and visually matched
validation samples; sample scores are aggregated to the episode level and a
simple automatic post-processing rule turns them into keep/drop decisions.

This module keeps RoboDrop's core pipeline at full fidelity — a per-episode
reliability score measured *against a reference distribution*, aggregated to the
episode level, then converted to filtering decisions by an automatic
outlier-detection rule — while substituting the paper's learned local-gradient
compatibility estimator (which needs a warm-up training run and per-sample
gradients) with a parameter-free proxy computed from statistics LeRobot already
stores. The corpus-aggregate feature statistics in ``meta.stats`` play the role
of the "task-semantic / visually matched validation" reference, and an episode's
standardized distance from that reference stands in for gradient incompatibility.
Corruptions RoboDrop targets — execution mistakes, sensor drift, timestamp
misalignment — surface as episodes whose action/state supervision diverges from
the corpus. The paper's separate benchmark/eval harness is intentionally out of
scope; evaluation belongs in a downstream change.

The output plugs directly into the existing episode-level curation machinery
(:func:`lerobot.datasets.dataset_tools.delete_episodes`), which is dataset-in /
dataset-out just like RoboDrop's filtering step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.constants import ACTION, OBS_STATE

if TYPE_CHECKING:
    from .lerobot_dataset import LeRobotDataset

# Consistent-with-a-normal-distribution scale factor: 1.4826 * MAD estimates the
# standard deviation for Gaussian data, so ``robust_z`` is comparable to a z-score.
_MAD_TO_STD = 1.4826

# Floor for standard deviations so near-constant dimensions don't blow the
# standardized distance up to infinity.
_STD_FLOOR = 1e-6

# Clip standardized per-dimension distances so a single shifted constant channel
# can't dominate the aggregated score.
_Z_CLIP = 1e4


def _default_feature_keys(meta) -> list[str]:
    """Numeric proprioceptive/action vector features present in the stats."""
    candidates = [ACTION, OBS_STATE]
    return [k for k in candidates if meta.stats is not None and k in meta.stats]


def _load_episode_stats(dataset: LeRobotDataset, episode_idx: int) -> dict[str, dict[str, np.ndarray]]:
    """Read one episode's per-feature statistics from its metadata parquet.

    ``load_episodes`` deliberately drops ``stats/*`` columns from
    ``meta.episodes``, so the per-episode statistics are read back from the
    episodes parquet. This reuses the same reader as the rest of ``dataset_tools``
    to keep the on-disk layout the single source of truth.
    """
    # Local import avoids a module-load cycle: ``dataset_tools`` lazily imports
    # this module from ``filter_unreliable_episodes``.
    from .dataset_tools import _load_episode_with_stats

    ep_full = _load_episode_with_stats(dataset, episode_idx)
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in ep_full.items():
        if not key.startswith("stats/"):
            continue
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, feature_name, stat_name = parts
        stats.setdefault(feature_name, {})[stat_name] = np.asarray(value)
    return stats


def score_episode_incompatibility(
    dataset: LeRobotDataset,
    feature_keys: list[str] | None = None,
) -> dict[int, float]:
    """Score every episode by how far its supervision diverges from the corpus.

    For each requested feature the episode mean is standardized by the corpus
    standard deviation (``|ep_mean - corpus_mean| / corpus_std``) and averaged
    over dimensions; the per-feature distances are then averaged into a single
    scalar. Higher scores mark episodes whose action/state supervision is least
    compatible with the bulk of the dataset — the parameter-free stand-in for
    RoboDrop's local gradient incompatibility.

    Args:
        dataset: Source dataset to audit. Must have finalized statistics.
        feature_keys: Numeric vector features to compare. Defaults to the action
            and observation-state features present in ``meta.stats``.

    Returns:
        Mapping from episode index to its non-negative incompatibility score.
    """
    meta = dataset.meta
    if meta.stats is None:
        raise ValueError(
            "Dataset has no aggregate statistics; call `finalize()` (or recompute stats) "
            "before scoring episode quality."
        )

    if feature_keys is None:
        feature_keys = _default_feature_keys(meta)
    if not feature_keys:
        raise ValueError(
            "No numeric feature keys available to score. Pass `feature_keys` explicitly "
            f"(dataset stats cover: {sorted(meta.stats)})."
        )

    missing = [k for k in feature_keys if k not in meta.stats]
    if missing:
        raise ValueError(f"feature_keys not found in dataset stats: {missing}")

    corpus: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in feature_keys:
        mean = np.asarray(meta.stats[key]["mean"], dtype=np.float64).ravel()
        std = np.asarray(meta.stats[key]["std"], dtype=np.float64).ravel()
        corpus[key] = (mean, np.maximum(std, _STD_FLOOR))

    scores: dict[int, float] = {}
    for ep_idx in range(meta.total_episodes):
        ep_stats = _load_episode_stats(dataset, ep_idx)
        per_feature: list[float] = []
        for key in feature_keys:
            if key not in ep_stats or "mean" not in ep_stats[key]:
                continue
            corpus_mean, corpus_std = corpus[key]
            ep_mean = np.asarray(ep_stats[key]["mean"], dtype=np.float64).ravel()
            if ep_mean.shape != corpus_mean.shape:
                continue
            z = np.abs(ep_mean - corpus_mean) / corpus_std
            per_feature.append(float(np.mean(np.clip(z, 0.0, _Z_CLIP))))
        scores[ep_idx] = float(np.mean(per_feature)) if per_feature else 0.0

    return scores


def select_unreliable_episodes(
    scores: dict[int, float],
    n_sigma: float = 3.0,
    max_drop_fraction: float = 0.5,
) -> list[int]:
    """Convert episode scores into drop decisions with an automatic MAD rule.

    Mirrors RoboDrop's "simple automatic post-processing rule": rather than a
    hand-tuned absolute threshold, episodes are flagged when their robust z-score
    (median-centered, MAD-scaled) exceeds ``n_sigma``. Only high-side outliers are
    dropped, and no more than ``max_drop_fraction`` of the dataset is ever removed
    (the most anomalous episodes win ties), so a run cannot prune away a healthy
    dataset.

    Args:
        scores: Per-episode incompatibility scores from
            :func:`score_episode_incompatibility`.
        n_sigma: Robust-z cutoff above the median for flagging an episode.
        max_drop_fraction: Upper bound on the fraction of episodes to drop.

    Returns:
        Sorted list of episode indices judged unreliable.
    """
    if not scores:
        return []

    idxs = sorted(scores)
    vals = np.array([scores[i] for i in idxs], dtype=np.float64)

    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    if mad > 0:
        scale = _MAD_TO_STD * mad
    else:
        # Degenerate MAD (>=half the scores identical): fall back to plain std.
        scale = float(vals.std())
    if scale <= 0:
        return []

    robust_z = (vals - median) / scale
    flagged = [(idxs[i], robust_z[i]) for i in range(len(idxs)) if robust_z[i] > n_sigma]

    max_drop = int(len(idxs) * max_drop_fraction)
    if max_drop <= 0:
        return []
    if len(flagged) > max_drop:
        flagged = sorted(flagged, key=lambda item: item[1], reverse=True)[:max_drop]

    dropped = sorted(idx for idx, _ in flagged)
    logging.info(
        f"Episode-quality audit flagged {len(dropped)}/{len(idxs)} episodes as unreliable "
        f"(n_sigma={n_sigma}, max_drop_fraction={max_drop_fraction})."
    )
    return dropped
