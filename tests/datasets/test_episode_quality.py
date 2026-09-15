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
"""Tests for RoboDrop-inspired episode-quality curation.

Exercises both the scoring module (:mod:`lerobot.datasets.episode_quality`) and
the wiring hook added to the existing curation entry point
(:func:`lerobot.datasets.dataset_tools.filter_unreliable_episodes`).
"""

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")


from lerobot.datasets.dataset_tools import filter_unreliable_episodes
from lerobot.datasets.episode_quality import (
    score_episode_incompatibility,
    select_unreliable_episodes,
)

# Episode index that receives a large, uniform action/state offset — the kind of
# execution mistake / sensor drift RoboDrop is meant to catch.
CORRUPTED_EPISODE = 3
NUM_EPISODES = 6


@pytest.fixture
def dataset_with_one_corrupt_episode(tmp_path, empty_lerobot_dataset_factory):
    """Six tightly-clustered clean episodes plus one shifted (corrupted) episode."""
    rng = np.random.default_rng(0)
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (32, 32, 3), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "corrupt_dataset",
        features=features,
    )

    for ep_idx in range(NUM_EPISODES):
        offset = 8.0 if ep_idx == CORRUPTED_EPISODE else 0.0
        for _ in range(10):
            frame = {
                "action": (rng.standard_normal(6) * 0.1 + offset).astype(np.float32),
                "observation.state": (rng.standard_normal(4) * 0.1 + offset).astype(np.float32),
                "observation.images.top": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                "task": "pick",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def test_corrupted_episode_scores_highest(dataset_with_one_corrupt_episode):
    scores = score_episode_incompatibility(dataset_with_one_corrupt_episode)

    assert set(scores) == set(range(NUM_EPISODES))
    worst = max(scores, key=scores.get)
    assert worst == CORRUPTED_EPISODE
    # The corrupted episode should stand clearly apart from the clean cluster.
    clean = [v for k, v in scores.items() if k != CORRUPTED_EPISODE]
    assert scores[CORRUPTED_EPISODE] > 3 * max(clean)


def test_select_unreliable_flags_the_outlier(dataset_with_one_corrupt_episode):
    scores = score_episode_incompatibility(dataset_with_one_corrupt_episode)
    to_drop = select_unreliable_episodes(scores)
    assert to_drop == [CORRUPTED_EPISODE]


def test_filter_unreliable_episodes_removes_corrupted(dataset_with_one_corrupt_episode, tmp_path):
    """The dataset_tools hook curates the corrupted episode out via delete_episodes."""
    output_dir = tmp_path / "curated"

    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        curated = filter_unreliable_episodes(
            dataset_with_one_corrupt_episode,
            output_dir=output_dir,
        )

    assert curated.meta.total_episodes == NUM_EPISODES - 1
    assert curated.meta.total_frames == (NUM_EPISODES - 1) * 10
    # Curation re-indexes to a contiguous range with the corrupted episode gone.
    episode_indices = {int(idx.item()) for idx in curated.hf_dataset["episode_index"]}
    assert episode_indices == set(range(NUM_EPISODES - 1))


def test_filter_returns_source_when_nothing_flagged(dataset_with_one_corrupt_episode):
    """With an impossibly strict cutoff nothing is dropped and the source is returned."""
    with patch("lerobot.datasets.episode_quality.select_unreliable_episodes", return_value=[]):
        result = filter_unreliable_episodes(dataset_with_one_corrupt_episode)
    assert result is dataset_with_one_corrupt_episode


def test_select_unreliable_handles_empty():
    assert select_unreliable_episodes({}) == []
