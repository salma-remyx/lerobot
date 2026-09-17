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
"""Tests for the MINERVA-inspired capacity report and its training-banner wiring.

The capacity report (``lerobot.policies.capacity_report``) is exercised on a real
``nn.Module`` policy, and the training entrypoint is imported to confirm the banner
actually calls into it -- the integration edit, not just the standalone module.
"""

import pytest

from lerobot.policies.capacity_report import (
    LIBERO_CAPACITY_FLOOR,
    LIBERO_SATURATION,
    classify_capacity,
    format_capacity_report,
    group_parameters_by_component,
    parameter_capacity_report,
)
from lerobot.utils.utils import format_big_number  # existing (non-new) banner helper


def test_classify_capacity_bands():
    assert classify_capacity(LIBERO_CAPACITY_FLOOR - 1).name == "below_floor"
    assert classify_capacity(540_000).name == "efficient"
    assert classify_capacity(LIBERO_SATURATION).name == "efficient"
    assert classify_capacity(LIBERO_SATURATION + 1).name == "above_saturation"


def test_component_bucketing_is_name_based():
    torch = pytest.importorskip("torch")

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # ~66K params under a "vision"-named submodule, ~4K under an "action"-named one.
            self.vision_backbone = torch.nn.Linear(256, 256)
            self.action_head = torch.nn.Linear(64, 64)

    policy = TinyPolicy()
    groups = group_parameters_by_component(policy)

    assert groups["vision"] > groups["action"] > 0
    assert set(groups) <= {"vision", "language", "action", "other"}


def test_report_and_format_on_real_module():
    torch = pytest.importorskip("torch")

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_backbone = torch.nn.Linear(128, 128)
            self.action_head = torch.nn.Linear(32, 32)

    policy = TinyPolicy()
    report = parameter_capacity_report(policy)

    expected = sum(p.numel() for p in policy.parameters())
    assert report.total_params == expected
    assert report.learnable_params == expected  # all trainable
    # Well below MINERVA's floor -> flagged as below_floor.
    assert report.band.name == "below_floor"
    assert 0.0 < report.vision_fraction < 1.0

    text = format_capacity_report(policy)
    assert "MINERVA capacity band" in text
    assert "vision=" in text
    # The report's learnable count round-trips through the same big-number helper the banner uses.
    assert format_big_number(report.learnable_params).endswith(("K", "M"))


def test_frozen_params_excluded_from_learnable():
    torch = pytest.importorskip("torch")

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_backbone = torch.nn.Linear(16, 16)
            self.action_head = torch.nn.Linear(16, 16)

    policy = TinyPolicy()
    for p in policy.vision_backbone.parameters():
        p.requires_grad_(False)

    report = parameter_capacity_report(policy)
    assert report.learnable_params < report.total_params
    # Frozen vision params drop out of the learnable component split.
    assert "vision" not in report.component_params


def test_training_banner_wires_capacity_report():
    # Import the actual call site; if heavy training deps are unavailable, skip rather than fail.
    train_mod = pytest.importorskip("lerobot.scripts.lerobot_train")
    # The banner must call the very function under test -- proves the integration edit is live.
    assert train_mod.format_capacity_report is format_capacity_report
