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
"""Capacity-aware reporting for visuomotor policies.

Adapted from MINERVA ("How Small Can a Manipulation Policy Be and Still Solve
LIBERO?", arXiv:2609.03715). MINERVA measures a *task-specific capacity floor*:
a 0.54M-parameter policy reaches 95.1% average success on the four standard
LIBERO suites, performance saturates near 1M parameters, and collapses toward
chance below ~0.25M. The same sweeps find that, among all architectural knobs,
only action-chunk length and vision capacity consistently move the needle.

This module turns those empirical findings into a lightweight, parameter-free
report that any ``nn.Module`` policy can be measured against. It is used by the
training banner to tell contributors, at a glance, where a policy sits relative
to the MINERVA capacity bands and how its parameters split across the vision and
action components the paper flags as the ones that matter.

Nothing here trains, prunes, or distills a policy -- it only reads
``named_parameters()``. Everything downstream (choosing a smaller backbone,
shortening the action chunk, distilling) is left to the contributor; the report
just makes the capacity picture explicit before those decisions are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

# MINERVA capacity landmarks for LIBERO-class manipulation, in learnable parameters.
# Below the floor, success collapses toward chance; the reference point is the paper's
# headline 0.54M policy; beyond saturation extra capacity buys little on this task family.
LIBERO_CAPACITY_FLOOR = 250_000
LIBERO_EFFICIENT_REFERENCE = 540_000
LIBERO_SATURATION = 1_000_000

# Substrings used to bucket parameters into the components MINERVA identifies as the
# dominant capacity levers. Matching is case-insensitive over the fully-qualified
# parameter name; the first bucket that matches wins, so order matters.
_VISION_KEYS = (
    "vision",
    "visual",
    "image",
    "img",
    "rgb",
    "backbone",
    "resnet",
    "patch_embed",
    "vit",
    "pixel",
)
_ACTION_KEYS = (
    "action",
    "act_head",
    "decoder",
    "head",
    "flow",
    "denoise",
    "diffusion",
    "unet",
    "chunk",
)
_LANGUAGE_KEYS = (
    "language",
    "text",
    "lang",
    "token",
    "llm",
    "word_embed",
    "gemma",
    "prompt",
)


@dataclass
class CapacityBand:
    """Where a parameter count falls relative to the MINERVA LIBERO landmarks."""

    name: str
    message: str


@dataclass
class CapacityReport:
    """Structured capacity summary for a policy."""

    total_params: int
    learnable_params: int
    band: CapacityBand
    component_params: dict[str, int] = field(default_factory=dict)

    @property
    def vision_fraction(self) -> float:
        """Share of learnable parameters in the vision component (0.0 if none counted)."""
        if self.learnable_params == 0:
            return 0.0
        return self.component_params.get("vision", 0) / self.learnable_params


def classify_capacity(learnable_params: int) -> CapacityBand:
    """Classify a learnable-parameter count against the MINERVA LIBERO capacity bands.

    Args:
        learnable_params: Number of trainable parameters in the policy.

    Returns:
        A :class:`CapacityBand` naming the regime and a one-line, actionable message.
    """
    if learnable_params < LIBERO_CAPACITY_FLOOR:
        return CapacityBand(
            "below_floor",
            f"below MINERVA's ~{LIBERO_CAPACITY_FLOOR / 1e6:.2f}M LIBERO capacity floor "
            "-- success can collapse toward chance on LIBERO-class tasks",
        )
    if learnable_params <= LIBERO_SATURATION:
        return CapacityBand(
            "efficient",
            f"within MINERVA's efficient regime (~{LIBERO_CAPACITY_FLOOR / 1e6:.2f}M-"
            f"{LIBERO_SATURATION / 1e6:.2f}M); the 0.54M reference hit 95.1% on LIBERO",
        )
    return CapacityBand(
        "above_saturation",
        f"above MINERVA's ~{LIBERO_SATURATION / 1e6:.2f}M saturation point -- extra capacity "
        "buys little on LIBERO-class tasks; consider distillation for deployment efficiency",
    )


def _component_for(param_name: str) -> str:
    """Bucket a fully-qualified parameter name into a policy component."""
    lowered = param_name.lower()
    for key in _VISION_KEYS:
        if key in lowered:
            return "vision"
    for key in _LANGUAGE_KEYS:
        if key in lowered:
            return "language"
    for key in _ACTION_KEYS:
        if key in lowered:
            return "action"
    return "other"


def group_parameters_by_component(module: nn.Module) -> dict[str, int]:
    """Sum learnable parameters per component (vision / language / action / other).

    Grouping is name-based and heuristic; it is meant to expose the vision-vs-action
    split MINERVA highlights, not to be an exact architectural accounting.

    Args:
        module: Any object exposing ``named_parameters()`` (e.g. an ``nn.Module``).

    Returns:
        Mapping from component name to its learnable-parameter count. Components with
        zero parameters are omitted.
    """
    totals: dict[str, int] = {}
    for name, param in module.named_parameters():
        if not getattr(param, "requires_grad", True):
            continue
        component = _component_for(name)
        totals[component] = totals.get(component, 0) + int(param.numel())
    return totals


def parameter_capacity_report(module: nn.Module) -> CapacityReport:
    """Build a :class:`CapacityReport` for a policy.

    Args:
        module: Any object exposing ``named_parameters()`` (e.g. an ``nn.Module``).

    Returns:
        A :class:`CapacityReport` with total/learnable counts, per-component split, and
        the MINERVA capacity band the policy falls into.
    """
    total = 0
    learnable = 0
    for param in module.parameters():
        n = int(param.numel())
        total += n
        if getattr(param, "requires_grad", True):
            learnable += n
    return CapacityReport(
        total_params=total,
        learnable_params=learnable,
        band=classify_capacity(learnable),
        component_params=group_parameters_by_component(module),
    )


def format_capacity_report(module: nn.Module) -> str:
    """Render a human-readable capacity report for logging.

    Args:
        module: The policy to measure.

    Returns:
        A multi-line string: the capacity band followed by the vision/action split.
    """
    report = parameter_capacity_report(module)
    lines = [
        f"MINERVA capacity band: {report.band.name} "
        f"({report.learnable_params / 1e6:.2f}M learnable params -- {report.band.message})",
    ]
    if report.component_params:
        ordered = ("vision", "language", "action", "other")
        parts = []
        for component in ordered:
            count = report.component_params.get(component)
            if count:
                share = 100.0 * count / max(report.learnable_params, 1)
                parts.append(f"{component}={count / 1e6:.2f}M ({share:.0f}%)")
        lines.append("Capacity split: " + ", ".join(parts))
    return "\n".join(lines)
