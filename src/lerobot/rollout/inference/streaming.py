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

"""Streaming inference engine with confidence-gated action reuse.

Adapted from *FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA
Inference* (https://arxiv.org/abs/2608.27384). FlashVLA's core insight is that a
flow-matching chunk decode is expensive, so a deployment should emit one
executable chunk per inference step and *reuse* already-decoded actions while the
predicted trajectory stays stable — raising the achievable control frequency
without re-running the decoder every tick.

This engine keeps that core mechanism at full fidelity: it maintains a streaming
buffer of postprocessed actions and, after each decode, adaptively picks how many
of them to execute before the next decode (the *reuse horizon*). The paper's
learned chunk-wise-causal-attention confidence is replaced with a parameter-free
proxy — the agreement between the overlapping predictions of successive chunks
(:class:`StreamingReuseGate`). High agreement means the trajectory is stable, so
more buffered actions are reused (fewer decodes → higher control frequency); low
agreement forces an earlier re-decode so the policy stays reactive. The paper's
trained multi-noise-level buffer and its RoboCasa benchmark suite are out of
scope — they need training infrastructure the rollout stack does not host, and
belong in a downstream change.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


class StreamingReuseGate:
    """Parameter-free confidence proxy for how many decoded actions to reuse.

    Confidence is the agreement between the two most recent chunks over the
    timesteps they both predict.  A chunk decoded ``consumed`` ticks after its
    predecessor overlaps the predecessor's tail ``prev_chunk[consumed:]`` with
    its own head, both describing the same absolute future timesteps.  Small
    disagreement there means the trajectory is stable and a longer reuse horizon
    is safe; large disagreement shrinks the horizon so the policy re-decodes
    sooner.  The score is scale-invariant so it behaves the same across policies
    with different action magnitudes.
    """

    def __init__(self, min_horizon: int = 1, max_horizon: int = 16, overlap: int = 8) -> None:
        if min_horizon < 1:
            raise ValueError(f"min_horizon must be >= 1, got {min_horizon}")
        if max_horizon < min_horizon:
            raise ValueError(f"max_horizon ({max_horizon}) must be >= min_horizon ({min_horizon})")
        if overlap < 1:
            raise ValueError(f"overlap must be >= 1, got {overlap}")
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.overlap = overlap

    def confidence(self, prev_chunk: torch.Tensor | None, new_chunk: torch.Tensor, consumed: int) -> float:
        """Agreement in ``[0, 1]`` between the chunks' overlapping predictions.

        Returns ``1.0`` when there is no evidence of drift (first decode, or the
        previous chunk was fully drained so nothing overlaps): with no signal the
        gate optimistically favours reuse, and the next decode re-checks.
        """
        if prev_chunk is None:
            return 1.0
        aligned = prev_chunk[consumed:]
        n = min(aligned.shape[0], new_chunk.shape[0], self.overlap)
        if n <= 0:
            return 1.0
        a = aligned[:n].float()
        b = new_chunk[:n].float()
        mad = (a - b).abs().mean()
        scale = b.abs().mean().clamp(min=1e-6)
        rel = (mad / scale).item()
        return 1.0 / (1.0 + rel)

    def horizon_for(self, confidence: float, chunk_len: int) -> int:
        """Reuse horizon for a decoded chunk: how many of its actions to execute."""
        hi = max(1, min(self.max_horizon, chunk_len))
        lo = max(1, min(self.min_horizon, hi))
        confidence = min(1.0, max(0.0, confidence))
        return int(round(lo + confidence * (hi - lo)))


class StreamingInferenceEngine(InferenceEngine):
    """Inline chunk decoding with confidence-gated action reuse.

    ``get_action`` returns the next buffered action when one is available (a
    *reuse* tick — no policy call), otherwise it decodes a fresh chunk via
    ``predict_action_chunk``, measures continuity against the previous chunk, and
    refills the buffer with an adaptive number of actions.  The buffer is emptied
    on a task change so a new instruction takes effect on the next tick.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        min_horizon: int = 1,
        max_horizon: int = 16,
        overlap: int = 8,
    ) -> None:
        super().__init__(task=task)
        base_impl = getattr(PreTrainedPolicy, "predict_action_chunk", None)
        if base_impl is not None and getattr(type(policy), "predict_action_chunk", None) is base_impl:
            raise ValueError(
                "StreamingInferenceEngine requires a chunking policy that overrides "
                "predict_action_chunk. Use --inference.type=sync for non-chunking policies."
            )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._gate = StreamingReuseGate(min_horizon=min_horizon, max_horizon=max_horizon, overlap=overlap)

        # Episode-scoped streaming state.
        self._buffer: list[torch.Tensor] = []
        self._prev_chunk: torch.Tensor | None = None
        self._last_horizon = 0
        self._chunk_task = task

        # Cumulative efficiency counters, surfaced via ``stats``.
        self._inference_count = 0
        self._reused_ticks = 0

        logger.info(
            "StreamingInferenceEngine initialized (device=%s, horizon=[%d,%d], overlap=%d)",
            self._device,
            min_horizon,
            max_horizon,
            overlap,
        )

    @property
    def stats(self) -> dict[str, float]:
        """Reuse efficiency so far: decodes, reused ticks, and actions per decode."""
        total = self._inference_count + self._reused_ticks
        return {
            "inferences": float(self._inference_count),
            "reused_ticks": float(self._reused_ticks),
            "reuse_ratio": (self._reused_ticks / total) if total else 0.0,
            "actions_per_inference": (total / self._inference_count) if self._inference_count else 0.0,
        }

    def start(self) -> None:
        """No background resources to start."""
        logger.info("StreamingInferenceEngine started (inline streaming decode)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("StreamingInferenceEngine stopped (stats=%s)", self.stats)

    def reset(self) -> None:
        """Reset the policy, processors, and streaming buffer."""
        logger.info("Resetting streaming inference state (policy + processors + buffer)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._buffer.clear()
        self._prev_chunk = None
        self._last_horizon = 0
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next buffered action, decoding a fresh chunk when empty."""
        if obs_frame is None:
            return None
        task, task_changed = self._take_task()
        if task_changed and self._buffer:
            # Drop actions decoded under the previous instruction so the new one
            # lands on this tick; continuity state is dropped with them.
            logger.info("Task changed to '%s' — dropping buffered actions", task)
            self._buffer.clear()
            self._prev_chunk = None

        if self._buffer:
            self._reused_ticks += 1
            self._set_dispatched_task(self._chunk_task)
            return self._buffer.pop(0)

        chunk = self._decode_chunk(obs_frame, task)
        confidence = self._gate.confidence(self._prev_chunk, chunk, self._last_horizon)
        horizon = self._gate.horizon_for(confidence, chunk_len=chunk.shape[0])

        self._buffer = [chunk[i] for i in range(horizon)]
        self._prev_chunk = chunk
        self._last_horizon = horizon
        self._chunk_task = task
        self._inference_count += 1
        logger.debug("Streaming decode: confidence=%.3f horizon=%d/%d", confidence, horizon, chunk.shape[0])

        self._set_dispatched_task(task)
        return self._buffer.pop(0)

    def _decode_chunk(self, obs_frame: dict, task: str) -> torch.Tensor:
        """Run the policy once and return the postprocessed chunk as ``[T, A]`` rows.

        Rows are reordered to the dataset action keys so every returned action is
        directly comparable and executable, matching the sync backend's contract.
        """
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            actions = self._policy.predict_action_chunk(observation)
            actions = self._postprocessor(actions)
        chunk = actions.squeeze(0).cpu()
        if chunk.ndim != 2:
            raise ValueError(f"Expected a 2D [T, A] action chunk, got shape={tuple(chunk.shape)}")
        rows = []
        for step in range(chunk.shape[0]):
            action_dict = make_robot_action(chunk[step], self._dataset_features)
            rows.append(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))
        return torch.stack(rows)

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """Decoding runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task = self.task
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            observation = self._mark_query(observation, query)
            observation = self._preprocessor(observation)
            return self._policy.generate_text(observation)
