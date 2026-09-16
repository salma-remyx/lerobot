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

"""Speed-sweep rollout strategy: temporal-robustness evaluation across speeds.

Runs a policy through fixed-length rollout episodes at a sweep of task-execution
speed factors, collecting an operator success verdict per episode, then reports
how much success degrades relative to the nominal speed.  See
:mod:`lerobot.rollout.temporal_robustness` for the scoring and its paper
attribution (arXiv:2609.01453).

The sweep reuses the same control-loop helpers as the other strategies; the only
per-speed change is the loop cadence, scaled by ``scaled_fps`` so a factor of 2.0
replays the demonstrated trajectory twice as fast.  No policy or environment
change is required.  Success is operator-labelled between episodes (``n`` =
success, ``r`` = failure, ``q`` = stop) exactly like the recording strategies,
because a deployed robot has no scripted success signal.
"""

from __future__ import annotations

import logging
import math
import time

from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import SpeedSweepStrategyConfig
from ..context import RolloutContext
from ..temporal_robustness import (
    TemporalRobustnessReport,
    build_temporal_robustness_report,
    scaled_fps,
)
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)

# Poll interval while waiting for the operator's between-episode success verdict.
_VERDICT_POLL_S = 0.05


class SpeedSweepStrategy(RolloutStrategy):
    """Evaluate a policy's temporal robustness across scaled task-execution speeds.

    For each speed factor in ``config.speed_factors`` (nominal first), runs
    ``config.episodes_per_speed`` rollout episodes of ``config.episode_time_s``
    seconds each, then asks the operator whether the task succeeded.  On
    ``teardown`` the collected outcomes are scored into a
    :class:`~lerobot.rollout.temporal_robustness.TemporalRobustnessReport`,
    which is also left on ``self.report`` for programmatic access.
    """

    config: SpeedSweepStrategyConfig

    def __init__(self, config: SpeedSweepStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events: dict | None = None
        self._outcomes: dict[float, list[bool]] = {}
        self.report: TemporalRobustnessReport | None = None

    def setup(self, ctx: RolloutContext) -> None:
        """Start the inference engine and attach the keyboard listener for verdicts."""
        self._init_engine(ctx)
        self._listener, self._events = init_keyboard_listener()
        logger.info("Speed-sweep strategy ready")

    def _speed_schedule(self) -> list[float]:
        """Ordered, de-duplicated speed factors with nominal evaluated first.

        The nominal factor is always included even if the user omits it — the
        temporal-robustness report needs it as the degradation baseline.
        """
        nominal = self.config.nominal_factor
        others = sorted(
            {f for f in self.config.speed_factors if not math.isclose(f, nominal)}
        )
        return [nominal, *others]

    def run(self, ctx: RolloutContext) -> None:
        """Sweep every speed factor, recording an operator verdict per episode."""
        cfg = ctx.runtime.cfg
        schedule = self._speed_schedule()
        logger.info(
            "Speed-sweep evaluation: factors=%s, %d episode(s) each",
            [f"{f:g}" for f in schedule],
            self.config.episodes_per_speed,
        )

        stop = False
        for factor in schedule:
            if stop or ctx.runtime.shutdown_event.is_set():
                break
            self._outcomes.setdefault(factor, [])
            for episode in range(self.config.episodes_per_speed):
                if ctx.runtime.shutdown_event.is_set():
                    stop = True
                    break
                log_say(
                    f"Speed factor {factor:g}, episode {episode + 1} of {self.config.episodes_per_speed}",
                    cfg.play_sounds,
                )
                self._run_episode(ctx, factor)
                success, stop = self._await_verdict(ctx)
                if success is not None:
                    self._outcomes[factor].append(success)
                if stop:
                    break

        self.report = self.build_report()
        if self.report is not None:
            logger.info("Speed-sweep results:\n%s", self.report.summary())
        else:
            logger.warning("Speed-sweep collected no labelled episodes at the nominal speed")

    def _run_episode(self, ctx: RolloutContext, speed_factor: float) -> None:
        """Drive the policy autonomously for one episode at ``speed_factor`` cadence."""
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        engine = self._engine
        interpolator = self._interpolator

        fps = scaled_fps(cfg.fps, speed_factor)
        timer = CycleTimer(fps, interpolator.multiplier, records_data=False)

        # Discard any leftover hidden state / action queue so each episode starts clean.
        engine.reset()
        interpolator.reset()
        engine.resume()

        start_time = time.perf_counter()
        try:
            while not ctx.runtime.shutdown_event.is_set():
                timer.tick(new_cycle=interpolator.needs_new_action())

                if (time.perf_counter() - start_time) >= self.config.episode_time_s:
                    break

                with timer.section("observe"):
                    obs = robot.get_observation()
                with timer.section("process_obs"):
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, timer):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                with timer.section("telemetry"):
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                timer.wait()
        finally:
            timer.log_run_summary()

    def _await_verdict(self, ctx: RolloutContext) -> tuple[bool | None, bool]:
        """Block until the operator labels the last episode.

        Returns ``(success, stop)``: ``success`` is ``True``/``False`` for the
        labelled outcome or ``None`` when unlabelled (no listener, or stop
        requested); ``stop`` is ``True`` when the sweep should end.
        """
        events = self._events
        if events is None:
            # Non-interactive run (piped / headless without a TTY): we cannot ask
            # for a verdict, so leave the episode unlabelled rather than guessing.
            logger.warning("No keyboard listener available; skipping success verdict for this episode")
            return None, False

        # Clear any control flags left over from the episode before prompting.
        events["exit_early"] = False
        events["rerecord_episode"] = False
        log_say("Episode done. n = success, r = failure, q = stop", ctx.runtime.cfg.play_sounds)

        while not ctx.runtime.shutdown_event.is_set():
            if events["stop_recording"]:
                return None, True
            # ``r``/left sets both rerecord and exit_early, so test it first.
            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                return False, False
            if events["exit_early"]:
                events["exit_early"] = False
                return True, False
            precise_sleep(_VERDICT_POLL_S)
        return None, True

    def build_report(self) -> TemporalRobustnessReport | None:
        """Score the collected outcomes, or ``None`` if the nominal speed is unlabelled."""
        labelled = {factor: labels for factor, labels in self._outcomes.items() if labels}
        try:
            return build_temporal_robustness_report(labelled, nominal_factor=self.config.nominal_factor)
        except ValueError as exc:
            logger.warning("Could not build temporal-robustness report: %s", exc)
            return None

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop the listener, log the final report, and disconnect hardware."""
        if self._listener is not None:
            self._listener.stop()

        if self.report is None:
            self.report = self.build_report()
        if self.report is not None:
            logger.info("Temporal-robustness summary:\n%s", self.report.summary())

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Speed-sweep strategy teardown complete")
