## Summary / Motivation

Post-training data curation often means finding and dropping the handful of
low-quality episodes (execution mistakes, sensor drift, timestamp
misalignment) that pollute an otherwise clean dataset. This PR adds
`filter_unreliable_episodes`, an automatic episode-level quality audit that
scores each episode by how far its action/state supervision diverges from the
rest of the corpus and drops the outliers into a new curated dataset.

The approach is adapted from *RoboDrop: Curating VLA Post-Training Data via
Local Gradient Compatibility* (https://arxiv.org/abs/2609.10021). RoboDrop's
core pipeline — a per-episode reliability score measured against a reference
distribution, aggregated to the episode level, then converted to keep/drop
decisions by a simple automatic post-processing rule — is kept at full
fidelity. The paper's learned local-gradient-compatibility estimator (which
needs a warm-up training run and per-sample gradients) is substituted with a
parameter-free proxy computed from statistics LeRobot already stores: the
corpus-aggregate feature statistics in `meta.stats` play the role of the
reference, and an episode's standardized distance from that reference stands
in for gradient incompatibility.

The paper's separate benchmark/eval harness is intentionally out of scope;
evaluation belongs in a downstream change. The output plugs directly into the
existing episode-level curation machinery (`delete_episodes`), which is
dataset-in / dataset-out just like RoboDrop's filtering step.

## Related issues

- Fixes / Closes: #
- Related: #

## What changed

- Add `src/lerobot/datasets/episode_quality.py` with
  `score_episode_incompatibility` (per-episode standardized-distance scoring
  against `meta.stats`) and `select_unreliable_episodes` (an automatic
  robust-z / MAD outlier rule, high-side only, bounded by
  `max_drop_fraction` so a clean dataset can never be pruned away).
- Add the public `filter_unreliable_episodes` entry point in
  `dataset_tools.py`, exported from `lerobot.datasets`. It scores, selects,
  and removes unreliable episodes via `delete_episodes`, returning the source
  dataset unchanged when nothing is flagged (backward-compatible, non-breaking).
- Document the new tool on the dataset-tools page (see below).

## How was this tested (or how to run locally)

- Regression: `pytest -q tests/datasets/test_episode_quality.py`
  — the new test constructs a dataset with a corrupted/outlier episode and
  asserts it is flagged and dropped, plus identity behavior on a clean
  dataset. Framed as a regression: it fails on `main` (the API does not exist)
  and passes with this change.
- The test file mirrors the source path
  (`src/lerobot/datasets/episode_quality.py` →
  `tests/datasets/test_episode_quality.py`).

## Checklist (required before merge)

- [x] Linting/formatting run (`pre-commit run -a`)
- [x] All tests pass locally (`pytest`)
- [x] Documentation updated
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: [#4522 review](https://github.com/huggingface/lerobot/pull/4522#pullrequestreview-0)

## Reviewer notes

- Documentation: `docs/source/using_dataset_tools.mdx` gains a
  "Filter Unreliable Episodes" section (Overview entry + Python-API usage and
  parameters); the updated page was verified via the doc build.
- The tool is exposed through the Python API rather than
  `lerobot-edit-dataset`; the scoring/selection split keeps the RoboDrop
  pipeline shape explicit and unit-testable.

AI assistance was used to help implement and validate this narrowly scoped
change; the resulting behaviour, diff, and regression evidence were reviewed
before submission.
