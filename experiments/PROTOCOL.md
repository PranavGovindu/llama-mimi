# TinyAya Mimi TTS Experiment Protocol

## Purpose
This protocol defines mandatory structure for reproducible experiments.
No "random tweak runs" should be executed outside this flow.

## Phases
1. `overfit_q1`
- Goal: prove end-to-end decoding and utterance reproduction on one sample.
- Gate: objective overfit pass (`gates/overall_pass`) sustained for configured consecutive eval points.

2. `overfit_q8`
- Goal: repeat overfit protocol on Q8 before larger training.
- Gate: same structure as Q1 with Q8 config.

3. `baseline`
- Goal: controlled medium/long runs for selected config path.
- Requirement: immutable snapshot + periodic objective eval.

4. `ablation`
- Goal: compare one factor at a time against baseline.
- Requirement: frozen baseline + explicit hypothesis per run.

## Mandatory Per-Run Artifacts
Under `<artifact_root>/run_snapshot`:
- `manifest.json`
- `resolved_config.json`
- `env.json`
- `git.json`
- `git_diff.patch` (if dirty)
- `git_sha.txt`
- `git_status.txt`
- `untracked_files.txt`
- `run_ids.json`

Under `<artifact_root>/sample_artifacts/step_XXXXXX/sample_0`:
- `target_audio.wav`, `generated_unconstrained_audio.wav`, `generated_constrained_audio.wav` (when available)
- `*_codes_qt.csv`
- `*_heatmap_qt.csv`
- `*_summary.md`
- `sample_eval_metrics.json`
- `gate_metrics.json`

## Overfit Pass Defaults (Moderate)
- `wer_max = 0.35`
- `cer_max = 0.18`
- `frame_ratio_min = 0.60`
- `frame_ratio_max = 1.40`
- `coverage_abs_diff_max = 0.02`
- `coverage_q_min = 0.00` (set >0 only if you need explicit per-codebook activity)
- `coverage_q_abs_diff_max = 1.00` (set lower for strict per-codebook match)
- `min_consecutive_passes = 3`
- Unconstrained audio must appear by deadline.

## Run Registry
All finalized runs must be appended to:
- `experiments/runs/index.jsonl`

`experiments/EXPERIMENT_LOG.md` is rendered from this registry.

## Launch and Finalize
Use:
- `python scripts/exp/launch.py ...`
- `python scripts/exp/finalize.py --experiment-id ...`
- `python scripts/exp/render_log.py`
