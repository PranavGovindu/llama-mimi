# Spark Q1 Overfit Snapshot

- date: 2026-03-06
- experiment_id: exp-20260305-1750-spark-q1-download-overfit1k
- codec: spark_bicodec
- modal_app_id: ap-sBuhdjYirHkIyBjnKq8ibd
- wandb_run_id: vi0sjojp
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/vi0sjojp
- config: codecs/spark_bicodec/configs/tinyaya_spark_q1_download_overfit_1sample.toml
- modal_path: spark_bicodec/overfit_download_q1
- run_status: completed
- final_step: 1000

## Final Metrics (step 1000, sample 0)

- target_frames: 999
- unconstrained_frames: 999
- constrained_frames: 999
- target_coverage_total: 0.1114501953125
- unconstrained_coverage_total: 0.1114501953125
- constrained_coverage_total: 0.1114501953125
- gate_overall_pass: true
- gate_consecutive_passes: 44
- gate_passed_ever: true

## Stored Artifacts

- modal volume root: `/outputs/tiny-aya-fire_fleurs_pretok-q1-s8192-overfit_spark_q1_download/sample_artifacts/step_001000/sample_0/`
- local (ignored) snapshot dir: `experiments/runs/spark_bicodec/exp-20260305-1750-spark-q1-download-overfit1k/`
- files validated:
  - `generated_unconstrained_audio.wav`
  - `generated_constrained_audio.wav`
  - `target_audio.wav`
  - `generated_codes_qt.csv`
  - `generated_constrained_codes_qt.csv`
  - `generated_heatmap_qt.csv`
  - `sample_eval_metrics.json`
  - `gate_metrics.json`
  - `generated_summary.md`
  - `generated_constrained_summary.md`
