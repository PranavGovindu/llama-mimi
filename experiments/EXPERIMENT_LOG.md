# Experiment Log

Auto-generated from `experiments/runs/index.jsonl`.

## 2026-03-02 - exp-20260302-verify-q8-samples-v3
- status: failed
- phase: 
- mode: 
- config: 
- modal_path: 
- wandb_run_id: es1olgdr
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/es1olgdr
- modal_app_id: ap-mlD4wugoCYnV1NRxLJTyVi
- notes: Direct detached smoke run; samples logged but unconstrained decode status stayed no_audio_span by step 25.

## 2026-03-02 - exp-20260302-131200-q8-download-overfit-h200-s8192-retry2
- status: stopped
- phase: overfit_q8_custom
- mode: modal
- config: config/tinyaya_q1_fleurs_overfit_1sample_strict.toml
- modal_path: overfit_download_q8
- wandb_run_id: u9jr38cn
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/u9jr38cn
- modal_app_id: ap-Ef3I76GcfqCT22HVqGFt64
- notes: Accidental duplicate detached launch; manually stopped early.

## 2026-03-02 - exp-20260302-verify-q8-samples-v2
- status: failed
- phase: overfit_q8_custom
- mode: modal
- config: config/tinyaya_q1_fleurs_overfit_1sample_strict.toml
- modal_path: overfit_download_q8
- wandb_run_id: 2ynenf1l
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/2ynenf1l
- modal_app_id: ap-qoaANXWtNEWM3qPM4hyIpF
- notes: Smoke verification: samples/media logged; run failed strict gate due no unconstrained decodable audio by step 25.

## 2026-02-28 - overfit1-viz5-smoke
- status: completed
- phase: overfit_q1
- mode: modal
- config: config/tinyaya_q1_fleurs_overfit_1sample_viz5.toml
- modal_path: overfit_viz5
- wandb_run_id: c14ebnit
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/c14ebnit
- modal_app_id: ap-3OCG7zfrDb0rTVaBHou12f
- notes: 5-step instrumentation check; media and artifacts synced.

## 2026-02-28 - overfit1-strict-2k-attempt
- status: stopped
- phase: overfit_q1
- mode: modal
- config: config/tinyaya_q1_fleurs_overfit_1sample_strict.toml
- modal_path: overfit_strict
- wandb_run_id: nem3xhxp
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/nem3xhxp
- modal_app_id: ap-TLJmiWBhLuZk3WjXovYASH
- notes: Stopped by request at step 100 during checkpoint save.

## 2026-02-28 - overfit1-strict-initial
- status: stopped
- phase: overfit_q1
- mode: modal
- config: config/tinyaya_q1_fleurs_overfit_1sample_strict.toml
- modal_path: overfit_strict
- wandb_run_id: m4t8k6vd
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/m4t8k6vd
- modal_app_id: ap-ZMEzSeeKsf6xUWw23C5QcW
- notes: Confirmed generated unconstrained/constrained and target audio keys in W&B.

