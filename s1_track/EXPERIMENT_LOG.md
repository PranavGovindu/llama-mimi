# S1 Track Experiment Log

Use this template for each run.

## Run Metadata

- Date:
- Owner:
- Experiment ID:
- Git SHA:
- Config file:
- Modal app ID:
- W&B run URL:

## Codec

- backend:
- source:
- model_ref or ckpt:
- sample_rate:
- codebook_size:
- num_quantizers:

## Data

- dataset_path:
- sample_id:
- utterance:
- duration_sec:

## Train Settings

- steps:
- seq_len:
- lr:
- checkpoint_interval:
- checkpoint_async_mode:

## Observations

- generated_audio_0 first seen step:
- status_unconstrained:
- status_constrained:
- codebook coverage notes:
- subjective audio quality notes:

## Outcome

- pass/fail:
- reason:
- next action:

---

## Completed Runs

### 2026-03-03 | exp-s1-q9-h200-20260303-1956

## Run Metadata

- Date: 2026-03-03
- Owner: pranav
- Experiment ID: exp-s1-q9-h200-20260303-1956
- Git SHA: 9650176
- Config file: `config/tinyaya_s1_q10_download_overfit_1sample.toml` (runtime overrides: `num_quantizers=9`, dataset path and codec model id)
- Modal app ID: `ap-WruRwPHsulZf2Um4nDMByr`
- W&B run URL: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/552xsfwg

## Codec

- backend: `s1_dac`
- source: `official_fish`
- model_ref or ckpt: `jordand/fish-s1-dac-min`
- sample_rate: `44100`
- codebook_size: `1024`
- num_quantizers: `9`

## Data

- dataset_path: `/vol/data/custom_download_s1_q9`
- sample_id: `download_001`
- utterance: `I don't know why, but lately I've been thinking about how strange it is...`
- duration_sec: `20.0`

## Train Settings

- steps: `1000`
- seq_len: `8192`
- lr: `5e-5`
- checkpoint_interval: `0`
- checkpoint_async_mode: `disabled`

## Observations

- generated_audio_0 first seen step: `140`
- status_unconstrained: `ok` from step 140 onward
- status_constrained: `ok` from step 140 onward
- codebook coverage notes: target coverage `0.96875`, generated coverage around `0.874` by step 1000
- subjective audio quality notes: generated audio present but still overfit-quality and not production quality

## Outcome

- pass/fail: `partial pass`
- reason: run completed 1000 steps and generated audio/codebook artifacts were produced; strict gate did not converge to full pass
- next action: shorten clip to 8-12s for faster stabilization, then curriculum to 20s/30s
