# Migration Guide v1 (Codec Refactor)

## Path Mapping

- `overfit_download_q8` -> `mimi/overfit_download_q8`
- `overfit_download_s1_q10` -> `s1_dac/overfit_download_q9`

## Script Mapping

- `s1_track/inference_tts_s1.py` -> `codecs/s1_dac/scripts/inference_tts.py`
- `s1_track/scripts/pretokenize_single_wav_s1.py` -> `codecs/s1_dac/scripts/pretokenize_single_wav.py`
- `s1_track/scripts/pretokenize_fleurs_s1.py` -> `codecs/s1_dac/scripts/pretokenize_fleurs.py`

## Config Mapping

- `config/tinyaya_q8_download_overfit_1sample.toml` -> `codecs/mimi/configs/tinyaya_mimi_q8_download_overfit_1sample.toml`
- `config/tinyaya_s1_q10_download_overfit_1sample.toml` -> `codecs/s1_dac/configs/tinyaya_s1_q9_download_overfit_1sample.toml`

## Experiment Registry Mapping

- old: `experiments/runs/index.jsonl`
- new primary:
  - `experiments/runs/mimi/index.jsonl`
  - `experiments/runs/s1_dac/index.jsonl`
- aggregate retained at: `experiments/runs/index.jsonl`

## One-Time Run Layout Migration

```bash
python scripts/ops/migrate_runs_to_codec_layout.py
python scripts/ops/migrate_runs_to_codec_layout.py --execute
```
