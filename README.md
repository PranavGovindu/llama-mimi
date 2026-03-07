# TinyAya TTS Lab (Multi-Codec)

This repository is now organized as a **TinyAya TTS experimentation lab** with codec-specific tracks.

## What This Repo Supports

- TinyAya overfit and training workflows
- Codec-specific pipelines (`mimi`, `s1_dac`, `spark_bicodec`, `dualcodec`, future codecs)
- Modal-first training and pretokenization
- Structured experiment tracking with per-codec registries

Legacy Llama-Mimi project documentation has moved to:
- `docs/legacy/llama-mimi.md`

## Layout

- `codecs/common/` shared codec architecture notes
- `codecs/mimi/` Mimi-specific configs and runbooks
- `codecs/s1_dac/` S1-DAC-specific configs, scripts, and logs
- `codecs/spark_bicodec/` Spark BiCodec configs, scripts, and logs
- `codecs/dualcodec/` DualCodec configs, scripts, and logs
- `config/` backward-compatible config aliases
- `scripts/exp/` launch/finalize/render experiment tooling
- `experiments/runs/<codec>/index.jsonl` per-codec run registries

## Setup

```bash
uv sync
```

## Canonical Codec Paths

### Mimi (Q8 custom overfit)

```bash
modal run --detach modal/app.py::train \
  --path mimi/overfit_download_q8 \
  --experiment-id exp-mimi-q8-001 \
  --steps 1000
```

### S1-DAC (Q9 custom overfit)

```bash
modal run --detach modal/app.py::train \
  --path s1_dac/overfit_download_q9 \
  --experiment-id exp-s1-q9-001 \
  --steps 1000 \
  --num-quantizers 9 \
  --audio-codec-backend s1_dac \
  --audio-codec-source official_fish \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

### Spark BiCodec (Q1 semantic stream)

```bash
modal run --detach modal/app.py::train \
  --path spark_bicodec/overfit_download_q1 \
  --experiment-id exp-spark-q1-001 \
  --steps 1000 \
  --num-quantizers 1 \
  --audio-codec-backend spark_bicodec \
  --audio-codec-model-id /root/spark-tts/pretrained_models/Spark-TTS-0.5B
```

### DualCodec (12hz Q8)

```bash
modal run --detach modal/app.py::train \
  --path dualcodec/overfit_download_12hz_q8 \
  --experiment-id exp-dualcodec-12hz-q8-001 \
  --steps 1000 \
  --num-quantizers 8 \
  --audio-codec-backend dualcodec \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id 12hz_v1
```

## Pretokenization

### Mimi

```bash
python scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_q8 \
  --num-quantizers 8
```

### S1-DAC

```bash
python codecs/s1_dac/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_s1_q9 \
  --num-quantizers 9 \
  --audio-codec-source official_fish \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

### Spark BiCodec

```bash
python codecs/spark_bicodec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_spark_q1 \
  --num-quantizers 1 \
  --audio-codec-model-id /root/spark-tts/pretrained_models/Spark-TTS-0.5B
```

### DualCodec

```bash
python codecs/dualcodec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_dualcodec_12hz_q8 \
  --num-quantizers 8 \
  --audio-codec-model-id 12hz_v1
```

## Compatibility Aliases (Deprecated)

These still work but print deprecation warnings:

- `--path overfit_download_q8` -> `mimi/overfit_download_q8`
- `--path overfit_download_s1_q10` -> `s1_dac/overfit_download_q9`
- `s1_track/*` scripts -> `codecs/s1_dac/scripts/*`

## Experiment Tracking

Per-codec indexes:

- `experiments/runs/mimi/index.jsonl`
- `experiments/runs/s1_dac/index.jsonl`
- `experiments/runs/spark_bicodec/index.jsonl`
- `experiments/runs/dualcodec/index.jsonl`

Aggregate views:

- `experiments/runs/index.jsonl`
- `experiments/EXPERIMENT_LOG.md`

Render logs:

```bash
python scripts/exp/render_log.py
```

## Migration Utilities

- Run layout migration:

```bash
python scripts/ops/migrate_runs_to_codec_layout.py
python scripts/ops/migrate_runs_to_codec_layout.py --execute
```

- Rollback using saved migration report:

```bash
python scripts/ops/migrate_runs_to_codec_layout.py \
  --rollback-manifest experiments/runs/migration_report.json --execute
```

## Notes

- `modal/app.py::train` runs on `H200` by default.
- Canonical S1 profile uses Q9 (`jordand/fish-s1-dac-min`).
- Canonical Spark profile uses Q1 semantic stream + global prompt tokens.
- Keep future codec-specific work inside `codecs/<codec>/`.
