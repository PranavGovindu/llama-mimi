# TinyAya TTS Lab (Multi-Codec)

This repository is now organized as a **TinyAya TTS experimentation lab** with codec-specific tracks.

## What This Repo Supports

- TinyAya overfit and training workflows
- Codec-specific pipelines (`mimi`, `s1_dac`, `spark_bicodec`, `dualcodec`, `qwen_codec`, future codecs)
- Modal-first training and pretokenization
- Structured experiment tracking with per-codec registries

Legacy Llama-Mimi project documentation has moved to:
- `docs/legacy/llama-mimi.md`

## Layout

- `recipes/` canonical run configs, organized by model family, codec, and intent
- `torchtitan/` core training, data, metrics, and model code
- `modal/` Modal entrypoints and probes
- `codecs/<codec>/` codec-specific scripts, runbooks, and notes
- `experiments/runs/<codec>/index.jsonl` structured run registries
- `research/` sweep plans, scorecards, and review artifacts
- `config/` and `codecs/*/configs/` generated compatibility aliases

Start with:
- [recipes/README.md](/home/pranav/TINYYAYAy/llama-mimi/recipes/README.md)
- [docs/repo_layout.md](/home/pranav/TINYYAYAy/llama-mimi/docs/repo_layout.md)

## Setup

```bash
uv sync
```

## Modal FA3 Training

Use the slim FA3 launcher in [modal/train_fa3.py](/home/pranav/TINYYAYAy/llama-mimi/modal/train_fa3.py), not the large mixed [modal/app.py](/home/pranav/TINYYAYAy/llama-mimi/modal/app.py) path.

This launcher supports two image modes:
- default:
  - [Dockerfile.train.base](/home/pranav/TINYYAYAy/llama-mimi/Dockerfile.train.base) for `probe_ngc_stack` and dataset sync
  - [Dockerfile.train](/home/pranav/TINYYAYAy/llama-mimi/Dockerfile.train) for kernels-backed FA3 probe and training
- faster later: set `MODAL_TRAIN_REGISTRY_IMAGE=<your-registry>/<image>:<tag>` and Modal will pull that prebuilt image instead

If the registry image is private, also set:

```bash
export MODAL_TRAIN_REGISTRY_SECRET=<modal-secret-name>
```

### 1. Probe the native NGC stack

```bash
modal run --detach modal/train_fa3.py::probe_ngc_stack
```

### 2. Probe FA3 itself

```bash
modal run --detach modal/train_fa3.py::probe_fa3
```

This uses the Hugging Face kernels backend:

```text
kernels-community/flash-attn3
```

### 3. Mirror the training dataset to the Modal volume

```bash
modal run --detach modal/train_fa3.py::sync_dataset \
  --repo-id Pranavz/emilia-en-mimi-q8-s4096-dynamic-20260329a-public \
  --local-dir /vol/data/datasets/emilia-en-mimi-q8-s4096-dynamic-20260329a-public
```

### 4. Launch FA3 training on Modal

```bash
modal run --detach modal/train_fa3.py::train \
  --config-file recipes/tinyaya/mimi/train/tinyaya_mimi_q8_s4096_emilia40k_en_clone_flat.toml \
  --dataset-path /vol/data/datasets/emilia-en-mimi-q8-s4096-dynamic-20260329a-public \
  --attn-implementation kernels-community/flash-attn3 \
  --steps 1000 \
  --experiment-id exp-emilia-clone-flat-fa3 \
  --checkpoint-interval 200 \
  --checkpoint-keep-latest-k 10 \
  --checkpoint-folder emilia_clone_flat_fa3
```

## Portable Spot-Instance Training

For serious training, use the repo’s reusable Docker path instead of rebuilding
training images inside Modal over and over.

### 1. Build the training image

This image uses the NVIDIA PyTorch NGC base and installs the Hugging Face
`kernels` package so training can use `kernels-community/flash-attn3`.

```bash
docker build -f Dockerfile.train -t tinyaya-mimi-train:fa3 .
```

If you are pulling directly from `nvcr.io`, log in first:

```bash
docker login nvcr.io
# username: $oauthtoken
# password: <your NGC API key>
```

### 2. Smoke the native stack

```bash
docker run --rm --gpus all --ipc=host tinyaya-mimi-train:fa3 probe-ngc-stack
```

### 3. Smoke FA3 itself

```bash
docker run --rm --gpus all --ipc=host \
  -e HF_TOKEN \
  tinyaya-mimi-train:fa3 probe-fa3
```

### 4. Mirror the HF dataset locally

The current training loader expects `training.dataset_path` to point at a local
Parquet tree.

```bash
docker run --rm \
  --env-file .env.train \
  -v /mnt/data:/mnt/data \
  tinyaya-mimi-train:fa3 sync-dataset \
    --repo-id Pranavz/emilia-en-mimi-q8-s4096-dynamic-20260329a-public \
    --local-dir /mnt/data/emilia-en-mimi-q8-s4096
```

### 5. Launch training

```bash
docker run --rm --gpus all --ipc=host \
  --env-file .env.train \
  -v /mnt/cache:/mnt/cache \
  -v /mnt/data:/mnt/data \
  -v /mnt/checkpoints:/outputs \
  tinyaya-mimi-train:fa3 train \
    --job.config_file recipes/tinyaya/mimi/train/tinyaya_mimi_q8_s4096_emilia40k_en_clone_flat.toml \
    --job.dump_folder /outputs \
    --training.dataset_path /mnt/data/emilia-en-mimi-q8-s4096 \
    --model.attn_implementation kernels-community/flash-attn3 \
    --checkpoint.enable_checkpoint true \
    --checkpoint.folder emilia_clone_flat \
    --checkpoint.interval 200 \
    --checkpoint.keep_latest_k 10 \
    --checkpoint.async_mode async
```

The entrypoint auto-defaults:
- HF cache under `/cache/huggingface`
- W&B files under `/cache/wandb`
- `torchrun` world setup from `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`

For a template env file, copy:

```bash
cp .env.train.example .env.train
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

### Qwen Codec (12Hz Q16)

```bash
modal run --detach modal/app.py::train \
  --path qwen_codec/overfit_download_12hz_q16 \
  --experiment-id exp-qwen12hz-q16-001 \
  --steps 1000 \
  --num-quantizers 16 \
  --audio-codec-backend qwen_codec \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id Qwen/Qwen3-TTS-Tokenizer-12Hz
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

### Qwen Codec

```bash
python codecs/qwen_codec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_qwen12hz_q16 \
  --num-quantizers 16 \
  --audio-codec-model-id Qwen/Qwen3-TTS-Tokenizer-12Hz
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
- `experiments/runs/qwen_codec/index.jsonl`

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
- Keep future runnable configs inside `recipes/`, not `config/` or `codecs/*/configs/`.
