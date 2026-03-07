# DualCodec Runbook

Canonical codec id: `dualcodec`

## 1) Pretokenize single sample (12hz Q1 smoke)

```bash
python codecs/dualcodec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_dualcodec_12hz_q1 \
  --split train \
  --lang en \
  --sample-id download_001 \
  --num-quantizers 1 \
  --max-seconds 20 \
  --audio-codec-model-id 12hz_v1
```

## 2) Pretokenize single sample (12hz Q8)

```bash
python codecs/dualcodec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_dualcodec_12hz_q8 \
  --split train \
  --lang en \
  --sample-id download_001 \
  --num-quantizers 8 \
  --max-seconds 20 \
  --audio-codec-model-id 12hz_v1
```

## 3) Modal train overfit (12hz Q8)

```bash
modal run --detach modal/app.py::train \
  --path dualcodec/overfit_download_12hz_q8 \
  --experiment-id exp-dualcodec-12hz-q8-overfit \
  --steps 1000 \
  --num-quantizers 8 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_dualcodec_12hz_q8 \
  --audio-codec-backend dualcodec \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id 12hz_v1
```

## 4) Modal train overfit (25hz Q12)

```bash
modal run --detach modal/app.py::train \
  --path dualcodec/overfit_download_25hz_q12 \
  --experiment-id exp-dualcodec-25hz-q12-overfit \
  --steps 1000 \
  --num-quantizers 12 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_dualcodec_25hz_q12 \
  --audio-codec-backend dualcodec \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id 25hz_v1
```

## 5) Inference wrapper

```bash
python codecs/dualcodec/scripts/inference_tts.py \
  --model-id <hf_or_local_model> \
  --text "hello world" \
  --lang en \
  --num-quantizers 8 \
  --audio-codec-model-id 12hz_v1 \
  --output-file out_dualcodec.wav
```

Notes:
- Supported model ids: `12hz_v1` (max 8 codebooks) and `25hz_v1` (max 12 codebooks).
- DualCodec depends on `dualcodec` package and w2v-bert semantic features.
- Optional env override for semantic model path: `DUALCODEC_W2V_PATH`.
