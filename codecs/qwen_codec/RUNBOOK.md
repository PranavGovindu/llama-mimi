# Qwen Codec Runbook

Canonical codec id: `qwen_codec`

Official codec source: `Qwen/Qwen3-TTS-Tokenizer-12Hz`

Notes:
- Qwen names this tokenizer `12Hz`, but the official config uses `encode_downsample_rate=1920` at `24 kHz`, which is `12.5 fps`.
- The tokenizer exposes `16` codebooks with codebook size `2048`.

## 1) Pretokenize single sample (Q16)

```bash
python codecs/qwen_codec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_qwen12hz_q16 \
  --split train \
  --lang en \
  --sample-id download_001 \
  --num-quantizers 16 \
  --max-seconds 20 \
  --audio-codec-model-id Qwen/Qwen3-TTS-Tokenizer-12Hz
```

## 2) Modal train smoke (Q16, 5 steps)

```bash
modal run --env main modal/app.py::train \
  --path qwen_codec/smoke_download_12hz_q16 \
  --experiment-id exp-qwen12hz-q16-smoke \
  --steps 5 \
  --num-quantizers 16 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_qwen12hz_q16 \
  --audio-codec-backend qwen_codec \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id Qwen/Qwen3-TTS-Tokenizer-12Hz
```

## 3) Modal train overfit (Q16, same sample)

```bash
modal run --detach --env main modal/app.py::train \
  --path qwen_codec/overfit_download_12hz_q16 \
  --experiment-id exp-qwen12hz-q16-download-overfit1k \
  --steps 1000 \
  --num-quantizers 16 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_qwen12hz_q16 \
  --audio-codec-backend qwen_codec \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id Qwen/Qwen3-TTS-Tokenizer-12Hz
```

## 4) Inference wrapper

```bash
python codecs/qwen_codec/scripts/inference_tts.py \
  --model-id <hf_or_local_model> \
  --text "hello world" \
  --lang en \
  --num-quantizers 16 \
  --audio-codec-model-id Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --output-file out_qwen12hz.wav
```
