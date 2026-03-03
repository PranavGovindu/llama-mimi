# S1 Track Runbook

This folder is an isolated experiment track for TinyAya + S1-DAC.
Existing Mimi workflows remain unchanged.

## 1) Pretokenize single sample (S1 Q10)

```bash
python s1_track/scripts/pretokenize_single_wav_s1.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_s1_q10 \
  --split train \
  --lang en \
  --sample-id download_001 \
  --num-quantizers 10 \
  --max-seconds 20 \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

## 2) Modal pretokenize single sample (S1)

```bash
modal run modal/app.py::pretokenize_single_wav_s1 \
  --input-wav-path /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_s1_q10 \
  --quantizers 10 \
  --max-seconds 20 \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

## 3) Modal train overfit (S1)

```bash
modal run --detach modal/app.py::train \
  --path overfit_download_s1_q10 \
  --experiment-id exp-s1-q10-overfit \
  --steps 1000 \
  --num-quantizers 10 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_s1_q10 \
  --audio-codec-backend s1_dac \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id jordand/fish-s1-dac-min \
  --checkpoint-interval 200 \
  --checkpoint-async-mode async
```

## 4) S1 inference wrapper

```bash
python s1_track/inference_tts_s1.py \
  --model-id <hf_or_local_model> \
  --text "hello world" \
  --lang en \
  --num-quantizers 10 \
  --audio-codec-source hf_pretrained \
  --audio-codec-model-id jordand/fish-s1-dac-min \
  --output-file out_s1.wav
```

## 5) Required gate signals by step <= 200

- `samples/generated_audio_0` exists
- `samples/generate_status_unconstrained_0 == ok`
- non-zero generated codebook coverage metrics

