# S1-DAC Runbook

Canonical codec id: `s1_dac`

## 1) Pretokenize single sample (Q9)

```bash
python codecs/s1_dac/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_s1_q9 \
  --split train \
  --lang en \
  --sample-id download_001 \
  --num-quantizers 9 \
  --max-seconds 20 \
  --audio-codec-source official_fish \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

## 2) Modal pretokenize single sample

```bash
modal run modal/app.py::pretokenize_single_wav_s1 \
  --input-wav-path /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_s1_q9 \
  --quantizers 9 \
  --max-seconds 20 \
  --audio-codec-source official_fish \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

## 3) Modal train overfit (Q9)

```bash
modal run --detach modal/app.py::train \
  --path s1_dac/overfit_download_q9 \
  --experiment-id exp-s1-q9-overfit \
  --steps 1000 \
  --num-quantizers 9 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_s1_q9 \
  --audio-codec-backend s1_dac \
  --audio-codec-source official_fish \
  --audio-codec-model-id jordand/fish-s1-dac-min
```

Legacy alias still accepted:

```bash
modal run --detach modal/app.py::train --path overfit_download_s1_q10
```

## 4) Inference wrapper

```bash
python codecs/s1_dac/scripts/inference_tts.py \
  --model-id <hf_or_local_model> \
  --text "hello world" \
  --lang en \
  --num-quantizers 9 \
  --audio-codec-source official_fish \
  --audio-codec-model-id jordand/fish-s1-dac-min \
  --output-file out_s1.wav
```
