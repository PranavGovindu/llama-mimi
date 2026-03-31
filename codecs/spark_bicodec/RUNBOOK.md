# Spark BiCodec Runbook

Canonical codec id: `spark_bicodec`

## 1) Pretokenize single sample (Q1 semantics)

```bash
python codecs/spark_bicodec/scripts/pretokenize_single_wav.py \
  --input-wav /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_spark_q1 \
  --split train \
  --lang en \
  --sample-id download_001 \
  --num-quantizers 1 \
  --max-seconds 20 \
  --audio-codec-model-id /root/spark-tts/pretrained_models/Spark-TTS-0.5B
```

## 2) Modal pretokenize single sample

```bash
modal run modal/app.py::pretokenize_single_wav \
  --input-wav-path /vol/data/raw/download.wav \
  --output-dir /vol/data/custom_download_spark_q1 \
  --quantizers 1 \
  --max-seconds 20 \
  --audio-codec-backend spark_bicodec \
  --audio-codec-model-id /root/spark-tts/pretrained_models/Spark-TTS-0.5B
```

## 3) Modal train overfit (Q1)

```bash
modal run --detach modal/app.py::train \
  --path spark_bicodec/overfit_download_q1 \
  --experiment-id exp-spark-q1-overfit \
  --steps 1000 \
  --num-quantizers 1 \
  --overfit-num-samples 1 \
  --dataset-path /vol/data/custom_download_spark_q1 \
  --audio-codec-backend spark_bicodec \
  --audio-codec-model-id /root/spark-tts/pretrained_models/Spark-TTS-0.5B
```

## 4) Inference wrapper

```bash
python codecs/spark_bicodec/scripts/inference_tts.py \
  --model-id <hf_or_local_model> \
  --text "hello world" \
  --lang en \
  --num-quantizers 1 \
  --spark-global-tokens "12,98,331" \
  --audio-codec-model-id /root/spark-tts/pretrained_models/Spark-TTS-0.5B \
  --output-file out_spark.wav
```

Notes:
- Spark decode requires global tokens. Provide them via `--spark-global-tokens`,
  `--spark-global-tokens-file`, or `--spark-prompt-audio`.
- Ensure Spark-TTS repo is mounted at `/root/spark-tts` for Modal jobs.
