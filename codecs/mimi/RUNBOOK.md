# Mimi Runbook

Canonical codec id: `mimi`

## Pretokenize FLEURS

```bash
python scripts/pretokenize_fleurs.py \
  --languages en hi es fr de ar sw ta bn zh \
  --split train \
  --num-quantizers 8 \
  --output-dir /vol/data/fleurs_pretok_q8 \
  --audio-codec-backend mimi
```

## Overfit custom sample (Q8)

```bash
modal run --detach modal/app.py::train \
  --path mimi/overfit_download_q8 \
  --experiment-id exp-mimi-q8-overfit \
  --steps 1000 \
  --num-quantizers 8 \
  --dataset-path /vol/data/custom_download_q8
```

Legacy alias still accepted:

```bash
modal run --detach modal/app.py::train --path overfit_download_q8
```
