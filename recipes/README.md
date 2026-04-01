# Recipes

This is the canonical home for runnable configs in the TinyAya TTS Lab.

Design rules:
- `recipes/` is the source of truth for new runs.
- `config/` and `codecs/*/configs/` are compatibility aliases generated from here.
- Paths are organized by model family first, then codec, then run intent.

## Layout

- `recipes/reference/llama3/`
  - reference TorchTitan / Llama configs kept for comparison and infra sanity
- `recipes/tinyaya/mimi/`
  - `fleurs/` multilingual baseline recipes
  - `overfit/` strict and custom overfit recipes
  - `smoke/` short validation and instrumentation recipes
  - `train/` real training recipes
  - `bench/` Hopper throughput and kernel benchmark recipes
- `recipes/tinyaya/s1_dac/overfit/`
- `recipes/tinyaya/spark_bicodec/overfit/`
- `recipes/tinyaya/spark_bicodec/train/`
- `recipes/tinyaya/dualcodec/overfit/`
- `recipes/tinyaya/dualcodec/smoke/`
- `recipes/tinyaya/qwen_codec/overfit/`
- `recipes/tinyaya/qwen_codec/smoke/`

## Naming

Filename rules:
- keep the full descriptive filename so W&B, Modal, and experiment logs are grep-friendly
- keep codec and major axes in the filename
- add suffixes only for:
  - `smoke`
  - `overfit`
  - `bench`
  - `hopper`

Path rules:
- use recipe paths in all new launch commands
- example:
  - `recipes/tinyaya/mimi/train/tinyaya_mimi_q8_s4096_emilia40k_en_clone_flat.toml`

## Compatibility

To regenerate legacy alias configs:

```bash
python scripts/dev/sync_config_aliases.py
```

That script rewrites:
- `config/*`
- `codecs/*/configs/*`

from the canonical recipe files in this directory.
