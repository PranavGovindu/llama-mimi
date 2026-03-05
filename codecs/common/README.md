# Codec Architecture

This repo supports multiple audio codecs for TinyAya TTS.

## Contract

Each codec folder should include:
- `configs/` canonical training and overfit configs
- `scripts/` pretokenize/inference wrappers for that codec
- `RUNBOOK.md` operational commands
- `EXPERIMENT_LOG.md` codec-local run notes

## Naming

- Codec id examples: `mimi`, `s1_dac`, `spark_bicodec`
- Canonical modal paths follow: `<codec>/<profile>`
- Legacy aliases remain supported with deprecation warnings.
