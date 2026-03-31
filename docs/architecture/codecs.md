# Multi-Codec Architecture

## Objective

Support many codecs without scattering codec-specific logic across the repo.

## Canonical Pattern

- Shared training/inference core stays in `torchtitan/` and top-level scripts.
- Codec-specific wrappers/configs/logs live under `codecs/<codec>/`.
- Modal profile IDs are codec-aware: `<codec>/<profile>`.

## Adding a New Codec

1. Create `codecs/<new_codec>/` with:
   - `configs/`
   - `scripts/`
   - `RUNBOOK.md`
   - `EXPERIMENT_LOG.md`
2. Add backend handling in `torchtitan/tools/audio_codec.py` and registry routing in `torchtitan/tools/codecs/registry.py`.
3. Add canonical modal path mapping in `modal/app.py`.
4. Add per-codec registry support by writing rows with `codec=<new_codec>`.

## Compatibility Policy

- Legacy aliases remain for one cleanup cycle.
- All aliases print deprecation warnings.
- Canonical references are always codec-aware.
