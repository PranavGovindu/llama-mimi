# Repo Layout

This repo is organized around the TinyAya TTS Lab workflow, not around the old
`llama-mimi` project name.

The goal is the same separation-of-concerns pattern used in repos like
NVIDIA's Nemotron:
- one place for canonical recipes
- one place for core framework code
- one place for platform launchers
- one place for experiment records

## Top-Level Map

- `recipes/`
  - canonical configs for real runs, smokes, overfit checks, and benchmarks
- `torchtitan/`
  - core training/runtime code
- `modal/`
  - Modal launchers and environment-specific entrypoints
- `scripts/`
  - automation, experiment utilities, migration tools, and portable helpers
- `experiments/`
  - run indexes, logs, and human-readable experiment summaries
- `research/`
  - sweeps, plans, scorecards, and review artifacts
- `codecs/`
  - codec-specific scripts and notes
- `docs/`
  - stable documentation and layout guides

## Naming Policy

Use `TinyAya TTS Lab` as the product/repo identity in docs, app names, and new
paths. Keep `llama-mimi` only where it is unavoidable for historical
compatibility, such as:
- old repo path names
- legacy docs
- backward-compatible config aliases
- old experiment records

## Config Policy

New configs should be added only under `recipes/`.

Legacy locations:
- `config/`
- `codecs/*/configs/`

are treated as generated alias layers and should not be edited directly.

## Experiment Policy

Treat these as separate concerns:
- `recipes/` = how to run
- `experiments/runs/*.jsonl` = what was run
- `research/` = why it was run
