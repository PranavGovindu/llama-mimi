# Autonomous Research OS

This repo now supports a structured, long-running TTS research loop. The current codebase remains lab infrastructure, not the thesis.

## Core Commands

Create a dedicated research worktree from the current dirty state:

```bash
python scripts/research/bootstrap_worktree.py --worktree-path ../llama-mimi-research
```

Initialize a campaign:

```bash
python scripts/research/init_campaign.py \
  --campaign-id enhi-expref-20260309 \
  --owner pranav \
  --languages en hi
```

Launch an experiment with campaign metadata:

```bash
python scripts/exp/launch.py \
  --mode modal \
  --modal-path spark_bicodec/overfit_download_q1 \
  --experiment-id exp-20260309-0001 \
  --campaign-id enhi-expref-20260309 \
  --track representation \
  --axis codec \
  --family semantic_global_tokens \
  --question "Does the baseline codec produce reliable EN/HI audio?" \
  --hypothesis "Semantic+global tokens are the fastest path to a controllable baseline."
```

Build a scorecard from local artifacts:

```bash
python scripts/research/build_scorecard.py \
  --experiment-id exp-20260309-0001 \
  --dump-root /path/to/run/output
```

Write the review memo:

```bash
python scripts/research/write_review.py \
  --experiment-id exp-20260309-0001 \
  --label inconclusive \
  --decision rerun \
  --summary "Audio decoded, but EN/HI scoring is still too weak." \
  --next-action "Repeat with corrected bilingual normalization."
```

## Operating Rules

- One experiment answers one question.
- A direction advances on cumulative evidence, not one exciting run.
- Use `invalid`, `negative`, `inconclusive`, and `positive` as run-level labels.
- Stop early on repeated malformed decodes, codebook collapse, or EN/HI language collapse.
- The system may run pilots and ablations autonomously, but it stops before a promoted long/full main run.
