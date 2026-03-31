# TinyAya TTS Research Roadmap

## Mission
Build the strongest TinyAya-based TTS system we can under the current stack, with priority on:
- expressive speech
- multilingual performance
- Hindi, Telugu, and English first
- research discipline: no random tweaks without hypotheses, gates, and artifacts

## Current State
- The repo already supports multiple codec backends, Modal execution, W&B logging, run snapshots, and sample/codebook artifacts.
- The current modeling path is still mostly a flat causal LM over serialized audio tokens.
- That is acceptable for smoke tests and low-codebook baselines, but it is a bottleneck for full multi-codebook codecs.

## Core Bottlenecks
1. Flat audio-token serialization scales poorly with many codebooks.
2. Evaluation is still too ASR-centric for expressive multilingual TTS.
3. Dataset use is not yet organized as a disciplined curriculum.
4. Hindi/Telugu-specific scorecards are not yet first-class.
5. There is no dedicated grouped-codebook predictor path yet for high-codebook codecs.

## Research Thesis
The best path is not a single experiment. It is a staged program:

1. Prove codec and tokenizer health on tiny pilots.
2. Establish strong multilingual content baselines with low-complexity representations.
3. Add expressive control and richer acoustic detail.
4. Replace flat high-codebook modeling with grouped or hierarchical prediction.
5. Evaluate every step with a stable scorecard, not only loss curves.

## Primary Tracks

### Track A: Stable Multilingual Baseline
Goal:
- get reliable English/Hindi/Telugu TTS with clean content accuracy

Initial representation bias:
- low-codebook or semantic-heavy codec tracks first

Why:
- current flat LM stack can support these without exploding sequence length

### Track B: Expressiveness
Goal:
- improve prosody, dialogue naturalness, style variation, and speaker realism

Likely ingredients:
- richer data
- speaker/style conditioning
- dialogue/emotion supervision
- stronger eval beyond WER/CER

### Track C: High-Codebook Modeling
Goal:
- support full acoustic codecs without forcing the backbone to autoregress over every residual token

Target design:
- primary codebook modeled by the backbone
- residual codebooks modeled by a grouped predictor / MTP-style module

### Track D: Data Program
Goal:
- move from one-sample overfit to disciplined multilingual corpora and fixed evaluation packs

Priority languages:
- English
- Hindi
- Telugu

Priority dataset progression:
1. tiny pilot subsets for codec audits
2. clean read-speech anchors
3. natural multispeaker expansion
4. expressive/dialogue data

## Immediate Priorities
1. Freeze the research protocol and scorecard.
2. Audit candidate codecs on tiny multilingual pilots before broad pretokenization.
3. Fix multilingual evaluation, especially native-script WER/CER handling.
4. Build a clean multilingual baseline on the easiest viable representation.
5. Design grouped-codebook modeling for the final path.

## Non-Goals
- chasing paper-comparison numbers with no reproducibility
- treating overfit success as product-level progress
- running full-codebook flat baselines as the final design
- tokenizing giant corpora before the representation passes a pilot audit

## Exit Criteria For Phase 1
- reproducible multilingual baseline on English/Hindi/Telugu
- stable W&B dashboards and finalized experiment logs
- objective content metrics across all three languages
- audible samples with consistent failure analysis

## Exit Criteria For Phase 2
- grouped high-codebook model integrated
- expressive improvements visible in both listening tests and scorecard
- no regression in multilingual intelligibility
- Hindi and Telugu quality are competitive, not sacrificed for English
