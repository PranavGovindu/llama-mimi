# TinyAya TTS Experiment Protocol v2

## Principle
Every experiment must answer one question.

Bad experiment:
- change codec, dataset, loss, batch size, and prompt format at once

Good experiment:
- one primary hypothesis
- one comparison baseline
- one pass/fail decision

## Research Axes
We will study these axes separately:
1. codec / representation
2. architecture
3. dataset mixture
4. loss design
5. conditioning method
6. evaluation stack

## Stage 0: Codec Audit
Before training on any sizable corpus, each candidate codec gets a research audit.

Candidate set:
- Spark/BiCodec
- Mimi
- Qwen 12Hz tokenizer
- X-Codec 2
- optional FireRed tokenizer later

For each codec on a tiny pilot subset:
- encode/decode success rate
- decode quality by listening
- token rate per second
- codebook occupancy and collapse checks
- reconstruction metrics where references exist
- memory and throughput cost

Exit criteria:
- no silent or malformed decodes
- stable parser behavior
- sensible codebook usage
- acceptable sequence length for the planned architecture

## Stage 1: One-sample overfit
Purpose:
- prove the training loop can produce audio at all
- debug tokenization, decoding, logging, sample rendering, and gating

Requirements:
- target audio logged
- generated audio logged
- utterance text logged with every sample
- local sample artifacts saved
- codebook plots saved

A run that only lowers loss but never yields valid audio is a failed systems test.

## Stage 2: Small-pack overfit
Use `32` to `128` samples, balanced across English, Hindi, Telugu.

Purpose:
- see whether the model memorizes a micro-distribution instead of one utterance
- catch script handling and language leakage problems early

## Stage 3: Representation baseline pilot
Train only small pilots, not full corpora.

Recommended pilot size:
- `1k` to `5k` utterances per language

Compare:
- Spark baseline under flat LM
- one stronger semantic-first codec path
- optional X-Codec 2 single-stream control

Decision rule:
- we do not continue a codec path only because it is fashionable
- it must win on a scorecard and engineering tractability

## Stage 4: Architecture ablation
This stage answers whether flat token LM is enough.

Compare:
- flat token LM baseline
- grouped residual predictor / MTP-style model
- optional style/global token conditioning

Decision rule:
- if grouped prediction is clearly better on quality, latency, or stability, flat high-codebook modeling is retired

## Stage 5: Dataset curriculum ablation
Compare:
- clean-only training
- clean + natural multispeaker
- clean + natural + expressive adaptation

Purpose:
- separate content gains from style gains
- avoid blaming codec for data problems

## Loss Design

### Core training loss
For discrete-token TTS, the primary loss remains token prediction cross-entropy.

Why:
- the model is predicting discrete speech units
- CE is the correct direct supervision for those units

### Why CE alone is not enough for decisions
CE is enough to train the discrete targets, but not enough to judge success.

It does not tell us:
- whether speech is natural
- whether speaker identity is preserved
- whether language drift is happening
- whether prosody is dead

### Recommended training losses by architecture

#### Flat low-codebook baseline
- audio-token CE
- optional text masking so audio loss dominates when desired

#### Grouped multi-codebook model
- primary semantic codebook CE
- weighted residual codebook CE
- optional stop/frame boundary objective

#### Style-conditioned model
- above losses
- optional speaker/style contrastive or classification auxiliary loss if a reference encoder is added

### What should stay as eval, not main training loss
Do not rush waveform or spectrogram losses into the TinyAya LM stage.
Use these first as evaluation tools:
- STOI
- PESQ
- mel distance or MCD
- UTMOSv2
- TTSDS

Reason:
- backpropagating through codec decode is expensive and complicates ablations
- first determine whether representation and architecture are correct

## Evaluation Scorecard
Every serious checkpoint needs:

### Content
- multilingual WER
- multilingual CER
- per-language breakdown

### Language fidelity
- language ID accuracy
- code-switch failure count

### Speaker
- speaker similarity embedding score

### Prosody
- F0 range and variance
- speaking rate
- pause ratio
- duration alignment for paired references

### Naturalness
- UTMOSv2
- TTSDS where applicable

### Codec health
- frame ratio
- codebook occupancy
- per-codebook drift
- malformed audio count

### Systems
- tokens per second
- memory
- inference latency
- checkpoint size

## Logging Rules
Every important run must capture:
- git SHA
- exact config
- dataset manifest
- W&B run id
- Modal app id
- sample artifacts on disk
- experiment card with hypothesis and next decision

## Stop / Go Gates

### Stop
- no generated audio by deadline
- malformed decodes
- script/language corruption
- dead prosody despite low loss
- evaluation regression in Hindi or Telugu

### Go
- valid generated audio
- stable multilingual transcription quality
- no major language collapse
- acceptable speaker/prosody metrics
- engineering cost justified for next scale step
