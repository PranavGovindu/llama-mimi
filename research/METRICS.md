# TinyAya TTS Research Metrics

## Principle
Loss is not the decision metric. Every important run needs a scorecard.

## Tier 1: Content Accuracy
- multilingual ASR WER
- multilingual ASR CER
- per-language breakdown for English, Hindi, Telugu

Recommended tools:
- strong multilingual Whisper or language-specific ASR where it clearly helps
- fix current text normalization so native scripts are not destroyed before scoring

Purpose:
- measure whether the model says the intended words

## Tier 2: Language Fidelity
- language identification accuracy
- code-switch robustness checks for English/Hindi/Telugu
- script-preserving text normalization for evaluation

Purpose:
- detect collapse into the wrong language or accent drift

## Tier 3: Speaker / Timbre Consistency
- speaker similarity score against reference audio
- reference-free clustering sanity for multispeaker runs

Recommended tools:
- ECAPA / CAM++ / strong speaker embedding baselines

Purpose:
- check identity preservation and timbre control

## Tier 4: Prosody / Expressiveness
- pause statistics
- speaking rate
- pitch range and variance
- energy dynamics
- duration alignment where paired reference exists
- listening panels for key checkpoints

Purpose:
- avoid optimizing only text correctness while sounding dead

## Tier 5: Naturalness
- UTMOSv2
- TTSDS where distributional evaluation is useful
- optional reference-aware PESQ/STOI/MCD style metrics on paired subsets

Purpose:
- monitor realism beyond intelligibility

## Tier 6: Codec / Token Behavior
- frame counts
- total codebook coverage
- per-codebook coverage
- generated vs target codebook drift
- malformed decode count

Purpose:
- expose representation collapse early

## Tier 7: Systems Metrics
- tokens/sec
- GPU memory
- checkpoint size
- inference latency

Purpose:
- prevent building an unusable system

## Mandatory Report Sections For Important Runs
1. hypothesis
2. exact config and git SHA
3. dataset definition
4. core metrics table
5. sample links
6. failure modes
7. next decision

## References
- https://github.com/sarulab-speech/UTMOSv2
- https://github.com/ttsds/ttsds
- https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
