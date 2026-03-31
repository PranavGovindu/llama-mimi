# Literature Notes (2026-03-09)

## Purpose
Summarize the open-system patterns that matter for TinyAya TTS design.

## 1. Speech representation trends

### Semantic-first tokenizers are winning for token-LM TTS
Recent open systems increasingly separate:
- a low-rate semantic or primary stream
- residual acoustic detail
- optional speaker/global tokens

Why this matters:
- the backbone LM should model linguistic progression, not every residual acoustic token equally
- this reduces autoregressive burden and improves streaming behavior

Examples:
- Llama-Mimi style: primary semantic codebook plus acoustic residual codebooks
- Qwen3-TTS 12Hz: one semantic/base codebook plus residual codebooks with grouped prediction
- Spark/BiCodec: semantic stream plus fixed global speaker/style tokens

### Single flat audio-token streams are still useful, but mostly as baselines
They are attractive because they are easy to train, but they do not scale cleanly once the representation becomes hierarchical.

Implication for TinyAya:
- flat token LM is acceptable for smoke tests and simple baselines
- it should not be the final design for large residual codebook tokenizers

## 2. Architecture trends

### Token-LM path
Strengths:
- natural fit for LM backbones like TinyAya
- direct text-to-speech-token modeling
- good for streaming and interactive generation

Weaknesses:
- quality depends heavily on tokenizer quality
- high-codebook tokenizers need better structure than pure flattening

### Flow / diffusion / DiT decoder path
Strengths:
- very strong waveform or mel quality in modern systems
- continuous modeling can improve naturalness

Weaknesses:
- heavier inference path
- less natural fit if we want TinyAya itself to stay central as the speech generator

Implication for TinyAya:
- keep TinyAya as the main speech-token planner
- do not replace the whole stack with a continuous decoder-only TTS design
- if needed, add a small residual or acoustic refinement module around TinyAya, not instead of TinyAya

## 3. Codec-specific takeaways

### Spark / BiCodec
Takeaway:
- best near-term controllable baseline because the representation is simple and global tokens carry speaker/style information explicitly

Research use:
- baseline for speaker/style conditioning
- baseline for quick multilingual pilots

### Qwen 12Hz tokenizer
Takeaway:
- strongest public evidence for a practical semantic-first low-rate multi-codebook tokenizer with grouped prediction

Research use:
- template for final architecture
- do not flatten all `16` codebooks and call it done

### Mimi
Takeaway:
- stable semantic-first tokenizer family and a good fallback if Qwen-specific integration becomes messy

Research use:
- safer grouped-codebook reference path

### X-Codec 2
Takeaway:
- attractive control baseline because it avoids residual grouped prediction entirely while staying tokenizer-based

Research use:
- useful to test whether one strong single-stream tokenizer beats a more complex grouped codec under the same TinyAya budget

## 4. Data trends

### Clean read speech is still necessary
Even if the goal is expressive TTS, recent strong systems still rely on clean anchors for alignment, pronunciation, and textual faithfulness.

Implication:
- do not start from only noisy expressive corpora
- use clean anchors first, then broaden style

### Natural and expressive data must be added deliberately
Natural multispeaker and expressive dialogue data improve realism, but they can also destabilize content if mixed too early or too aggressively.

Implication:
- curriculum matters
- clean -> natural -> expressive is a better research program than mixing everything from the start

## 5. Evaluation trends

### WER/CER remain necessary but are not enough
Good TTS can still fail on:
- speaker identity
- prosody
- naturalness
- language fidelity
- code-switch stability

Implication:
- token loss trains the model
- scorecards choose the model

### Human evaluation remains necessary at milestones
Automatic metrics are useful for ranking and ablations, but key checkpoints should still get human listening evaluation.

## 6. Immediate research conclusions for TinyAya

### Conclusion A
Do not commit to one codec before running tiny codec audits.

### Conclusion B
Do not scale flat high-codebook modeling.

### Conclusion C
Use a two-horizon program:
- quick baseline: Spark/BiCodec-style
- final system: grouped semantic-first multi-codebook path

### Conclusion D
Treat Hindi and Telugu as primary evaluation languages, not auxiliary ones.

## Sources
Local papers:
- [2509.14882v1.pdf](/home/pranav/TINYYAYAy/2509.14882v1.pdf)
- [fish.pdf](/home/pranav/TINYYAYAy/fish.pdf)
- [sparrk.pdf](/home/pranav/TINYYAYAy/sparrk.pdf)
- [Qwen3_TTS.pdf](/home/pranav/TINYYAYAy/Qwen3-TTS/assets/Qwen3_TTS.pdf)

Official repos and project pages:
- https://github.com/QwenLM/Qwen3-TTS
- https://github.com/spark-audio/Spark-TTS
- https://github.com/SWivid/F5-TTS
- https://github.com/AI4Bharat/IndicF5
- https://huggingface.co/datasets/ai4bharat/indicvoices_r
- https://huggingface.co/datasets/ai4bharat/Rasa
