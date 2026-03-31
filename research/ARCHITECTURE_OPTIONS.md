# TinyAya TTS Architecture Options

## Goal
Turn TinyAya into a strong multilingual and expressive TTS backbone for English, Hindi, and Telugu.

The important architectural point is that TinyAya does not need to do every job itself. The best systems separate:
- linguistic planning
- speaker/style control
- acoustic detail prediction
- waveform decoding

## What TinyAya Should Be
TinyAya should be the text-conditioned backbone that predicts the speech representation.

The open question is which representation it should predict.

## Representation Families

### Option A: Spark / BiCodec style
Structure:
- one semantic token stream at about `50` tokens per second
- small fixed global token set per utterance for speaker and global style
- external codec decoder reconstructs waveform

Pros:
- shortest path under the current repo
- already partially integrated here
- explicit global token path is useful for speaker/style control
- sequence is simpler than multi-codebook RVQ

Cons:
- semantic-only stream may cap final fidelity versus richer acoustic tokenizers
- style control is global, not frame-structured
- less faithful to the strongest recent streaming token-LM designs

Best use:
- first controllable multilingual baseline
- fast style/speaker experimentation

### Option B: Mimi / Qwen-12Hz / FireRed-style semantic-first multi-codebook codec
Structure:
- about `12.5` frames per second
- one primary semantic/base codebook
- multiple residual acoustic codebooks
- codec decoder directly reconstructs waveform

Pros:
- strongest long-term fit for TinyAya-as-speech-LM
- lower autoregressive frame rate than `50 TPS` semantic codecs
- strong recent tokenizer quality, especially Qwen 12Hz
- streaming-friendly path

Cons:
- current repo is architecturally wrong for this when `Q` is large
- flat serialization makes the backbone predict every residual token serially
- needs grouped or hierarchical prediction

Best use:
- main long-term architecture
- final multilingual expressive system if we are serious about quality and latency

### Option C: X-Codec 2 style single-codebook tokenizer
Structure:
- one codebook
- `50` tokens per second
- large codebook size
- direct LM over one speech token stream

Pros:
- simpler than multi-codebook grouped prediction
- no residual head needed
- multilingual semantic support is strong in the original release

Cons:
- long sequence length
- `16 kHz` in the public release
- less attractive for highest-fidelity expressive generation than stronger 24 kHz tokenizers

Best use:
- strong control baseline
- fallback if grouped multi-codebook work is delayed

### Option D: FACodec / factorized attribute codec
Structure:
- separate content, prosody, timbre, and residual factors
- attribute-aware decoding

Pros:
- best conceptual fit for expressive speech research
- disentanglement is attractive for style transfer and controllability

Cons:
- much more invasive integration
- current repo is not close to supporting this cleanly
- high engineering cost before we even know the best baseline

Best use:
- later research track, not the first architecture we build

## Current Repo Constraint
The current repo is a plain causal LM over a flat token stream.

That is acceptable for:
- Spark-style semantic stream plus global tokens
- low-codebook baselines
- smoke tests and overfit debugging

It is not the right final design for:
- Qwen 12Hz `Q16`
- Mimi `Q8+`
- FireRed-style residual codebooks

## Recommended Final Architecture

### Backbone
TinyAya predicts:
- text-conditioned primary semantic speech stream
- optional style/global control tokens

### Residual head
A smaller grouped predictor predicts residual acoustic codebooks conditioned on:
- TinyAya hidden states
- primary semantic codes
- optional speaker/style embedding

### Decoder
A pretrained codec decoder reconstructs waveform from:
- primary codebook
- residual codebooks
- optional global/style codes if the codec uses them

## Recommended Immediate Architecture Program

### Stage A: Easy baseline
Use Spark/BiCodec-style representation under the current flat stack.

Why:
- shortest path to a credible multilingual baseline
- explicit global tokens help speaker/style work early
- minimal architecture churn

### Stage B: Serious long-term path
Implement grouped prediction for a semantic-first `12.5 Hz` multi-codebook codec.

Preferred target order:
1. Qwen 12Hz tokenizer if integration and licensing stay clean
2. Mimi as the safer fallback
3. FireRed tokenizer as a later comparative track

### Stage C: Expressiveness upgrade
Add style conditioning on top of Stage B:
- reference speaker embedding
- optional style label / natural-language style prompt
- optional global style tokens

## Decision
The repo should stop pretending flat `Q16` is a final architecture.

Research decision:
- baseline architecture: Spark/BiCodec-style semantic + global tokens
- target architecture: TinyAya backbone + grouped residual predictor over a `12.5 Hz` multi-codebook codec
- factorized prosody/timbre codecs remain a later track, not the first bet

## Why This Is The Right Research Bet
- It preserves TinyAya as the core model.
- It gives a short path to a real baseline.
- It aligns the final architecture with where the best open speech token systems have moved by early 2026.
- It separates concerns instead of forcing one flat token loss to carry content, speaker, style, and acoustics all at once.

## References
- [2509.14882v1.pdf](/home/pranav/TINYYAYAy/2509.14882v1.pdf)
- [fish.pdf](/home/pranav/TINYYAYAy/fish.pdf)
- [sparrk.pdf](/home/pranav/TINYYAYAy/sparrk.pdf)
- [Qwen3_TTS.pdf](/home/pranav/TINYYAYAy/Qwen3-TTS/assets/Qwen3_TTS.pdf)
- https://github.com/QwenLM/Qwen3-TTS
- https://github.com/FireRedTeam/FireRedTTS2
- https://github.com/zhenye234/X-Codec-2.0
- https://github.com/lifeiteng/naturalspeech3_facodec
