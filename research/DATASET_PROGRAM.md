# TinyAya TTS Dataset Program

## Principle
We do not choose one dataset. We build a dataset stack.

Different datasets solve different problems:
- read-speech alignment
- multilingual coverage
- natural speaking style
- emotion and dialogue
- speaker diversity

## Dataset Roles

### FLEURS
Role:
- smoke tests
- tokenizer sanity
- language/script coverage checks
- tiny multilingual pilot evaluation

Why not final training:
- not a premium TTS corpus
- too small and too benchmark-oriented
- useful for controlled sanity checks, not for the best final model

### IndicTTS
Role:
- clean paired read-speech anchor for Hindi and Telugu
- pronunciation and alignment stability

Strength:
- studio-style, clean, reliable

Weakness:
- limited speaker diversity
- limited expressiveness

### Rasa
Role:
- expressive Indian-language supervision
- emotions, commands, narration, conversational styles

Strength:
- directly relevant to expressive Indic TTS

Weakness:
- speaker-language coverage is structured but not unlimited
- should be mixed carefully with cleaner anchor data

### IndicVoices-R
Role:
- large-scale natural Indian speech
- many speakers
- demographic diversity
- more realistic speaking patterns

Strength:
- best open Indian multilingual data source for scale and speaker diversity

Weakness:
- more variable recording conditions than studio speech
- needs good filtering and curriculum

### English corpora
Role:
- keep English quality competitive instead of treating it as an afterthought

Recommended English sources:
- clean read anchor: LibriTTS-R or equivalent clean English paired corpus
- expressive scale: selected Emilia-English or similarly filtered in-the-wild English speech

Why:
- the target model is multilingual, not India-only
- English needs enough clean and expressive coverage to avoid becoming the weakest language

## Recommended Dataset Stack

### Phase 0: Research pilots
Per codec, use only tiny subsets:
- `100` to `500` utterances per language
- English, Hindi, Telugu

Purpose:
- validate tokenization
- validate decoding
- measure token rates and failure modes
- inspect codebook usage

No full-dataset pretokenization before this passes.

### Phase 1: Clean multilingual content baseline
Use:
- English clean read corpus
- IndicTTS Hindi
- IndicTTS Telugu
- optional tiny FLEURS slice for script and formatting sanity

Goal:
- maximize intelligibility and alignment stability
- prove multilingual content learning before style work

### Phase 2: Natural multispeaker expansion
Add:
- IndicVoices-R Hindi
- IndicVoices-R Telugu
- selected English multispeaker natural speech

Goal:
- improve speaker diversity
- reduce overfitting to studio cadence

### Phase 3: Expressive adaptation
Add:
- Rasa Hindi
- Rasa Telugu
- expressive English subset with style metadata or synthetic style descriptions

Goal:
- emotion
- prosody
- dialogue delivery
- command and conversational style

## Curriculum
Do not mix all datasets with equal probability from day one.

Recommended curriculum:
1. clean read anchor
2. natural multispeaker expansion
3. expressive/style adaptation

Why:
- content accuracy and pronunciation should stabilize before style diversity is increased
- otherwise the model may sound expressive but say the wrong words

## Required Metadata Normalization
Every dataset row should eventually carry:
- language
- script
- speaker id if available
- gender if available
- style/domain label if available
- emotion label if available
- transcript
- normalized transcript
- source dataset
- audio duration

This is necessary for:
- balanced sampling
- language-specific evaluation
- speaker/style ablations

## Evaluation Splits
Create fixed, versioned evaluation packs.

### Pack A: content accuracy
- `100` utterances per language
- English, Hindi, Telugu
- balanced by sentence length

### Pack B: speaker similarity
- prompt + target text pairs
- at least `25` speakers per language if available

### Pack C: expressiveness
- emotion/style prompts
- Rasa-backed held-out samples for Hindi and Telugu
- matched English expressive held-out set

### Pack D: code-switch stress
- English-Hindi
- English-Telugu
- mixed-script and native-script prompts

## Data Decisions
- FLEURS is not the final dataset.
- IndicTTS is the clean anchor, not the whole solution.
- IndicVoices-R is the scale engine for Indic languages.
- Rasa is the style engine for Indic languages.
- English needs its own strong clean and expressive sources.

## References
- https://huggingface.co/datasets/ai4bharat/indicvoices_r
- https://huggingface.co/datasets/ai4bharat/Rasa
- https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
- https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi
- https://github.com/AI4Bharat/IndicF5
- https://huggingface.co/datasets/amphion/Emilia-Dataset
