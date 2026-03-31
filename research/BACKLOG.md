# TinyAya TTS Research Backlog

## P0
- [ ] freeze multilingual research protocol for English/Hindi/Telugu
- [ ] implement proper multilingual text normalization and per-language WER/CER
- [ ] run codec audit pilots before any full corpus pretokenization
- [ ] standardize experiment cards and run scorecards

## P1
- [ ] benchmark Spark/BiCodec as the easiest controllable baseline
- [ ] benchmark at least one stronger semantic-first codec path
- [ ] identify the best baseline track for multilingual intelligibility
- [ ] quantify Hindi and Telugu failure cases separately from English

## P2
- [ ] design grouped-codebook / residual predictor path for high-codebook codecs
- [ ] implement training targets and generation flow for grouped prediction
- [ ] compare grouped prediction against flat baseline

## P3
- [ ] define the clean read-speech anchor mixture
- [ ] define the natural multispeaker mixture
- [ ] define the expressive/dialogue adaptation mixture
- [ ] add speaker-similarity and prosody scorecards

## P4
- [ ] add speaker/style conditioning experiments
- [ ] add controlled style prompts or style labels
- [ ] test code-switch robustness for English/Hindi/Telugu

## P5
- [ ] long-run multilingual training
- [ ] controlled ablations on codec, loss, and conditioning
- [ ] checkpoint selection and model packaging
