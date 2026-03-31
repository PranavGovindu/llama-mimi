# TinyAya TTS Decision Register

## 2026-03-09

### Decision 1
FLEURS is a smoke and evaluation dataset, not the final training corpus.

Reason:
- good for controlled multilingual sanity checks
- not enough quality or expressiveness to carry the final system

### Decision 2
The current flat token LM remains acceptable only for low-complexity baselines.

Reason:
- it scales poorly with many codebooks
- it is not a credible final design for `12.5 Hz` `Q8+` or `Q16` tokenizers

### Decision 3
The architecture program splits into two tracks.

Track A:
- Spark/BiCodec-style baseline for quick, controllable multilingual TTS

Track B:
- grouped residual predictor over a `12.5 Hz` semantic-first multi-codebook codec as the final target path

### Decision 4
Loss is not the selection metric.

Reason:
- token CE trains the model
- scorecards decide the model

### Decision 5
Hindi and Telugu will not be treated as secondary languages.

Reason:
- all major experiments should report per-language results for English, Hindi, and Telugu
- regressions in Hindi or Telugu block promotion even if English improves
