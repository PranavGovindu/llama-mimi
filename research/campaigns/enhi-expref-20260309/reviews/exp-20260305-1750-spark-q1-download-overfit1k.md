# Review Memo: exp-20260305-1750-spark-q1-download-overfit1k

- label: `invalid`
- decision: `rerun`
- campaign_id: `enhi-expref-20260309`
- owner: `pranav`
- created_at_utc: `2026-03-09T15:17:20.698869+00:00`

## Summary
Metadata is intact but no local sample artifacts were found under the run registry path.

## Evidence
- scorecard: `/home/pranav/TINYYAYAy/llama-mimi/experiments/runs/spark_bicodec/exp-20260305-1750-spark-q1-download-overfit1k/scorecard.json`
- verdict_suggestion: `invalid`
- content: `{'cer_constrained': {'count': 0}, 'cer_unconstrained': {'count': 0}, 'constrained_decode_count': 0, 'samples_with_eval': 0, 'unconstrained_decode_count': 0, 'wer_constrained': {'count': 0}, 'wer_unconstrained': {'count': 0}}`
- gate_summary: `{'first_pass_step': None, 'last_gate': {}, 'steps_with_gate': 0}`

## Next Action
Point the scorecard builder at the actual dump_root or sync run artifacts locally before judging the direction.
