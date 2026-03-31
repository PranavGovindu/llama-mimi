# W&B Core Panel Contract

These keys are the minimum dashboard set to monitor every run.

## Train/Val
- `core/train_loss`
- `core/train_loss_ema`
- `core/grad_norm_ema`
- `core/val_loss`

## Audio + Text
- `core/target_audio_0`
- `core/generated_audio_unconstrained_0`
- `core/generated_audio_constrained_0`
- `core/target_utterance_0`
- `core/generated_utterance_unconstrained_0`
- `core/generated_utterance_constrained_0`

## ASR Objective Metrics
- `core/generated_wer_unconstrained_0`
- `core/generated_cer_unconstrained_0`
- `core/generated_wer_constrained_0`
- `core/generated_cer_constrained_0`

## Codebook Summary
- `core/target_codebook_frames_0`
- `core/target_codebook_coverage_total_0`
- `samples/target_codebook_qstats_0`
- `core/generated_unconstrained_codebook_frames_0`
- `core/generated_unconstrained_codebook_coverage_total_0`
- `samples/generated_unconstrained_codebook_heatmap_0`
- `samples/generated_unconstrained_codebook_qstats_0`
- `core/generated_constrained_codebook_frames_0`
- `core/generated_constrained_codebook_coverage_total_0`
- `samples/generated_constrained_codebook_heatmap_0`
- `samples/generated_constrained_codebook_qstats_0`

## Gate Status
- `core/gate_unconstrained_audio_seen`
- `core/gate_overall_pass`
- `core/gate_coverage_q_min`
- `core/gate_coverage_q_abs_diff_max`
- `core/overfit_gate_passed_ever`
- `core/gate_consecutive_passes`
