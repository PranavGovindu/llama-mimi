# Experiment Log

Auto-generated from `experiments/runs/dualcodec/index.jsonl`.

## 2026-03-07 - exp-dualcodec-12hz-q8-download-overfit1k-s4096
- codec: dualcodec
- status: running
- phase: overfit_dualcodec_12hz_q8_custom
- mode: modal
- config: codecs/dualcodec/configs/tinyaya_dualcodec_12hz_q8_download_overfit_1sample.toml
- modal_path: dualcodec/overfit_download_12hz_q8
- wandb_run_id: 68nkqsg2
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/68nkqsg2
- modal_app_id: ap-Zk0d4iUMcqA36Yb9VhIFcZ
- notes: Detached retry with seq_len=4096 to avoid OOM; run in progress.

## 2026-03-07 - exp-dualcodec-12hz-q8-download-overfit1k
- codec: dualcodec
- status: failed
- phase: overfit_dualcodec_12hz_q8_custom
- mode: modal
- config: codecs/dualcodec/configs/tinyaya_dualcodec_12hz_q8_download_overfit_1sample.toml
- modal_path: dualcodec/overfit_download_12hz_q8
- wandb_run_id: 8ckkglry
- wandb_url: https://wandb.ai/violaze25-rumik/tinyaya-mimi-tts/runs/8ckkglry
- modal_app_id: ap-9ECbnvUXbH9xGqpqzBWRFF
- notes: Failed at step 1 with CUDA OOM on H200 at seq_len=8192.
