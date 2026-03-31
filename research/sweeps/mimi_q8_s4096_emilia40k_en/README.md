# Mimi Q8 / 4096 Emilia-40k EN Sweep

This campaign fixes the backbone to `TinyAya + frozen Mimi`, `Q=8`, `seq_len=4096`, English only, with one immutable `40k`-utterance Emilia-English subset:

- `38k` train
- `1k` validation
- `1k` final holdout test

Base config: [tinyaya_mimi_q8_s4096_emilia40k_en.toml](/home/pranav/TINYYAYAy/llama-mimi/codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en.toml)

## Dataset Build

Plan or run the fixed subset build:

```bash
cd /home/pranav/TINYYAYAy/llama-mimi
python scripts/exp/launch_mimi_emilia40k_q8_ablation.py pretokenize
python scripts/exp/launch_mimi_emilia40k_q8_ablation.py pretokenize --execute
```

Output dataset root defaults to `/vol/data/emilia_en40k_mimi_q8`.

Publish the same frozen dataset to Hugging Face from Modal:

```bash
cd /home/pranav/TINYYAYAy/llama-mimi
modal run modal/app.py --mode pretokenize_emilia_to_hf --quantizers 8 --output-dir /vol/data/emilia_en40k_mimi_q8 --hf-dataset-repo emilia-mimi-40k
```

This creates or updates a dataset repo named `emilia-mimi-40k` under the namespace inferred from the active HF token. The upload is private by default.

## 2k Screen

Write the screen plan and print all launch commands:

```bash
cd /home/pranav/TINYYAYAy/llama-mimi
python scripts/exp/launch_mimi_emilia40k_q8_ablation.py screen2k
```

Launch the full screen on Modal:

```bash
cd /home/pranav/TINYYAYAy/llama-mimi
python scripts/exp/launch_mimi_emilia40k_q8_ablation.py screen2k --execute
```

Artifacts:

- plan TSV: `research/sweeps/mimi_q8_s4096_emilia40k_en/screen2k_plan.tsv`
- per-run scorecards: `experiments/runs/mimi/<experiment_id>/scorecard.json`
- leaderboard TSV: `experiments/runs/mimi/results.tsv`

## 8k Finalists

After the `2k` screen, rerun the shortlisted variants with more steps and more seeds:

```bash
cd /home/pranav/TINYYAYAy/llama-mimi
python scripts/exp/launch_mimi_emilia40k_q8_ablation.py finalists8k --variants anchor,lr_1e-4,batch_64
python scripts/exp/launch_mimi_emilia40k_q8_ablation.py finalists8k --variants anchor,lr_1e-4,batch_64 --execute
```

## Ranking Order

`results.tsv` ranks rows within each `(campaign_id, stage, step, eval_pack)` by:

1. `wer_mean`
2. `cer_mean`
3. `malformed_decode_rate`
4. `speaker_similarity_mean`
5. `dnsmos_p808_mean`
6. `utmos_mean`
7. `mel_l1_mean`
8. `coverage_q_abs_diff_max_mean`

Preview samples in W&B are qualitative only. Ranked decisions come from the full validation pack plus the generated TSV and scorecards.
