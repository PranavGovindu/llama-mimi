<div align="center">

# Llama-Mimi
#### Autoregressive Speech Language Modeling with Interleaved Semantic and Acoustic Tokens
| [📃Paper](https://arxiv.org/abs/2509.14882) | [🤗Models](https://huggingface.co/llm-jp/Llama-Mimi-1.3B) | [🗣️Online Demo](https://speed1313.github.io/llama-mimi/) |

<img src="assets/llama-mimi.png" width="60%"/>

</div>



## Introduction
Llama-Mimi is a speech language model that uses a unified tokenizer (Mimi) and a single Transformer decoder (Llama) to jointly model sequences of interleaved semantic and acoustic tokens.
Trained on ~240k hours of English audio, Llama-Mimi achieves state-of-the-art performance in acoustic consistency on [SALMon](https://arxiv.org/abs/2409.07437) and effectively preserves speaker identity.

Visit our [demo site](https://speed1313.github.io/llama-mimi/) to hear generated speech samples.


## Repository Overview
This repository lets you:
- Run inference with our pretrained models
- Pre-train Llama-Mimi on [The People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
- Evaluate the model on multiple benchmarks

## Setup


Install dependencies using uv:
```bash
uv sync
```

## TinyAya + Mimi TTS (Modal-Ready)

This repo now includes a TinyAya adaptation path with two training presets:
- `config/tinyaya_q1_fleurs.toml` for a fast multilingual baseline (`Q=1`)
- `config/tinyaya_q8_fleurs.toml` for higher-fidelity interleaved training (`Q=8`)

### 1) Pre-tokenize FLEURS with Mimi
```bash
uv run python scripts/pretokenize_fleurs.py \
  --languages en hi es fr de ar sw ta bn zh \
  --split train \
  --num-quantizers 1 \
  --output-dir /vol/data/fleurs_pretok_q1
```

### 2) Train TinyAya (Q=1)
```bash
torchrun --nproc_per_node=1 -m torchtitan.train \
  --job.config_file config/tinyaya_q1_fleurs.toml
```

### 3) Train TinyAya (Q=8)
```bash
torchrun --nproc_per_node=1 -m torchtitan.train \
  --job.config_file config/tinyaya_q8_fleurs.toml
```

### 4) Text-to-Speech Inference
```bash
uv run python inference_tts.py \
  --model-id <checkpoint_or_hf_model> \
  --text "hello world" \
  --lang en \
  --num-quantizers 1 \
  --output-file output_tts.wav
```

### 5) Modal Jobs
```bash
modal run modal/app.py::pretokenize_fleurs --split train --quantizers 1
modal run modal/app.py::train --path q1
```

### 6) One-Sample Overfit Smoke Test (recommended first)
This mode repeats one training sample to verify the full path works end-to-end.
It also logs generated-vs-target audio clips to W&B every few steps.

Local:
```bash
export WANDB_API_KEY="<your_wandb_key>"
export WANDB_PROJECT="tinyaya-mimi-tts"
./scripts/run_overfit_one_sample.sh
```

Modal:
```bash
# Ensure a Modal secret named "wandb" contains WANDB_API_KEY
modal run modal/app.py::train --path overfit1
# Strict gating run (fails if unconstrained generated audio never appears):
modal run modal/app.py::train --path overfit_strict
```

Overfit config file:
`config/tinyaya_q1_fleurs_overfit_1sample.toml`
Strict config file:
`config/tinyaya_q1_fleurs_overfit_1sample_strict.toml`

Logged artifacts:
- `loss_metrics/global_avg_loss_ema`
- `grad_norm_ema`
- `samples/generated_audio_unconstrained_0`
- `samples/generated_audio_constrained_0`
- `samples/target_audio_0`
- `samples/generated_text_unconstrained_0`
- `core/*` (compact dashboard contract)
- `gates/*` (objective overfit gate signals)

### 7) Experiment Tracking (recommended)
Protocol docs:
- `experiments/PROTOCOL.md`
- `experiments/WANDB_CORE_PANELS.md`

Run registry + generated log:
- `experiments/runs/index.jsonl` (source of truth)
- `experiments/EXPERIMENT_LOG.md` (auto-rendered view)

Launcher/finalizer scripts:
```bash
# launch (local)
python scripts/exp/launch.py \
  --mode local \
  --config config/tinyaya_q1_fleurs_overfit_1sample_strict.toml \
  --phase overfit_q1 \
  --question "Can model overfit one sample?" \
  --hypothesis "Strict decoding + gates will pass." \
  --owner pranav

# launch (modal)
python scripts/exp/launch.py --mode modal --modal-path overfit_strict --phase overfit_q1

# finalize + append registry
python scripts/exp/finalize.py --experiment-id <exp-id> --status completed

# regenerate markdown report
python scripts/exp/render_log.py
```

This keeps W&B runs, Modal app IDs, immutable run snapshots, and configs reproducible.

## Generate Speech

Generate audio continuations from a given audio prompt using our pretrained model (Llama-Mimi-1.3B):
```bash
uv run python inference.py
```

[▶️ Listen to samples on our demo site](https://speed1313.github.io/llama-mimi)

## Pre-train Llama-Mimi on The People's Speech

To pre-train Llama-Mimi on [The People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) (30k hours), first download the dataset locally:
```bash
uv run huggingface-cli download  MLCommons/peoples_speech  --repo-type dataset --local-dir data/peoples_speech
```

Then launch training with:
```bash
torchrun --nproc_per_node=8 --local-ranks-filter 0 \
      --role rank --tee 3 -m torchtitan.train \
      --job.config_file config/llama3_2_1b_peoples_speech.toml
```
This configuration trains Llama-Mimi-1.3B for 5,000 steps with a global batch size of 1,024 on 8 GPUs, taking about 8 hours.
Training progress can be monitored with Weights & Biases (W&B).

<div align="center">
<img src="assets/log_validation.png" width="40%"/>
</div>

To use a custom dataset, update the configuration in `torchtitan/datasets/hf_datasets.py`. We recommend downloading multiple large datasets, shuffling them, and then using `load_dataset()` with local files.

After training, convert dcp checkpoint to HuggingFace format to use the model with `transformers` library:

```bash
uv run python scripts/convert_dcp_to_hf.py
```


## Evaluation
Evaluate models on [SALMon](https://github.com/slp-rl/salmon), [sLM21](https://arxiv.org/abs/2104.14700) (sWUGGY and sBLIMP), and [sStoryCloze](https://github.com/slp-rl/SpokenStoryCloze) tasks.

SALMon:
```bash
uv run python eval/salmon.py --model_name llm-jp/Llama-Mimi-1.3B
```

sStoryCloze:
```bash
uv run python eval/sStoryCloze.py --model_name llm-jp/Llama-Mimi-1.3B
```

sLM21:
```bash
uv run python eval/sLM21.py --model_name llm-jp/Llama-Mimi-1.3B
```



## Acknowledge

- Our training code is built on top of [TorchTitan](https://github.com/pytorch/torchtitan).

- Our model employs [Llama 3](https://arxiv.org/abs/2407.21783) as the base language model, and [Mimi](https://arxiv.org/abs/2410.00037) as the audio tokenizer.


## Citation
Star us on GitHub if you find this repository useful! ⭐

If you find this work interesting, please cite our paper:
```
@misc{sugiura2025llamamimispeechlanguagemodels,
      title={Llama-Mimi: Speech Language Models with Interleaved Semantic and Acoustic Tokens},
      author={Issa Sugiura and Shuhei Kurita and Yusuke Oda and Ryuichiro Higashinaka},
      year={2025},
      eprint={2509.14882},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.14882},
}
```
