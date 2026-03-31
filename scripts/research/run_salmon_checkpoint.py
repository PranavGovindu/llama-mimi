#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Audio, get_dataset_config_names, load_dataset
import torch
import torch.distributed.checkpoint as dcp
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.datasets.hf_datasets import audio_array_to_text
from torchtitan.train import expand_tokenizer_with_unit_tokens


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SALMon directly from a local DCP checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-name", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--num-quantizers", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-tasks", type=int, default=0)
    return parser.parse_args()


def _resolve_device(raw: str) -> str:
    if raw.strip():
        return raw.strip()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(
    *,
    checkpoint_dir: str,
    model_name: str,
    num_quantizers: int,
    codebook_size: int,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, MimiModel, AutoFeatureExtractor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer = expand_tokenizer_with_unit_tokens(
        tokenizer,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
    )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    wrapped_model = ModelWrapper(model)
    dcp.load(wrapped_model.state_dict(), checkpoint_id=str(Path(checkpoint_dir).resolve()))
    model.config.num_quantizers = int(num_quantizers)
    model.to(device).eval()

    audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    return model, tokenizer, audio_tokenizer, feature_extractor


def _compute_loss(
    *,
    audio,
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
    tokenizer,
    model,
    device: str,
) -> torch.Tensor:
    text = audio_array_to_text(
        audio,
        audio_tokenizer,
        feature_extractor,
        num_quantizers,
    )
    inputs = tokenizer(text, return_tensors="pt")
    labels = inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    inputs = inputs.to(device)
    outputs = model(input_ids=inputs.input_ids, labels=labels)
    return outputs.loss


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    model, tokenizer, audio_tokenizer, feature_extractor = _build_model(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        num_quantizers=int(args.num_quantizers),
        codebook_size=int(args.codebook_size),
        device=device,
    )

    tasks = [task for task in get_dataset_config_names("slprl/SALMon") if not task.startswith("all_")]
    if args.max_tasks > 0:
        tasks = tasks[: int(args.max_tasks)]

    per_task: dict[str, float] = {}
    sample_count_by_task: dict[str, int] = {}

    for task in tasks:
        ds = load_dataset("slprl/SALMon", task, split="train")
        ds = ds.cast_column(
            "negative_audio",
            Audio(sampling_rate=feature_extractor.sampling_rate),
        )
        ds = ds.cast_column(
            "positive_audio",
            Audio(sampling_rate=feature_extractor.sampling_rate),
        )

        total_correct = 0
        total_samples = 0
        for example in tqdm(ds, desc=f"SALMon {task}", leave=False):
            negative_audio = example["negative_audio"]["array"]
            positive_audio = example["positive_audio"]["array"]
            with torch.no_grad():
                neg_loss = _compute_loss(
                    audio=negative_audio,
                    audio_tokenizer=audio_tokenizer,
                    feature_extractor=feature_extractor,
                    num_quantizers=int(args.num_quantizers),
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )
                pos_loss = _compute_loss(
                    audio=positive_audio,
                    audio_tokenizer=audio_tokenizer,
                    feature_extractor=feature_extractor,
                    num_quantizers=int(args.num_quantizers),
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )
            total_correct += int((neg_loss > pos_loss).item())
            total_samples += 1

        if total_samples > 0:
            per_task[task] = float(total_correct / total_samples)
            sample_count_by_task[task] = int(total_samples)

    salmon_mean = float(sum(per_task.values()) / max(len(per_task), 1)) if per_task else None
    payload = {
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "model_name": args.model_name,
        "num_quantizers": int(args.num_quantizers),
        "salmon_mean": salmon_mean,
        "salmon_by_task": per_task,
        "sample_count_by_task": sample_count_by_task,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
