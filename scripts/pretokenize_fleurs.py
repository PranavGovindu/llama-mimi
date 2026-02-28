import argparse
import os
from pathlib import Path
from typing import Iterable

import torch
from datasets import Audio, Dataset, get_dataset_config_names, load_dataset
from transformers import AutoFeatureExtractor, MimiModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize FLEURS with Mimi.")
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Language codes (e.g. en hi fr).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="FLEURS split to process.",
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=1,
        help="Number of Mimi quantizers to export (1 or 8).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory (e.g. /vol/data/fleurs_pretok_q1).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Records per parquet shard.",
    )
    parser.add_argument(
        "--max-samples-per-language",
        type=int,
        default=0,
        help="Optional cap per language (0 = no cap).",
    )
    return parser.parse_args()


def resolve_fleurs_configs(languages: Iterable[str]) -> dict[str, str]:
    available = get_dataset_config_names("google/fleurs", trust_remote_code=True)
    resolved: dict[str, str] = {}
    for raw_lang in languages:
        lang = raw_lang.strip().lower()
        if lang in available:
            resolved[lang] = lang
            continue
        candidates = [cfg for cfg in available if cfg.startswith(f"{lang}_")]
        if not candidates:
            raise ValueError(f"Could not resolve FLEURS config for language '{lang}'.")
        resolved[lang] = sorted(candidates)[0]
    return resolved


def encode_audio_codes(
    audio_array,
    feature_extractor,
    mimi_model: MimiModel,
    num_quantizers: int,
) -> list[list[int]]:
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(mimi_model.device)
    with torch.no_grad():
        outputs = mimi_model.encode(
            inputs["input_values"],
            inputs["padding_mask"],
            num_quantizers=num_quantizers,
        )
    # (B, Q, T) -> (T, Q)
    codes_tq = outputs.audio_codes[0].transpose(0, 1).cpu().tolist()
    return codes_tq


def flush_records(records: list[dict], output_file: Path) -> None:
    if not records:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_file))
    records.clear()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root = Path(args.output_dir)
    split_root = output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)

    mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device)
    mimi_model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    lang_to_config = resolve_fleurs_configs(args.languages)
    print(f"Resolved FLEURS configs: {lang_to_config}")

    for lang, fleurs_cfg in lang_to_config.items():
        print(f"Processing language={lang} config={fleurs_cfg} split={args.split}")
        ds = load_dataset(
            "google/fleurs",
            fleurs_cfg,
            split=args.split,
            trust_remote_code=True,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

        out_dir = split_root / f"lang={lang}"
        out_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        seen = 0
        buffer: list[dict] = []
        for sample in ds:
            if args.max_samples_per_language and seen >= args.max_samples_per_language:
                break

            text = sample.get("transcription") or sample.get("raw_transcription") or ""
            if not text:
                continue

            audio = sample["audio"]["array"]
            try:
                mimi_codes = encode_audio_codes(
                    audio,
                    feature_extractor,
                    mimi_model,
                    args.num_quantizers,
                )
            except Exception as exc:
                print(f"Skipping sample due to Mimi encode failure: {exc}")
                continue

            buffer.append(
                {
                    "id": str(sample.get("id", f"{lang}_{seen}")),
                    "lang": lang,
                    "text": text,
                    "mimi_codes": mimi_codes,
                    "sample_rate": int(feature_extractor.sampling_rate),
                    "duration_sec": float(len(audio) / feature_extractor.sampling_rate),
                    "split": args.split,
                }
            )
            seen += 1

            if len(buffer) >= args.shard_size:
                flush_records(buffer, out_dir / f"part-{shard_idx:05d}.parquet")
                shard_idx += 1

        if buffer:
            flush_records(buffer, out_dir / f"part-{shard_idx:05d}.parquet")

        print(f"Finished {lang}: wrote {seen} samples to {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
