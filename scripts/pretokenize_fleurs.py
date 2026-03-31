import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from datasets import Audio, Dataset, get_dataset_config_names, load_dataset

from torchtitan.config_manager import JobConfig
from torchtitan.tools.audio_codec import load_audio_codec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize FLEURS with audio codec.")
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
        help="Number of codec quantizers/codebooks to export.",
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
    parser.add_argument(
        "--audio-codec-backend",
        choices=["mimi", "s1_dac", "spark_bicodec", "dualcodec", "qwen_codec"],
        default="mimi",
        help="Audio codec backend used for tokenization.",
    )
    parser.add_argument(
        "--audio-codec-source",
        choices=["official_fish", "hf_pretrained"],
        default="official_fish",
        help="Codec source hint used by the adapter.",
    )
    parser.add_argument(
        "--audio-codec-model-id",
        default="",
        help="Codec HF model id/path override.",
    )
    parser.add_argument(
        "--audio-codec-ckpt-path",
        default="",
        help="Optional local checkpoint path for official codec assets.",
    )
    parser.add_argument(
        "--audio-codec-trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for codec model loading.",
    )
    return parser.parse_args()


def _get_fleurs_config_names() -> list[str]:
    try:
        return get_dataset_config_names("google/fleurs", trust_remote_code=True)
    except TypeError:
        return get_dataset_config_names("google/fleurs")


def _load_fleurs_split(config_name: str, split: str):
    try:
        return load_dataset(
            "google/fleurs",
            config_name,
            split=split,
            trust_remote_code=True,
        )
    except TypeError:
        return load_dataset(
            "google/fleurs",
            config_name,
            split=split,
        )


def resolve_fleurs_configs(languages: Iterable[str]) -> dict[str, str]:
    available = _get_fleurs_config_names()
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
    audio_tokenizer,
    num_quantizers: int,
    codec_backend: str = "mimi",
) -> tuple[list[list[int]], list[int], list[int]]:
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(audio_tokenizer.device)
    padding_mask = inputs.get("padding_mask")
    if padding_mask is None:
        padding_mask = inputs.get("attention_mask")
    if padding_mask is None:
        padding_mask = torch.ones_like(inputs["input_values"], dtype=torch.long)
    with torch.no_grad():
        outputs = audio_tokenizer.encode(
            inputs["input_values"],
            padding_mask,
            num_quantizers=num_quantizers,
        )
    audio_codes = outputs.audio_codes
    if audio_codes.ndim == 2:
        audio_codes = audio_codes.unsqueeze(0)
    if audio_codes.ndim != 3:
        raise RuntimeError(f"Unsupported codec output shape: {tuple(audio_codes.shape)}")
    # (B, Q, T) -> (T, Q)
    codes_tq = audio_codes[0].transpose(0, 1).detach().cpu().tolist()

    semantic_tokens: list[int] = []
    global_tokens: list[int] = []
    if codec_backend.strip().lower() == "spark_bicodec":
        semantic_tokens = audio_codes[0, 0].detach().cpu().to(torch.int64).tolist()
        raw_global = getattr(outputs, "global_codes", None)
        if raw_global is not None:
            if not torch.is_tensor(raw_global):
                raw_global = torch.as_tensor(raw_global)
            if raw_global.ndim == 1:
                raw_global = raw_global.unsqueeze(0)
            if raw_global.ndim == 3:
                if raw_global.shape[1] == 1:
                    raw_global = raw_global[:, 0, :]
                elif raw_global.shape[2] == 1:
                    raw_global = raw_global[:, :, 0]
                else:
                    raw_global = raw_global.reshape(raw_global.shape[0], -1)
            if raw_global.ndim == 2 and raw_global.shape[0] >= 1:
                global_tokens = raw_global[0].detach().cpu().to(torch.int64).tolist()
    return codes_tq, semantic_tokens, global_tokens


def flush_records(records: list[dict], output_file: Path) -> None:
    if not records:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_file))
    records.clear()


def _write_dataset_manifest(output_root: Path, payload: dict) -> None:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload = dict(payload)
    payload["fingerprint_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root = Path(args.output_dir)
    split_root = output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)

    job_cfg = JobConfig()
    job_cfg.model.num_quantizers = int(args.num_quantizers)
    job_cfg.audio_codec.backend = args.audio_codec_backend
    job_cfg.audio_codec.source = args.audio_codec_source
    if args.audio_codec_model_id.strip():
        job_cfg.audio_codec.model_id = args.audio_codec_model_id.strip()
    if args.audio_codec_ckpt_path.strip():
        job_cfg.audio_codec.codec_ckpt_path = args.audio_codec_ckpt_path.strip()
    job_cfg.audio_codec.trust_remote_code = bool(args.audio_codec_trust_remote_code)
    audio_tokenizer, feature_extractor, codec_info = load_audio_codec(job_cfg, device)

    lang_to_config = resolve_fleurs_configs(args.languages)
    print(f"Resolved FLEURS configs: {lang_to_config}")
    print(
        "Audio codec: "
        f"backend={codec_info.backend} source={codec_info.source} "
        f"model_ref={codec_info.model_ref} sr={codec_info.sampling_rate} "
        f"codebook_size={codec_info.codebook_size} "
        f"max_codebooks={codec_info.max_codebooks}"
    )

    manifest: dict[str, object] = {
        "dataset_name": "google/fleurs",
        "split": args.split,
        "languages": list(lang_to_config.keys()),
        "resolved_fleurs_configs": lang_to_config,
        "num_quantizers": int(args.num_quantizers),
        "shard_size": int(args.shard_size),
        "max_samples_per_language": int(args.max_samples_per_language),
        "output_dir": str(output_root),
        "audio_codec": {
            "backend": codec_info.backend,
            "source": codec_info.source,
            "model_ref": codec_info.model_ref,
            "sample_rate": int(codec_info.sampling_rate),
            "codebook_size": int(codec_info.codebook_size),
            "max_codebooks": int(codec_info.max_codebooks),
        },
        "languages_detail": {},
        "num_rows_total": 0,
    }

    for lang, fleurs_cfg in lang_to_config.items():
        print(f"Processing language={lang} config={fleurs_cfg} split={args.split}")
        ds = _load_fleurs_split(fleurs_cfg, args.split)
        ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

        out_dir = split_root / f"lang={lang}"
        out_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        seen = 0
        buffer: list[dict] = []
        written_files: list[str] = []
        for sample in ds:
            if args.max_samples_per_language and seen >= args.max_samples_per_language:
                break

            text = sample.get("transcription") or sample.get("raw_transcription") or ""
            if not text:
                continue

            audio = sample["audio"]["array"]
            try:
                audio_codes, spark_semantic_tokens, spark_global_tokens = encode_audio_codes(
                    audio,
                    feature_extractor,
                    audio_tokenizer,
                    args.num_quantizers,
                    codec_backend=args.audio_codec_backend,
                )
            except Exception as exc:
                print(f"Skipping sample due to codec encode failure: {exc}")
                continue

            record = {
                "id": str(sample.get("id", f"{lang}_{seen}")),
                "lang": lang,
                "text": text,
                "audio_codes": audio_codes,
                "mimi_codes": audio_codes,
                "sample_rate": int(feature_extractor.sampling_rate),
                "duration_sec": float(len(audio) / feature_extractor.sampling_rate),
                "split": args.split,
            }
            if args.audio_codec_backend.strip().lower() == "spark_bicodec":
                record["spark_semantic_tokens"] = [
                    int(x) for x in spark_semantic_tokens
                ]
                record["spark_global_tokens"] = [int(x) for x in spark_global_tokens]
            buffer.append(record)
            seen += 1

            if len(buffer) >= args.shard_size:
                output_file = out_dir / f"part-{shard_idx:05d}.parquet"
                flush_records(buffer, output_file)
                written_files.append(str(output_file.relative_to(output_root)))
                shard_idx += 1

        if buffer:
            output_file = out_dir / f"part-{shard_idx:05d}.parquet"
            flush_records(buffer, output_file)
            written_files.append(str(output_file.relative_to(output_root)))

        print(f"Finished {lang}: wrote {seen} samples to {out_dir}")
        manifest["languages_detail"][lang] = {
            "fleurs_config": fleurs_cfg,
            "num_rows": int(seen),
            "num_shards": int(len(written_files)),
            "shards": written_files,
        }
        manifest["num_rows_total"] = int(manifest["num_rows_total"]) + int(seen)

    _write_dataset_manifest(output_root, manifest)
    print(f"Wrote dataset manifest to {output_root / 'dataset_manifest.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
