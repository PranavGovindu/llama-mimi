import argparse
import hashlib
import io
import json
import math
import os
import heapq
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Audio, Dataset, load_dataset
from transformers import AutoTokenizer

from torchtitan.config_manager import JobConfig
from torchtitan.datasets.hf_datasets import _build_audio_only_prompt, process_pretokenized_tts
from torchtitan.tools.audio_codec import load_audio_codec


MIMI_ENCODE_DOWNSAMPLE = 1920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize Emilia-English into model-ready TinyAya + Mimi tensors."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="amphion/Emilia-Dataset")
    parser.add_argument("--data-files", default="Emilia/EN/*.tar")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--num-quantizers", type=int, default=8)
    parser.add_argument("--tokenizer-name", default="CohereLabs/tiny-aya-fire")
    parser.add_argument(
        "--export-format",
        choices=["codec_only", "model_ready"],
        default="codec_only",
    )
    parser.add_argument(
        "--emit-static-references",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write ref_* fields directly into exported rows. Disabled by default because "
            "training now samples same-speaker references dynamically."
        ),
    )
    parser.add_argument("--log-prefix", default="")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--reference-seq-len", type=int, default=1024)
    parser.add_argument(
        "--mask-text-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--language-tokens", action="store_true", default=False)
    parser.add_argument("--keep-audio-codes", action="store_true")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument(
        "--split-strategy",
        choices=["auto", "train_only", "subset", "exact_hash"],
        default="auto",
    )
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--batch-max-clips", type=int, default=16)
    parser.add_argument("--batch-max-audio-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--audio-codec-backend",
        choices=["mimi", "s1_dac", "spark_bicodec", "dualcodec", "qwen_codec"],
        default="mimi",
    )
    parser.add_argument(
        "--audio-codec-source",
        choices=["official_fish", "hf_pretrained"],
        default="hf_pretrained",
    )
    parser.add_argument("--audio-codec-model-id", default="kyutai/mimi")
    parser.add_argument("--audio-codec-ckpt-path", default="")
    parser.add_argument("--audio-codec-trust-remote-code", action="store_true")
    return parser.parse_args()


def _log(args: argparse.Namespace, stage: str, message: str, **fields: Any) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    prefix = getattr(args, "log_prefix", "").strip()
    rendered_prefix = f"[{prefix}]" if prefix else ""
    rendered_fields = ""
    if fields:
        rendered_fields = " " + " ".join(
            f"{key}={fields[key]}" for key in sorted(fields)
        )
    print(
        f"{timestamp} {rendered_prefix}[{stage}] {message}{rendered_fields}",
        flush=True,
    )


@dataclass
class SourceRecord:
    sample_id: str
    text: str
    lang: str
    speaker_id: str
    duration_sec: float
    audio_bytes: bytes
    audio_path: str
    source_url: str


def _canonical_data_files(raw: str) -> str | list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return raw.strip()
    if len(parts) == 1:
        return parts[0]
    return parts


def _resolve_hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()


def _load_source(
    dataset_id: str,
    data_files: str,
    source_split: str,
    *,
    decode_audio: bool,
):
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "data_files": _canonical_data_files(data_files),
        "split": source_split,
        "streaming": True,
    }
    token = _resolve_hf_token()
    if token:
        kwargs["token"] = token
    ds = load_dataset(**kwargs)
    if decode_audio:
        ds = ds.cast_column("mp3", Audio(sampling_rate=24_000))
    else:
        ds = ds.cast_column("mp3", Audio(decode=False))
    return ds


def _hash_u64(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _score_to_unit_interval(score: int) -> float:
    return score / float(2**64 - 1)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_json_record(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_row(row: dict[str, Any]) -> SourceRecord | None:
    metadata = _coerce_json_record(row.get("json"))
    audio = row.get("mp3")
    if not isinstance(audio, dict):
        return None
    audio_bytes = audio.get("bytes")
    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
        return None
    sample_id = _safe_text(metadata.get("id") or row.get("id") or row.get("__key__"))
    text = _safe_text(metadata.get("text") or row.get("text"))
    if not sample_id or not text:
        return None
    lang = _safe_text(metadata.get("language") or row.get("language") or "en") or "en"
    speaker_id = _safe_text(metadata.get("speaker") or row.get("speaker"))
    duration = metadata.get("duration", row.get("duration"))
    try:
        duration_sec = float(duration)
    except (TypeError, ValueError):
        duration_sec = 0.0
    source_url = _safe_text(row.get("__url__"))
    audio_path = _safe_text(audio.get("path") or metadata.get("wav") or row.get("wav"))
    return SourceRecord(
        sample_id=sample_id,
        text=text,
        lang=lang,
        speaker_id=speaker_id,
        duration_sec=duration_sec,
        audio_bytes=bytes(audio_bytes),
        audio_path=audio_path,
        source_url=source_url,
    )


def _build_tokenizer(
    tokenizer_name: str,
    *,
    codebook_size: int,
    num_quantizers: int,
    language_tokens: bool,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    new_tokens = [f"<{x}_{q}>" for x in range(codebook_size) for q in range(num_quantizers)]
    tokenizer.add_tokens([tok for tok in new_tokens if tok not in tokenizer.get_vocab()])
    special_tokens = ["<audio>", "</audio>"]
    if language_tokens:
        special_tokens.append("<lang_en>")
    tokenizer.add_tokens([tok for tok in special_tokens if tok not in tokenizer.get_vocab()])
    return tokenizer


def _build_labels(
    input_ids: list[int],
    attention_mask: list[int],
    *,
    audio_start_id: int | None,
    audio_end_id: int | None,
    mask_text_loss: bool,
) -> list[int]:
    labels = list(input_ids)
    if not mask_text_loss:
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100
        return labels
    if audio_start_id is None or audio_end_id is None:
        return [-100] * len(input_ids)
    try:
        audio_start_idx = input_ids.index(audio_start_id)
        audio_end_idx = input_ids.index(audio_end_id)
    except ValueError:
        return [-100] * len(input_ids)
    for i, mask in enumerate(attention_mask):
        if mask == 0 or i <= audio_start_idx or i > audio_end_idx:
            labels[i] = -100
    return labels


def _decode_waveform(audio_bytes: bytes, target_sr: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform.squeeze(0).to(torch.float32).contiguous()


def _encode_batch_mimi(
    batch_waveforms: list[torch.Tensor],
    feature_extractor,
    audio_tokenizer,
    num_quantizers: int,
    device: str,
) -> list[list[list[int]]]:
    valid_lengths = [int(waveform.numel()) for waveform in batch_waveforms]
    extracted_values: list[torch.Tensor] = []
    extracted_masks: list[torch.Tensor] = []
    max_length = 0

    # Extract each waveform independently, then pad manually. This follows the same
    # Mimi feature-extractor contract as the runtime path while avoiding failures from
    # variable-length batch collation inside the HF processor on remote workers.
    for waveform in batch_waveforms:
        single = feature_extractor(
            raw_audio=waveform.detach().cpu().numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        input_values = single["input_values"]
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)
        if input_values.ndim != 3:
            raise RuntimeError(
                f"Unsupported Mimi feature shape: {tuple(input_values.shape)}"
            )
        padding_mask = single.get("padding_mask")
        if padding_mask is None:
            padding_mask = single.get("attention_mask")
        if padding_mask is None:
            padding_mask = torch.ones(
                input_values.shape[0],
                input_values.shape[-1],
                dtype=torch.long,
            )
        input_values = input_values[0]
        padding_mask = padding_mask[0].to(dtype=torch.long)
        extracted_values.append(input_values)
        extracted_masks.append(padding_mask)
        max_length = max(max_length, int(input_values.shape[-1]))

    padded_values: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for input_values, padding_mask in zip(extracted_values, extracted_masks, strict=True):
        pad_width = max_length - int(input_values.shape[-1])
        if pad_width > 0:
            input_values = F.pad(input_values, (0, pad_width))
            padding_mask = F.pad(padding_mask, (0, pad_width))
        padded_values.append(input_values)
        padded_masks.append(padding_mask)

    inputs = {
        "input_values": torch.stack(padded_values, dim=0).to(device),
        "padding_mask": torch.stack(padded_masks, dim=0).to(device),
    }
    padding_mask = inputs.get("padding_mask")
    if padding_mask is None:
        padding_mask = inputs.get("attention_mask")
    if padding_mask is None:
        padding_mask = torch.ones(
            inputs["input_values"].shape[0],
            inputs["input_values"].shape[-1],
            dtype=torch.long,
            device=device,
        )
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
        raise RuntimeError(f"Unsupported Mimi output shape: {tuple(audio_codes.shape)}")
    max_frames = int(audio_codes.shape[-1])
    rows: list[list[list[int]]] = []
    for idx, num_samples in enumerate(valid_lengths):
        valid_frames = max(1, math.ceil(num_samples / MIMI_ENCODE_DOWNSAMPLE))
        valid_frames = min(valid_frames, max_frames)
        rows.append(audio_codes[idx, :, :valid_frames].transpose(0, 1).cpu().tolist())
    return rows


def _encode_sample_fallback(
    waveform: torch.Tensor,
    feature_extractor,
    audio_tokenizer,
    num_quantizers: int,
) -> list[list[int]]:
    inputs = feature_extractor(
        raw_audio=waveform,
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
    return audio_codes[0].transpose(0, 1).cpu().tolist()


def _flush_records(records: list[dict[str, Any]], output_file: Path) -> int:
    if not records:
        return 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_file))
    count = len(records)
    records.clear()
    return count


def _write_manifest(output_root: Path, payload: dict[str, Any]) -> None:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload = dict(payload)
    payload["fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _select_exact_holdout_ids(
    args: argparse.Namespace,
) -> tuple[set[str], set[str], dict[str, int]]:
    total_holdout = max(0, int(args.validation_count)) + max(0, int(args.test_count))
    if total_holdout == 0:
        return set(), set(), {"eligible": 0, "scanned": 0}
    _log(
        args,
        "holdout-scan",
        "start",
        validation_count=int(args.validation_count),
        test_count=int(args.test_count),
    )
    ds = _load_source(args.dataset_id, args.data_files, args.source_split, decode_audio=False)
    heap: list[tuple[int, str]] = []
    scanned = 0
    eligible = 0
    for row in ds:
        if args.max_samples and scanned >= args.max_samples:
            break
        scanned += 1
        record = _normalize_row(row)
        if record is None:
            continue
        if record.duration_sec < args.min_seconds or record.duration_sec > args.max_seconds:
            continue
        eligible += 1
        score = _hash_u64(record.sample_id, args.seed)
        item = (-score, record.sample_id)
        if len(heap) < total_holdout:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
        if scanned % 10_000 == 0:
            _log(
                args,
                "holdout-scan",
                "progress",
                scanned=scanned,
                eligible=eligible,
                kept=len(heap),
            )
    selected = sorted([(-neg_score, sample_id) for neg_score, sample_id in heap], key=lambda x: (x[0], x[1]))
    val_count = min(int(args.validation_count), len(selected))
    test_count = min(int(args.test_count), max(0, len(selected) - val_count))
    validation_ids = {sample_id for _, sample_id in selected[:val_count]}
    test_ids = {sample_id for _, sample_id in selected[val_count : val_count + test_count]}
    _log(
        args,
        "holdout-scan",
        "done",
        scanned=scanned,
        eligible=eligible,
        validation=len(validation_ids),
        test=len(test_ids),
    )
    return validation_ids, test_ids, {"eligible": eligible, "scanned": scanned}


def _resolve_split_strategy(args: argparse.Namespace) -> str:
    if args.split_strategy != "auto":
        return args.split_strategy
    if any(
        value > 0
        for value in (
            int(args.max_train_samples),
            int(args.max_validation_samples),
            int(args.max_test_samples),
        )
    ):
        return "subset"
    if int(args.validation_count) <= 0 and int(args.test_count) <= 0:
        return "train_only"
    return "exact_hash"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(
        args,
        "startup",
        "begin pretokenize_emilia",
        output_dir=str(output_root),
        dataset_id=args.dataset_id,
        data_files=args.data_files,
        source_split=args.source_split,
        tokenizer_name=args.tokenizer_name,
        codec_backend=args.audio_codec_backend,
        codec_source=args.audio_codec_source,
        codec_model_id=args.audio_codec_model_id or "-",
        num_quantizers=int(args.num_quantizers),
        seq_len=int(args.seq_len),
        reference_seq_len=int(args.reference_seq_len),
        max_samples=int(args.max_samples),
        max_seconds=float(args.max_seconds),
        split_strategy=args.split_strategy,
        export_format=args.export_format,
    )

    job_cfg = JobConfig()
    job_cfg.model.num_quantizers = int(args.num_quantizers)
    job_cfg.audio_codec.backend = args.audio_codec_backend
    job_cfg.audio_codec.source = args.audio_codec_source
    if args.audio_codec_model_id.strip():
        job_cfg.audio_codec.model_id = args.audio_codec_model_id.strip()
    if args.audio_codec_ckpt_path.strip():
        job_cfg.audio_codec.codec_ckpt_path = args.audio_codec_ckpt_path.strip()
    job_cfg.audio_codec.trust_remote_code = bool(args.audio_codec_trust_remote_code)
    _log(args, "codec", "loading audio codec", device=device)
    audio_tokenizer, feature_extractor, codec_info = load_audio_codec(job_cfg, device)
    _log(
        args,
        "codec",
        "audio codec ready",
        backend=codec_info.backend,
        source=codec_info.source,
        model_ref=codec_info.model_ref,
        sample_rate=int(codec_info.sampling_rate),
        codebook_size=int(codec_info.codebook_size),
        max_codebooks=int(codec_info.max_codebooks),
    )

    tokenizer = None
    audio_start_id = None
    audio_end_id = None
    if args.export_format == "model_ready":
        _log(args, "tokenizer", "loading tokenizer")
        tokenizer = _build_tokenizer(
            args.tokenizer_name,
            codebook_size=int(codec_info.codebook_size),
            num_quantizers=int(args.num_quantizers),
            language_tokens=bool(args.language_tokens),
        )
        _log(
            args,
            "tokenizer",
            "tokenizer ready",
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        vocab = tokenizer.get_vocab()
        audio_start_id = vocab.get("<audio>")
        audio_end_id = vocab.get("</audio>")
    else:
        _log(args, "tokenizer", "skipped tokenizer load for codec_only export")

    split_strategy = _resolve_split_strategy(args)
    _log(
        args,
        "splits",
        "resolved split strategy",
        dataset=args.dataset_id,
        data_files=args.data_files,
        source_split=args.source_split,
        split_strategy=split_strategy,
    )
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    selection_stats = {"eligible": 0, "scanned": 0}
    if split_strategy == "exact_hash":
        validation_ids, test_ids, selection_stats = _select_exact_holdout_ids(args)

    counts = {"train": 0, "validation": 0, "test": 0}
    reference_counts = {"train": 0, "validation": 0, "test": 0}
    dropped = {
        "missing_fields": 0,
        "duration": 0,
        "token_overflow": 0,
        "decode_error": 0,
        "assignment_skip": 0,
    }
    shard_index = {"train": 0, "validation": 0, "test": 0}
    buffers: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    written_files: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    sample_id_hashers = {
        "train": hashlib.sha256(),
        "validation": hashlib.sha256(),
        "test": hashlib.sha256(),
    }
    last_speaker_example: dict[str, dict[str, dict[str, Any]]] | None = None
    if args.emit_static_references:
        last_speaker_example = {
            "train": {},
            "validation": {},
            "test": {},
        }

    def flush_split(split: str) -> None:
        if not buffers[split]:
            return
        shard_path = (
            output_root
            / split
            / "lang=en"
            / f"part-{shard_index[split]:05d}.parquet"
        )
        written = _flush_records(buffers[split], shard_path)
        if written:
            written_files[split].append(str(shard_path.relative_to(output_root)))
            shard_index[split] += 1
            _log(
                args,
                "flush",
                "wrote parquet shard",
                split=split,
                rows=written,
                file=shard_path.name,
            )

    def assign_subset_split() -> str | None:
        if args.max_train_samples and counts["train"] < int(args.max_train_samples):
            return "train"
        if args.max_validation_samples and counts["validation"] < int(args.max_validation_samples):
            return "validation"
        if args.max_test_samples and counts["test"] < int(args.max_test_samples):
            return "test"
        return None

    pending_records: list[tuple[str, SourceRecord, torch.Tensor]] = []
    pending_seconds = 0.0
    scanned = 0

    def flush_pending_batch() -> None:
        nonlocal pending_seconds
        if not pending_records:
            return
        _log(
            args,
            "batch",
            "encoding pending batch",
            clips=len(pending_records),
            audio_seconds=round(pending_seconds, 2),
        )
        waveforms = [waveform for _, _, waveform in pending_records]
        if args.audio_codec_backend == "mimi":
            codes_per_record = _encode_batch_mimi(
                waveforms,
                feature_extractor,
                audio_tokenizer,
                int(args.num_quantizers),
                device,
            )
        else:
            codes_per_record = [
                _encode_sample_fallback(
                    waveform,
                    feature_extractor,
                    audio_tokenizer,
                    int(args.num_quantizers),
                )
                for waveform in waveforms
            ]
        for (split, record, waveform), audio_codes in zip(pending_records, codes_per_record, strict=True):
            row: dict[str, Any] = {
                "sample_id": record.sample_id,
                "text": record.text,
                "lang": record.lang,
                "audio_codes": audio_codes,
                "speaker_id": record.speaker_id,
                "duration_sec": float(record.duration_sec),
                "source_dataset": args.dataset_id,
                "source_audio_field": "mp3",
                "source_audio_path": record.audio_path,
                "source_url": record.source_url,
                "audio_token_count": int(len(audio_codes) * int(args.num_quantizers)),
                "source_split": split,
            }
            if args.export_format == "model_ready":
                assert tokenizer is not None
                proto = {
                    "text": record.text,
                    "lang": record.lang,
                    "audio_codes": audio_codes,
                }
                prompt = process_pretokenized_tts(
                    proto,
                    int(args.num_quantizers),
                    bool(args.language_tokens),
                    args.audio_codec_backend,
                )
                raw = tokenizer(prompt, padding=False, truncation=False)
                input_ids = list(raw["input_ids"])
                if len(input_ids) > int(args.seq_len):
                    dropped["token_overflow"] += 1
                    continue
                tokenized = tokenizer(
                    prompt,
                    max_length=int(args.seq_len),
                    padding="max_length",
                    truncation=True,
                )
                labels = _build_labels(
                    list(tokenized["input_ids"]),
                    list(tokenized["attention_mask"]),
                    audio_start_id=audio_start_id,
                    audio_end_id=audio_end_id,
                    mask_text_loss=bool(args.mask_text_loss),
                )
                row["total_token_count"] = int(sum(tokenized["attention_mask"]))
                row["input_ids"] = list(tokenized["input_ids"])
                row["attention_mask"] = list(tokenized["attention_mask"])
                row["labels"] = labels
            prior_reference = None
            if last_speaker_example is not None and record.speaker_id:
                prior_reference = last_speaker_example[split].get(record.speaker_id)
            if prior_reference is not None:
                if args.export_format == "model_ready":
                    assert tokenizer is not None
                    ref_prompt = _build_audio_only_prompt(
                        {"audio_codes": prior_reference["audio_codes"]},
                        int(args.num_quantizers),
                        args.audio_codec_backend,
                    )
                    ref_tokenized = tokenizer(
                        ref_prompt,
                        max_length=int(args.reference_seq_len),
                        padding="max_length",
                        truncation=True,
                    )
                    row["ref_input_ids"] = list(ref_tokenized["input_ids"])
                    row["ref_attention_mask"] = list(ref_tokenized["attention_mask"])
                row["ref_sample_id"] = str(prior_reference["sample_id"])
                row["ref_audio_codes"] = prior_reference["audio_codes"]
                reference_counts[split] += 1
            elif args.export_format == "model_ready" and args.keep_audio_codes:
                row["audio_codes"] = audio_codes
            buffers[split].append(row)
            counts[split] += 1
            sample_id_hashers[split].update(record.sample_id.encode("utf-8"))
            sample_id_hashers[split].update(b"\n")
            if last_speaker_example is not None and record.speaker_id:
                last_speaker_example[split][record.speaker_id] = {
                    "sample_id": record.sample_id,
                    "audio_codes": audio_codes,
                }
            if len(buffers[split]) >= int(args.shard_size):
                flush_split(split)
        pending_records.clear()
        pending_seconds = 0.0

    ds = _load_source(args.dataset_id, args.data_files, args.source_split, decode_audio=False)
    _log(args, "dataset", "source stream opened")
    for row in ds:
        if args.max_samples and scanned >= args.max_samples:
            break
        scanned += 1
        record = _normalize_row(row)
        if record is None:
            dropped["missing_fields"] += 1
            continue
        if record.duration_sec < args.min_seconds or record.duration_sec > args.max_seconds:
            dropped["duration"] += 1
            continue
        if split_strategy == "train_only":
            if args.max_train_samples and counts["train"] >= int(args.max_train_samples):
                break
            split = "train"
        elif split_strategy == "exact_hash":
            if record.sample_id in validation_ids:
                split = "validation"
            elif record.sample_id in test_ids:
                split = "test"
            else:
                split = "train"
        else:
            split = assign_subset_split()
            if split is None:
                dropped["assignment_skip"] += 1
                if all(
                    counts[name] >= limit
                    for name, limit in (
                        ("train", int(args.max_train_samples or 0)),
                        ("validation", int(args.max_validation_samples or 0)),
                        ("test", int(args.max_test_samples or 0)),
                    )
                    if limit > 0
                ):
                    break
                continue
        try:
            waveform = _decode_waveform(record.audio_bytes, int(codec_info.sampling_rate))
        except Exception:
            dropped["decode_error"] += 1
            continue
        pending_records.append((split, record, waveform))
        pending_seconds += float(waveform.numel()) / float(codec_info.sampling_rate)
        if (
            len(pending_records) >= int(args.batch_max_clips)
            or pending_seconds >= float(args.batch_max_audio_seconds)
        ):
            flush_pending_batch()
        if scanned % 10_000 == 0:
            _log(
                args,
                "tokenize",
                "progress",
                scanned=scanned,
                train=counts["train"],
                validation=counts["validation"],
                test=counts["test"],
                dropped_missing=dropped["missing_fields"],
                dropped_duration=dropped["duration"],
                dropped_decode=dropped["decode_error"],
                dropped_overflow=dropped["token_overflow"],
            )
    flush_pending_batch()
    for split in ("train", "validation", "test"):
        flush_split(split)

    manifest = {
        "dataset_name": args.dataset_id,
        "data_files": args.data_files,
        "source_split": args.source_split,
        "artifact_kind": (
            "model_ready_tts" if args.export_format == "model_ready" else "codec_pretok_tts"
        ),
        "objective": "text_audio_pair",
        "export_format": args.export_format,
        "reference_conditioning": {
            "enabled": True,
            "reference_seq_len": int(args.reference_seq_len),
            "pairing_rule": (
                "previous_same_speaker_within_split"
                if args.emit_static_references
                else "dynamic_same_speaker_at_train_time"
            ),
            "counts_with_reference": reference_counts,
            "static_references_emitted": bool(args.emit_static_references),
        },
        "num_quantizers": int(args.num_quantizers),
        "tokenizer_name": args.tokenizer_name,
        "seq_len": int(args.seq_len),
        "mask_text_loss": bool(args.mask_text_loss),
        "language_tokens": bool(args.language_tokens),
        "split_strategy": split_strategy,
        "selection": {
            "min_seconds": float(args.min_seconds),
            "max_seconds": float(args.max_seconds),
            "max_train_samples": int(args.max_train_samples),
            "max_validation_samples": int(args.max_validation_samples),
            "max_test_samples": int(args.max_test_samples),
            "validation_count": int(args.validation_count),
            "test_count": int(args.test_count),
            "max_samples": int(args.max_samples),
            "seed": int(args.seed),
        },
        "audio_codec": {
            "backend": codec_info.backend,
            "source": codec_info.source,
            "model_ref": codec_info.model_ref,
            "sample_rate": int(codec_info.sampling_rate),
            "codebook_size": int(codec_info.codebook_size),
            "max_codebooks": int(codec_info.max_codebooks),
        },
        "counts": counts,
        "dropped": dropped,
        "shard_size": int(args.shard_size),
        "written_files": written_files,
        "source_scan": selection_stats if split_strategy == "exact_hash" else {"scanned": scanned},
        "sample_id_sha256_by_split": {
            split: hasher.hexdigest() for split, hasher in sample_id_hashers.items()
        },
    }
    _write_manifest(output_root, manifest)
    _log(
        args,
        "complete",
        "manifest written",
        train=counts["train"],
        validation=counts["validation"],
        test=counts["test"],
        references_train=reference_counts["train"],
        dropped_total=sum(int(value) for value in dropped.values()),
        manifest_path=str(output_root / "dataset_manifest.json"),
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
