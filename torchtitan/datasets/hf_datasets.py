# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random
from collections import defaultdict
from typing import Any

import torch
import pyarrow.parquet as pq

from datasets import Audio, Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


def audio_array_to_text(
    audio_array: torch.tensor,
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
    max_seconds: int = 20,
    codec_backend: str = "mimi",
) -> str:
    # truncate the audio array to the expected length
    if audio_array.shape[-1] > max_seconds * feature_extractor.sampling_rate:
        audio_array = audio_array[: max_seconds * feature_extractor.sampling_rate]
        #
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
        # Encode the audio input to get the audio codes
        # This will return a tensor of shape (batch_size, num_quantizers, sequence_length)
        # where each quantizer's output is in a separate dimension
        encoder_outputs = audio_tokenizer.encode(
            inputs["input_values"],
            padding_mask,
            num_quantizers=num_quantizers,
        )
    backend = codec_backend.strip().lower()
    if backend == "spark_bicodec":
        semantic_codes = encoder_outputs.audio_codes
        if semantic_codes.ndim == 3:
            semantic_codes = semantic_codes[:, 0, :]
        if semantic_codes.ndim == 1:
            semantic_codes = semantic_codes.unsqueeze(0)
        semantic_ids = semantic_codes[0].detach().cpu().tolist()
        semantic_text = "".join(f"<|bicodec_semantic_{int(x)}|>" for x in semantic_ids)

        global_ids: list[int] = []
        raw_global = getattr(encoder_outputs, "global_codes", None)
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
            if raw_global.ndim == 2 and raw_global.shape[0] > 0:
                global_ids = raw_global[0].detach().cpu().to(torch.int64).tolist()

        global_text = spark_global_tokens_to_text(global_ids)
        text = f"{global_text}<audio>{semantic_text}</audio>"
    else:
        flatten_audio_codes = encoder_outputs.audio_codes.transpose(1, 2).reshape(-1)
        assert flatten_audio_codes.numel() % num_quantizers == 0
        steps = []
        for i in range(0, flatten_audio_codes.numel(), num_quantizers):
            group = [
                f"<{flatten_audio_codes[i + j].item()}_{j}>"
                for j in range(num_quantizers)
            ]
            steps.append(group)
        parts = [tok for step in steps for tok in step]
        text = "".join(parts)

    del inputs, encoder_outputs
    torch.cuda.empty_cache()
    if backend == "spark_bicodec":
        return text
    return f"<audio>{text}</audio>"


def mimi_codes_to_text(mimi_codes: list[list[int]], num_quantizers: int) -> str:
    steps = []
    for frame in mimi_codes:
        if len(frame) < num_quantizers:
            continue
        for q in range(num_quantizers):
            steps.append(f"<{int(frame[q])}_{q}>")
    return f"<audio>{''.join(steps)}</audio>"


def _normalize_lang(lang: str) -> str:
    return lang.strip().lower().replace("-", "_")


def process_audio(
    sample: dict[str, Any],
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
    task: str = "a2a",
    max_audio_seconds: int = 20,
    language_tokens: bool = False,
    codec_backend: str = "mimi",
) -> str:
    audio_sample = sample["audio"]["array"]
    audio_text = audio_array_to_text(
        audio_sample,
        audio_tokenizer,
        feature_extractor,
        num_quantizers,
        max_seconds=max_audio_seconds,
        codec_backend=codec_backend,
    )
    backend = codec_backend.strip().lower()
    if task == "tts":
        transcription = sample["text"]
        if language_tokens and sample.get("lang"):
            transcription = f"<lang_{_normalize_lang(sample['lang'])}>{transcription}"
        if backend == "spark_bicodec":
            return (
                "<|task_tts|>"
                f"<|start_content|>{transcription}<|end_content|>"
                f"{audio_text}"
            )
        return transcription + audio_text
    return audio_text


def _coerce_audio_codes(raw_codes: Any, num_quantizers: int) -> list[list[int]]:
    if not isinstance(raw_codes, list) or len(raw_codes) == 0:
        return []

    # Expected layout is [T, Q], but handle [Q, T] too.
    if isinstance(raw_codes[0], list):
        rows = len(raw_codes)
        cols = len(raw_codes[0])

        def _from_tq() -> list[list[int]]:
            if cols < num_quantizers:
                return []
            out: list[list[int]] = []
            for frame in raw_codes:
                if not isinstance(frame, list) or len(frame) < num_quantizers:
                    continue
                out.append([int(frame[q]) for q in range(num_quantizers)])
            return out

        def _from_qt() -> list[list[int]]:
            if rows < num_quantizers:
                return []
            q_rows = raw_codes[:num_quantizers]
            if not all(isinstance(r, list) and len(r) > 0 for r in q_rows):
                return []
            t_count = min(len(r) for r in q_rows)
            out: list[list[int]] = []
            for t in range(t_count):
                out.append([int(q_rows[q][t]) for q in range(num_quantizers)])
            return out

        # Strong shape hints first.
        # [T, Q] examples:
        # - very short clips where T=1 and Q>=num_quantizers
        # - exact-width Q columns.
        if cols >= num_quantizers and (rows < num_quantizers or cols == num_quantizers):
            return _from_tq()

        # [Q, T] examples where time axis is shorter than requested quantizers.
        if rows >= num_quantizers and cols < num_quantizers:
            return _from_qt()

        # Typical [T, Q] layout (many frames, few quantizers). We allow
        # truncating a higher-Q sample (e.g. Q8) to a lower requested Q.
        if cols <= 32 and rows > cols:
            return _from_tq()

        # Typical [Q, T] layout (few quantizers, many frames).
        if rows <= 32 and cols > rows:
            return _from_qt()

        # Backward-compatible exact-shape fallback.
        if cols == num_quantizers:
            return _from_tq()
        if rows == num_quantizers:
            return _from_qt()
    return []


def _coerce_token_vector(raw_tokens: Any) -> list[int]:
    if not isinstance(raw_tokens, list):
        return []
    out: list[int] = []
    for item in raw_tokens:
        if isinstance(item, list):
            if not item:
                continue
            out.append(int(item[0]))
        else:
            out.append(int(item))
    return out


def spark_semantic_tokens_to_text(semantic_tokens: list[int]) -> str:
    if not semantic_tokens:
        return "<audio></audio>"
    joined = "".join(f"<|bicodec_semantic_{int(tok)}|>" for tok in semantic_tokens)
    return f"<audio>{joined}</audio>"


def spark_global_tokens_to_text(global_tokens: list[int]) -> str:
    joined = "".join(f"<|bicodec_global_{int(tok)}|>" for tok in global_tokens)
    return f"<|start_global_token|>{joined}<|end_global_token|>"


def process_pretokenized_tts(
    sample: dict[str, Any],
    num_quantizers: int,
    language_tokens: bool = False,
    codec_backend: str = "mimi",
) -> str:
    transcription = sample["text"]
    if language_tokens and sample.get("lang"):
        transcription = f"<lang_{_normalize_lang(sample['lang'])}>{transcription}"

    if codec_backend.strip().lower() == "spark_bicodec":
        semantic_tokens = _coerce_token_vector(sample.get("spark_semantic_tokens"))
        if not semantic_tokens:
            # Fallback for mixed-format shards: reuse first channel from [T,Q].
            codes = _coerce_audio_codes(sample.get("audio_codes"), max(num_quantizers, 1))
            semantic_tokens = [int(frame[0]) for frame in codes if frame]
        global_tokens = _coerce_token_vector(sample.get("spark_global_tokens"))
        if not global_tokens:
            global_tokens = _coerce_token_vector(sample.get("global_tokens"))
        prompt = (
            "<|task_tts|>"
            f"<|start_content|>{transcription}<|end_content|>"
            f"{spark_global_tokens_to_text(global_tokens)}"
        )
        return prompt + spark_semantic_tokens_to_text(semantic_tokens)

    codes = _coerce_audio_codes(sample.get("audio_codes"), num_quantizers)
    if not codes:
        codes = _coerce_audio_codes(sample.get("mimi_codes"), num_quantizers)
    audio_text = mimi_codes_to_text(codes, num_quantizers)
    return transcription + audio_text


def _coerce_mimi_codes(raw_codes: Any, num_quantizers: int) -> list[list[int]]:
    """Backward-compatible alias retained for existing tests/imports."""
    return _coerce_audio_codes(raw_codes, num_quantizers)


def _has_precomputed_tensors(sample: dict[str, Any]) -> bool:
    return all(key in sample for key in ("input_ids", "attention_mask"))


def _has_precomputed_reference(sample: dict[str, Any]) -> bool:
    return all(key in sample for key in ("ref_input_ids", "ref_attention_mask"))


def _coerce_int_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, list):
            if not item:
                continue
            out.append(int(item[0]))
        else:
            out.append(int(item))
    return out


def _build_audio_only_prompt(
    sample: dict[str, Any],
    num_quantizers: int,
    codec_backend: str = "mimi",
) -> str:
    if codec_backend.strip().lower() == "spark_bicodec":
        semantic_tokens = _coerce_token_vector(sample.get("spark_semantic_tokens"))
        if not semantic_tokens:
            codes = _coerce_audio_codes(sample.get("audio_codes"), max(num_quantizers, 1))
            semantic_tokens = [int(frame[0]) for frame in codes if frame]
        global_tokens = _coerce_token_vector(sample.get("spark_global_tokens"))
        if not global_tokens:
            global_tokens = _coerce_token_vector(sample.get("global_tokens"))
        return (
            f"{spark_global_tokens_to_text(global_tokens)}"
            f"{spark_semantic_tokens_to_text(semantic_tokens)}"
        )

    codes = _coerce_audio_codes(sample.get("audio_codes"), num_quantizers)
    if not codes:
        codes = _coerce_audio_codes(sample.get("mimi_codes"), num_quantizers)
    return mimi_codes_to_text(codes, num_quantizers)


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: BaseTokenizer,
        audio_tokenizer=None,
        feature_extractor=None,
        num_quantizers: int = 4,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        task: str = "a2a",
        split_name: str = "train",
        dataset_path: str | None = None,
        max_audio_seconds: int = 20,
        mask_text_loss: bool = False,
        language_tokens: bool = False,
        overfit_num_samples: int = 0,
        codec_backend: str = "mimi",
        shuffle: bool = True,
        enable_reference_conditioning: bool = False,
        reference_seq_len: int = 1024,
        reference_sampling_strategy: str = "auto",
        reference_pool_size: int = 4,
        dynamic_padding: bool = False,
        length_bucket_buffer_size: int = 0,
    ) -> None:
        parquet_files: list[str] = []
        if dataset_name == "peoples_speech":
            ds = load_dataset(
                "parquet",
                data_files="data/peoples_speech/**/*.parquet",
                split="train",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name == "all":
            logger.warning("Dataset alias 'all' maps to 'peoples_speech'.")
            ds = load_dataset(
                "parquet",
                data_files="data/peoples_speech/**/*.parquet",
                split="train",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name == "librispeech_asr_train":
            ds = load_dataset(
                "openslr/librispeech_asr",
                split="train.other.500",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name == "librispeech_asr_test":
            ds = load_dataset(
                "openslr/librispeech_asr",
                split="test.clean",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name == "librispeech_asr":
            split = "train.other.500" if split_name == "train" else "test.clean"
            ds = load_dataset(
                "openslr/librispeech_asr",
                split=split,
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name in {"fleurs_pretok", "tts_pretok"}:
            if not dataset_path:
                raise ValueError(
                    "training.dataset_path must be set for dataset 'fleurs_pretok'/'tts_pretok'."
                )
            requested_split = split_name
            parquet_patterns = [
                os.path.join(dataset_path, requested_split, "**/*.parquet"),
                os.path.join(dataset_path, f"{requested_split}*.parquet"),
            ]
            parquet_files = sorted(
                {
                    p
                    for pattern in parquet_patterns
                    for p in glob.glob(pattern, recursive=True)
                }
            )
            if not parquet_files and requested_split == "validation":
                logger.warning(
                    "No pretokenized validation split found; falling back to train split."
                )
                parquet_patterns = [
                    os.path.join(dataset_path, "train", "**/*.parquet"),
                    os.path.join(dataset_path, "train*.parquet"),
                ]
                parquet_files = sorted(
                    {
                        p
                        for pattern in parquet_patterns
                        for p in glob.glob(pattern, recursive=True)
                    }
                )
            if parquet_files:
                ds = load_dataset(
                    "parquet",
                    data_files=parquet_files,
                    split="train",
                    streaming=True,
                )
            else:
                json_patterns = [
                    os.path.join(dataset_path, requested_split, "**/*.jsonl"),
                    os.path.join(dataset_path, f"{requested_split}*.jsonl"),
                ]
                json_files = sorted(
                    {
                        p
                        for pattern in json_patterns
                        for p in glob.glob(pattern, recursive=True)
                    }
                )
                if not json_files and requested_split == "validation":
                    json_patterns = [
                        os.path.join(dataset_path, "train", "**/*.jsonl"),
                        os.path.join(dataset_path, "train*.jsonl"),
                    ]
                    json_files = sorted(
                        {
                            p
                            for pattern in json_patterns
                            for p in glob.glob(pattern, recursive=True)
                        }
                    )
                if not json_files:
                    raise ValueError(
                        f"No parquet/jsonl files found for split '{requested_split}' in {dataset_path}."
                    )
                ds = load_dataset(
                    "json",
                    data_files=json_files,
                    split="train",
                    streaming=True,
                )

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported. ")

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        if shuffle:
            self._data = self._data.shuffle(seed=42, buffer_size=10_000)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.feature_extractor = feature_extractor
        self.num_quantizers = num_quantizers
        self.seq_len = seq_len
        self.task = task
        self.infinite = infinite
        self.max_audio_seconds = max_audio_seconds
        self.mask_text_loss = mask_text_loss
        self.language_tokens = language_tokens
        self.overfit_num_samples = overfit_num_samples
        self.codec_backend = codec_backend.strip().lower()
        self.enable_reference_conditioning = enable_reference_conditioning
        self.reference_seq_len = int(reference_seq_len)
        self.reference_sampling_strategy = str(reference_sampling_strategy).strip().lower()
        self.reference_pool_size = max(0, int(reference_pool_size))
        self.dynamic_padding = bool(dynamic_padding)
        self.length_bucket_buffer_size = max(0, int(length_bucket_buffer_size))
        self._reference_rng = random.Random(42 + int(dp_rank))
        self._speaker_reference_pool: dict[str, list[dict[str, Any]]] = {}
        self._dynamic_reference_pool_enabled = False
        self._pretok_parquet_files = list(parquet_files)

        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        self.audio_start_id = vocab.get("<audio>")
        self.audio_end_id = vocab.get("</audio>")
        self.pad_token_id = (
            int(tokenizer.pad_token_id) if getattr(tokenizer, "pad_token_id", None) is not None else 0
        )

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self._overfit_cache: list[dict[str, Any]] = []

        self._maybe_initialize_dynamic_reference_pool()

    @staticmethod
    def _clone_tokenized_sample(sample: dict[str, Any]) -> dict[str, Any]:
        # Keep state/checkpoint payload independent from mutable collator outputs.
        return {k: (list(v) if isinstance(v, list) else v) for k, v in sample.items()}

    def _build_labels(
        self,
        input_ids: list[int],
        attention_mask: list[int],
    ) -> list[int]:
        labels = list(input_ids)
        if not self.mask_text_loss:
            for i, mask in enumerate(attention_mask):
                if mask == 0:
                    labels[i] = -100
            return labels
        if self.audio_start_id is None or self.audio_end_id is None:
            return [-100] * len(input_ids)

        # Supervise only generated audio tokens and the closing </audio> marker.
        try:
            audio_start_idx = input_ids.index(self.audio_start_id)
            audio_end_idx = input_ids.index(self.audio_end_id)
        except ValueError:
            return [-100] * len(input_ids)

        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100
                continue
            if i <= audio_start_idx or i > audio_end_idx:
                labels[i] = -100
        return labels

    @staticmethod
    def _effective_length(attention_mask: list[int], fallback_length: int) -> int:
        if attention_mask:
            effective = sum(int(mask_value) for mask_value in attention_mask)
            if effective > 0:
                return effective
        return fallback_length

    def _trim_main_tokens(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        labels: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        if not self.dynamic_padding:
            return input_ids, attention_mask, labels
        effective = self._effective_length(attention_mask, len(input_ids))
        return input_ids[:effective], attention_mask[:effective], labels[:effective]

    def _trim_reference_tokens(
        self,
        input_ids: list[int],
        attention_mask: list[int],
    ) -> tuple[list[int], list[int]]:
        if not self.dynamic_padding:
            return input_ids, attention_mask
        if not any(attention_mask):
            return [self.pad_token_id], [0]
        effective = self._effective_length(attention_mask, len(input_ids))
        return input_ids[:effective], attention_mask[:effective]

    def _tokenize_text(
        self,
        text: str,
        *,
        max_length: int,
    ) -> dict[str, list[int]]:
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=False if self.dynamic_padding else "max_length",
            truncation=True,
        )

    def _annotate_length_hint(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        tokenized["_length_hint"] = self._effective_length(
            tokenized.get("attention_mask", []),
            len(tokenized.get("input_ids", [])),
        )
        return tokenized

    def _maybe_initialize_dynamic_reference_pool(self) -> None:
        if not self.enable_reference_conditioning:
            return
        if self.reference_sampling_strategy not in {"auto", "dynamic_same_speaker"}:
            return
        if self.dataset_name not in {"fleurs_pretok", "tts_pretok"}:
            return
        if not self._pretok_parquet_files:
            return
        if self.reference_pool_size <= 0:
            return

        speaker_samples: dict[str, list[dict[str, Any]]] = {}
        speaker_seen: dict[str, int] = defaultdict(int)
        files_scanned = 0
        rows_scanned = 0

        for parquet_path in self._pretok_parquet_files:
            try:
                parquet_file = pq.ParquetFile(parquet_path)
            except Exception as exc:
                logger.warning(
                    f"Skipping reference-pool scan for {parquet_path}: could not open parquet ({exc})."
                )
                continue
            available = set(parquet_file.schema_arrow.names)
            required = {"speaker_id", "audio_codes"}
            if not required.issubset(available):
                continue
            columns = ["speaker_id", "audio_codes"]
            if "sample_id" in available:
                columns.append("sample_id")

            files_scanned += 1
            for batch in parquet_file.iter_batches(columns=columns, batch_size=2048):
                rows = batch.to_pylist()
                rows_scanned += len(rows)
                for row in rows:
                    speaker_id = row.get("speaker_id")
                    if speaker_id is None:
                        continue
                    speaker_key = str(speaker_id).strip()
                    if not speaker_key:
                        continue
                    audio_codes = row.get("audio_codes")
                    if not isinstance(audio_codes, list) or not audio_codes:
                        continue
                    candidate = {
                        "sample_id": str(row.get("sample_id") or ""),
                        "audio_codes": audio_codes,
                    }
                    speaker_seen[speaker_key] += 1
                    bucket = speaker_samples.setdefault(speaker_key, [])
                    if len(bucket) < self.reference_pool_size:
                        bucket.append(candidate)
                        continue
                    replace_idx = self._reference_rng.randint(0, speaker_seen[speaker_key] - 1)
                    if replace_idx < self.reference_pool_size:
                        bucket[replace_idx] = candidate

        self._speaker_reference_pool = speaker_samples
        self._dynamic_reference_pool_enabled = bool(self._speaker_reference_pool)
        if self._dynamic_reference_pool_enabled:
            logger.info(
                "Built dynamic same-speaker reference pool: "
                f"{len(self._speaker_reference_pool)} speakers from {files_scanned} parquet files "
                f"({rows_scanned} rows scanned, reservoir={self.reference_pool_size})."
            )
        elif self.reference_sampling_strategy == "dynamic_same_speaker":
            logger.warning(
                "Dynamic same-speaker reference sampling was requested, but no usable "
                "speaker/audio rows were found. Falling back to precomputed references."
            )

    def _sample_dynamic_reference(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        if not self._dynamic_reference_pool_enabled:
            return None
        speaker_id = sample.get("speaker_id")
        if speaker_id is None:
            return None
        speaker_key = str(speaker_id).strip()
        if not speaker_key:
            return None
        candidates = self._speaker_reference_pool.get(speaker_key, [])
        if not candidates:
            return None
        sample_id = str(sample.get("sample_id") or "")
        if sample_id:
            filtered = [candidate for candidate in candidates if candidate.get("sample_id") != sample_id]
            if filtered:
                candidates = filtered
        if not candidates:
            return None
        return self._reference_rng.choice(candidates)

    def _tokenize_reference_source(self, ref_source: dict[str, Any]) -> tuple[list[int], list[int]]:
        ref_prompt = _build_audio_only_prompt(
            {
                "audio_codes": ref_source.get("audio_codes"),
                "mimi_codes": ref_source.get("mimi_codes"),
                "spark_semantic_tokens": ref_source.get("spark_semantic_tokens"),
                "spark_global_tokens": ref_source.get("spark_global_tokens"),
                "global_tokens": ref_source.get("global_tokens"),
            },
            self.num_quantizers,
            self.codec_backend,
        )
        if ref_prompt.strip():
            ref_tokenized = self._tokenize_text(
                ref_prompt,
                max_length=self.reference_seq_len,
            )
            return self._trim_reference_tokens(
                list(ref_tokenized["input_ids"]),
                list(ref_tokenized["attention_mask"]),
            )
        return [self.pad_token_id] * self.reference_seq_len, [0] * self.reference_seq_len

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def _tokenize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.dataset_name in {"fleurs_pretok", "tts_pretok"} and _has_precomputed_tensors(
            sample
        ):
            input_ids = _coerce_int_list(sample.get("input_ids"))
            attention_mask = _coerce_int_list(sample.get("attention_mask"))
            if not input_ids or not attention_mask:
                raise ValueError("Precomputed sample is missing input_ids/attention_mask.")
            labels = _coerce_int_list(sample.get("labels"))
            if not labels or len(labels) != len(input_ids):
                labels = self._build_labels(input_ids, attention_mask)
            input_ids, attention_mask, labels = self._trim_main_tokens(
                input_ids,
                attention_mask,
                labels,
            )
            tokenized = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if self.enable_reference_conditioning:
                ref_input_ids = _coerce_int_list(sample.get("ref_input_ids"))
                ref_attention_mask = _coerce_int_list(sample.get("ref_attention_mask"))
                if ref_input_ids and ref_attention_mask and len(ref_input_ids) == len(ref_attention_mask):
                    ref_input_ids, ref_attention_mask = self._trim_reference_tokens(
                        ref_input_ids,
                        ref_attention_mask,
                    )
                    tokenized["ref_input_ids"] = ref_input_ids
                    tokenized["ref_attention_mask"] = ref_attention_mask
                else:
                    tokenized["ref_input_ids"] = [self.pad_token_id] * self.reference_seq_len
                    tokenized["ref_attention_mask"] = [0] * self.reference_seq_len
            return self._annotate_length_hint(tokenized)
        if self.dataset_name in {"fleurs_pretok", "tts_pretok"}:
            sample_text = process_pretokenized_tts(
                sample,
                self.num_quantizers,
                self.language_tokens,
                self.codec_backend,
            )
        else:
            sample_text = process_audio(
                sample,
                self.audio_tokenizer,
                self.feature_extractor,
                self.num_quantizers,
                self.task,
                self.max_audio_seconds,
                self.language_tokens,
                self.codec_backend,
            )
        tokenized = self._tokenize_text(
            sample_text,
            max_length=self.seq_len,
        )
        input_ids = list(tokenized["input_ids"])
        attention_mask = list(tokenized["attention_mask"])
        labels = self._build_labels(
            input_ids,
            attention_mask,
        )
        input_ids, attention_mask, labels = self._trim_main_tokens(
            input_ids,
            attention_mask,
            labels,
        )
        tokenized["input_ids"] = input_ids
        tokenized["attention_mask"] = attention_mask
        tokenized["labels"] = labels
        if self.enable_reference_conditioning:
            dynamic_ref = self._sample_dynamic_reference(sample)
            if dynamic_ref is not None:
                ref_input_ids, ref_attention_mask = self._tokenize_reference_source(dynamic_ref)
                tokenized["ref_input_ids"] = ref_input_ids
                tokenized["ref_attention_mask"] = ref_attention_mask
            elif _has_precomputed_reference(sample):
                ref_input_ids = _coerce_int_list(sample.get("ref_input_ids"))
                ref_attention_mask = _coerce_int_list(sample.get("ref_attention_mask"))
                if ref_input_ids and ref_attention_mask and len(ref_input_ids) == len(ref_attention_mask):
                    tokenized["ref_input_ids"] = ref_input_ids
                    tokenized["ref_attention_mask"] = ref_attention_mask
                else:
                    tokenized["ref_input_ids"] = [self.pad_token_id] * self.reference_seq_len
                    tokenized["ref_attention_mask"] = [0] * self.reference_seq_len
            else:
                ref_input_ids, ref_attention_mask = self._tokenize_reference_source(
                    {
                        "audio_codes": sample.get("ref_audio_codes"),
                        "mimi_codes": sample.get("ref_mimi_codes"),
                        "spark_semantic_tokens": sample.get("ref_spark_semantic_tokens"),
                        "spark_global_tokens": sample.get("ref_spark_global_tokens"),
                        "global_tokens": sample.get("ref_global_tokens"),
                    }
                )
                tokenized["ref_input_ids"] = ref_input_ids
                tokenized["ref_attention_mask"] = ref_attention_mask
        return self._annotate_length_hint(tokenized)

    def _prepare_overfit_cache(self) -> None:
        if self.overfit_num_samples <= 0 or self._overfit_cache:
            return
        data_iter = self._get_data_iter()
        while len(self._overfit_cache) < self.overfit_num_samples:
            try:
                sample = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                logger.error(
                    f"Error while iterating over dataset {self.dataset_name}: {e}"
                )
                self._sample_idx += 1
                continue
            try:
                tokenized = self._tokenize_sample(sample)
                self._overfit_cache.append(self._clone_tokenized_sample(tokenized))
                self._sample_idx += 1
            except Exception as e:
                logger.error(
                    f"Error while processing sample in dataset {self.dataset_name}: {e}"
                )
                self._sample_idx += 1
        if not self._overfit_cache:
            raise RuntimeError(
                f"Could not build overfit cache from dataset {self.dataset_name}."
            )
        logger.warning(
            f"Overfit mode active: repeating {len(self._overfit_cache)} sample(s)."
        )

    @staticmethod
    def _sort_tokenized_buffer(buffer: list[dict[str, Any]]) -> None:
        buffer.sort(
            key=lambda sample: int(
                sample.get("_length_hint", len(sample.get("input_ids", [])))
            )
        )

    def __iter__(self):
        if self.overfit_num_samples > 0:
            self._prepare_overfit_cache()
            while True:
                for sample in self._overfit_cache:
                    # Return copies to avoid accidental in-place mutations by collators.
                    yield self._clone_tokenized_sample(sample)
                if not self.infinite:
                    break
            return

        while True:
            data_iter = self._get_data_iter()
            bucket_buffer: list[dict[str, Any]] = []
            while True:
                try:
                    sample = next(data_iter)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(
                        f"Error while iterating over dataset {self.dataset_name}: {e}"
                    )
                    self._sample_idx += 1
                    continue

                try:
                    tokenized = self._tokenize_sample(sample)
                    self._sample_idx += 1
                    if self.length_bucket_buffer_size > 1:
                        bucket_buffer.append(tokenized)
                        if len(bucket_buffer) >= self.length_bucket_buffer_size:
                            self._sort_tokenized_buffer(bucket_buffer)
                            for buffered in bucket_buffer:
                                yield buffered
                            bucket_buffer = []
                    else:
                        yield tokenized
                except Exception as e:
                    logger.error(
                        f"Error while processing sample in dataset {self.dataset_name}: {e}"
                    )
                    self._sample_idx += 1
                    continue

            if bucket_buffer:
                self._sort_tokenized_buffer(bucket_buffer)
                for buffered in bucket_buffer:
                    yield buffered

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        cached_samples = state_dict.get("overfit_cache", [])
        if self.overfit_num_samples > 0 and isinstance(cached_samples, list):
            self._overfit_cache = [
                self._clone_tokenized_sample(sample)
                for sample in cached_samples
                if isinstance(sample, dict)
            ]
        else:
            self._overfit_cache = []

        if isinstance(self._data, Dataset):
            self._sample_idx = int(state_dict.get("sample_idx", 0))
            if self.overfit_num_samples > 0 and not self._overfit_cache:
                # Backward compatibility: older checkpoints did not persist
                # cached overfit samples and had already advanced sample_idx.
                # Rewind so cache rebuild uses the original overfit window.
                rewound_idx = max(0, self._sample_idx - self.overfit_num_samples)
                if rewound_idx != self._sample_idx:
                    logger.warning(
                        "Overfit resume without cached samples; rewinding sample_idx "
                        f"from {self._sample_idx} to {rewound_idx}."
                    )
                    self._sample_idx = rewound_idx
        else:
            if "data" in state_dict:
                self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}
        if self.overfit_num_samples > 0 and self._overfit_cache:
            _state_dict["overfit_cache"] = [
                self._clone_tokenized_sample(sample)
                for sample in self._overfit_cache
            ]

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    audio_tokenizer,
    feature_extractor,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        feature_extractor=feature_extractor,
        num_quantizers=job_config.model.num_quantizers,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        task=job_config.training.task,
        split_name="train",
        dataset_path=job_config.training.dataset_path,
        max_audio_seconds=job_config.training.max_audio_seconds,
        mask_text_loss=job_config.training.mask_text_loss,
        language_tokens=job_config.training.language_tokens,
        overfit_num_samples=job_config.training.overfit_num_samples,
        codec_backend=job_config.audio_codec.backend,
        shuffle=True,
        enable_reference_conditioning=job_config.training.enable_reference_conditioning,
        reference_seq_len=job_config.training.reference_seq_len,
        reference_sampling_strategy=job_config.training.reference_sampling_strategy,
        reference_pool_size=job_config.training.reference_pool_size,
        dynamic_padding=job_config.training.dynamic_padding,
        length_bucket_buffer_size=job_config.training.length_bucket_buffer_size,
    )

    collate_fn = build_tts_collate_fn(
        pad_token_id=hf_ds.pad_token_id,
        pad_to_multiple_of=job_config.training.pad_to_multiple_of,
    )

    overfit_mode = job_config.training.overfit_num_samples > 0
    num_workers = 0 if overfit_mode else max(0, int(job_config.training.dataloader_num_workers))
    prefetch_factor = (
        None
        if num_workers <= 0
        else max(1, int(job_config.training.dataloader_prefetch_factor))
    )
    persistent_workers = bool(
        not overfit_mode
        and num_workers > 0
        and job_config.training.dataloader_persistent_workers
    )
    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=bool(job_config.training.dataloader_pin_memory and not overfit_mode),
        persistent_workers=persistent_workers,
    )


def build_hf_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    audio_tokenizer,
    feature_extractor,
    job_config: JobConfig,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        feature_extractor=feature_extractor,
        num_quantizers=job_config.model.num_quantizers,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=False,
        task=job_config.training.task,
        split_name=job_config.validation.split,
        dataset_path=job_config.training.dataset_path,
        max_audio_seconds=job_config.training.max_audio_seconds,
        mask_text_loss=job_config.training.mask_text_loss,
        language_tokens=job_config.training.language_tokens,
        overfit_num_samples=0,
        codec_backend=job_config.audio_codec.backend,
        shuffle=False,
        enable_reference_conditioning=job_config.training.enable_reference_conditioning,
        reference_seq_len=job_config.training.reference_seq_len,
        reference_sampling_strategy=job_config.training.reference_sampling_strategy,
        reference_pool_size=job_config.training.reference_pool_size,
        dynamic_padding=job_config.training.dynamic_padding,
        length_bucket_buffer_size=0,
    )

    collate_fn = build_tts_collate_fn(
        pad_token_id=hf_ds.pad_token_id,
        pad_to_multiple_of=job_config.training.pad_to_multiple_of,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        persistent_workers=False,
    )


def _round_up_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    return ((length + multiple - 1) // multiple) * multiple


def _pad_1d(values: list[int], target_len: int, pad_value: int) -> list[int]:
    if len(values) >= target_len:
        return list(values[:target_len])
    return list(values) + [pad_value] * (target_len - len(values))


class TTSBatchCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        pad_to_multiple_of: int = 1,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        sanitized_batch = [
            {k: v for k, v in sample.items() if not str(k).startswith("_")}
            for sample in batch
        ]
        main_target_len = max(len(sample["input_ids"]) for sample in sanitized_batch)
        main_target_len = _round_up_to_multiple(
            main_target_len,
            self.pad_to_multiple_of,
        )

        collated: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(
                [
                    _pad_1d(sample["input_ids"], main_target_len, self.pad_token_id)
                    for sample in sanitized_batch
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    _pad_1d(sample["attention_mask"], main_target_len, 0)
                    for sample in sanitized_batch
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [
                    _pad_1d(sample["labels"], main_target_len, -100)
                    for sample in sanitized_batch
                ],
                dtype=torch.long,
            ),
        }

        if any("ref_input_ids" in sample for sample in sanitized_batch):
            ref_target_len = max(
                len(sample.get("ref_input_ids", [])) for sample in sanitized_batch
            )
            ref_target_len = _round_up_to_multiple(
                ref_target_len,
                self.pad_to_multiple_of,
            )
            collated["ref_input_ids"] = torch.tensor(
                [
                    _pad_1d(
                        sample.get("ref_input_ids", []),
                        ref_target_len,
                        self.pad_token_id,
                    )
                    for sample in sanitized_batch
                ],
                dtype=torch.long,
            )
            collated["ref_attention_mask"] = torch.tensor(
                [
                    _pad_1d(sample.get("ref_attention_mask", []), ref_target_len, 0)
                    for sample in sanitized_batch
                ],
                dtype=torch.long,
            )
        return collated


def build_tts_collate_fn(
    *,
    pad_token_id: int,
    pad_to_multiple_of: int = 1,
):
    return TTSBatchCollator(
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
