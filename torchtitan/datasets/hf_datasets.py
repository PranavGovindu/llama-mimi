# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from typing import Any

import torch

from datasets import Audio, Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from transformers import default_data_collator


def audio_array_to_text(
    audio_array: torch.tensor,
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
    max_seconds: int = 20,
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
    with torch.no_grad():
        # Encode the audio input to get the audio codes
        # This will return a tensor of shape (batch_size, num_quantizers, sequence_length)
        # where each quantizer's output is in a separate dimension
        encoder_outputs = audio_tokenizer.encode(
            inputs["input_values"],
            inputs["padding_mask"],
            num_quantizers=num_quantizers,
        )
    flatten_audio_codes = encoder_outputs.audio_codes.transpose(1, 2).reshape(-1)
    assert flatten_audio_codes.numel() % num_quantizers == 0
    steps = []
    for i in range(0, flatten_audio_codes.numel(), num_quantizers):
        group = [
            f"<{flatten_audio_codes[i + j].item()}_{j}>" for j in range(num_quantizers)
        ]
        steps.append(group)

    parts = [tok for step in steps for tok in step]

    text = "".join(parts)

    del inputs, encoder_outputs, flatten_audio_codes
    torch.cuda.empty_cache()
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
) -> str:
    audio_sample = sample["audio"]["array"]
    text = audio_array_to_text(
        audio_sample,
        audio_tokenizer,
        feature_extractor,
        num_quantizers,
        max_seconds=max_audio_seconds,
    )
    if task == "tts":
        transcription = sample["text"]
        if language_tokens and sample.get("lang"):
            transcription = f"<lang_{_normalize_lang(sample['lang'])}>{transcription}"
        text = transcription + text
    return text


def _coerce_mimi_codes(raw_codes: Any, num_quantizers: int) -> list[list[int]]:
    if not isinstance(raw_codes, list) or len(raw_codes) == 0:
        return []

    # Expected layout is [T, Q], but handle [Q, T] too.
    if isinstance(raw_codes[0], list):
        rows = len(raw_codes)
        cols = len(raw_codes[0])

        # Typical [T, Q] layout (many frames, few quantizers). We allow
        # truncating a higher-Q sample (e.g. Q8) to a lower requested Q.
        if cols <= 32 and rows >= cols:
            if cols < num_quantizers:
                return []
            out: list[list[int]] = []
            for frame in raw_codes:
                if not isinstance(frame, list) or len(frame) < num_quantizers:
                    continue
                out.append([int(frame[q]) for q in range(num_quantizers)])
            return out

        # Typical [Q, T] layout (few quantizers, many frames).
        if rows <= 32 and cols > rows:
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

        # Backward-compatible exact-shape fallback.
        if cols == num_quantizers:
            return [[int(x) for x in frame] for frame in raw_codes if isinstance(frame, list)]
        if rows == num_quantizers:
            transposed = list(map(list, zip(*raw_codes)))
            return [[int(x) for x in frame] for frame in transposed]
    return []


def process_pretokenized_tts(
    sample: dict[str, Any],
    num_quantizers: int,
    language_tokens: bool = False,
) -> str:
    transcription = sample["text"]
    if language_tokens and sample.get("lang"):
        transcription = f"<lang_{_normalize_lang(sample['lang'])}>{transcription}"

    codes = _coerce_mimi_codes(sample.get("mimi_codes"), num_quantizers)
    audio_text = mimi_codes_to_text(codes, num_quantizers)
    return transcription + audio_text


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
    ) -> None:
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
        elif dataset_name == "fleurs_pretok":
            if not dataset_path:
                raise ValueError(
                    "training.dataset_path must be set for dataset 'fleurs_pretok'."
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

        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        self.audio_start_id = vocab.get("<audio>")
        self.audio_end_id = vocab.get("</audio>")

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self._overfit_cache: list[dict[str, Any]] = []

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
        if self.dataset_name == "fleurs_pretok":
            sample_text = process_pretokenized_tts(
                sample,
                self.num_quantizers,
                self.language_tokens,
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
            )
        tokenized = self.tokenizer(
            sample_text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
        )
        tokenized["labels"] = self._build_labels(
            tokenized["input_ids"],
            tokenized["attention_mask"],
        )
        return tokenized

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
                self._overfit_cache.append(tokenized)
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

    def __iter__(self):
        if self.overfit_num_samples > 0:
            self._prepare_overfit_cache()
            while True:
                for sample in self._overfit_cache:
                    # Return copies to avoid accidental in-place mutations by collators.
                    yield {k: (list(v) if isinstance(v, list) else v) for k, v in sample.items()}
                if not self.infinite:
                    break
            return

        while True:
            data_iter = self._get_data_iter()
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
                    yield tokenized
                except Exception as e:
                    logger.error(
                        f"Error while processing sample in dataset {self.dataset_name}: {e}"
                    )
                    self._sample_idx += 1
                    continue

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
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

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
    )

    collate_fn = default_data_collator

    overfit_mode = job_config.training.overfit_num_samples > 0
    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0 if overfit_mode else 2,
        prefetch_factor=None if overfit_mode else 2,
        pin_memory=not overfit_mode,
        persistent_workers=False if overfit_mode else True,
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
        split_name="validation",
        dataset_path=job_config.training.dataset_path,
        max_audio_seconds=job_config.training.max_audio_seconds,
        mask_text_loss=job_config.training.mask_text_loss,
        language_tokens=job_config.training.language_tokens,
        overfit_num_samples=0,
    )

    collate_fn = default_data_collator

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
