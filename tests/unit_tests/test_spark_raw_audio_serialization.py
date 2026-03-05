from types import SimpleNamespace

import torch

from torchtitan.datasets.hf_datasets import audio_array_to_text, process_audio


class _DummyBatch(dict):
    def to(self, _device):
        return self


class _DummyFeatureExtractor:
    sampling_rate = 16000

    def __call__(self, raw_audio, sampling_rate, return_tensors="pt"):
        del raw_audio, sampling_rate, return_tensors
        input_values = torch.zeros((1, 320), dtype=torch.float32)
        return _DummyBatch(
            {
                "input_values": input_values,
                "padding_mask": torch.ones_like(input_values, dtype=torch.long),
            }
        )


class _DummyAudioTokenizer:
    device = "cpu"

    def encode(self, input_values, padding_mask, num_quantizers):
        del input_values, padding_mask, num_quantizers
        return SimpleNamespace(
            audio_codes=torch.tensor([[[1, 2, 3]]], dtype=torch.long),
            global_codes=torch.tensor([[9, 8]], dtype=torch.long),
        )


def test_audio_array_to_text_spark_includes_global_and_audio_spans():
    text = audio_array_to_text(
        audio_array=torch.zeros((320,), dtype=torch.float32),
        audio_tokenizer=_DummyAudioTokenizer(),
        feature_extractor=_DummyFeatureExtractor(),
        num_quantizers=1,
        max_seconds=20,
        codec_backend="spark_bicodec",
    )
    assert text.startswith("<|start_global_token|>")
    assert "<|bicodec_global_9|><|bicodec_global_8|>" in text
    assert text.endswith("</audio>")
    assert "<audio><|bicodec_semantic_1|><|bicodec_semantic_2|><|bicodec_semantic_3|></audio>" in text


def test_process_audio_spark_tts_uses_task_and_content_framing():
    sample = {
        "audio": {"array": torch.zeros((320,), dtype=torch.float32)},
        "text": "hello world",
        "lang": "en",
    }
    text = process_audio(
        sample=sample,
        audio_tokenizer=_DummyAudioTokenizer(),
        feature_extractor=_DummyFeatureExtractor(),
        num_quantizers=1,
        task="tts",
        max_audio_seconds=20,
        language_tokens=True,
        codec_backend="spark_bicodec",
    )
    assert text.startswith("<|task_tts|><|start_content|>")
    assert "<lang_en>hello world<|end_content|>" in text
    assert "<|start_global_token|>" in text
    assert "<audio><|bicodec_semantic_1|>" in text
