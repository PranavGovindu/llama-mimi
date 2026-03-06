from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import torch

from torchtitan.config_manager import AudioCodec, JobConfig
from torchtitan.tools.codecs.registry import build_codec_from_registry


@dataclass
class CodecRuntimeInfo:
    backend: str
    source: str
    model_ref: str
    sampling_rate: int
    codebook_size: int
    max_codebooks: int
    global_codebook_size: int = 0
    global_token_count: int = 0


class BatchTensorDict(dict):
    def to(self, device: torch.device | str) -> "BatchTensorDict":
        for key, value in list(self.items()):
            if torch.is_tensor(value):
                self[key] = value.to(device)
        return self


class SimpleFeatureExtractor:
    def __init__(self, sampling_rate: int):
        self.sampling_rate = int(sampling_rate)

    def __call__(
        self,
        raw_audio: Any,
        sampling_rate: int,
        return_tensors: str = "pt",
    ) -> BatchTensorDict:
        if return_tensors != "pt":
            raise ValueError("SimpleFeatureExtractor supports only return_tensors='pt'.")

        audio = torch.as_tensor(raw_audio, dtype=torch.float32).flatten()
        in_sr = int(sampling_rate)
        if in_sr != self.sampling_rate:
            # Lazy import to avoid forcing librosa for code paths that never need this.
            import librosa

            audio_np = librosa.resample(
                audio.detach().cpu().numpy(),
                orig_sr=in_sr,
                target_sr=self.sampling_rate,
            )
            audio = torch.as_tensor(audio_np, dtype=torch.float32)

        audio = audio.unsqueeze(0)
        return BatchTensorDict(
            {
                "input_values": audio,
                "padding_mask": torch.ones_like(audio, dtype=torch.long),
            }
        )


class BaseCodecAdapter:
    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        codebook_size: int,
        max_codebooks: int,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = getattr(model, "device", torch.device("cpu"))
        self.config = SimpleNamespace(codebook_size=int(codebook_size))
        self.max_codebooks = int(max_codebooks)

    @property
    def sampling_rate(self) -> int:
        return int(self.feature_extractor.sampling_rate)

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
        num_quantizers: int,
    ) -> SimpleNamespace:
        raise NotImplementedError

    def decode(
        self,
        codes_bqt: torch.Tensor,
        global_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class MimiCodecAdapter(BaseCodecAdapter):
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
        num_quantizers: int,
    ) -> SimpleNamespace:
        outputs = self.model.encode(
            input_values,
            padding_mask,
            num_quantizers=num_quantizers,
        )
        return SimpleNamespace(audio_codes=outputs.audio_codes)

    def decode(
        self,
        codes_bqt: torch.Tensor,
        global_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del global_codes
        return self.model.decode(codes_bqt)


class S1DacCodecAdapter(BaseCodecAdapter):
    def _call_encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
        num_quantizers: int,
    ) -> Any:
        kwargs_candidates = [
            {"num_quantizers": int(num_quantizers)},
            {"n_quantizers": int(num_quantizers)},
            {"num_codebooks": int(num_quantizers)},
        ]
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values, dtype=torch.long)

        attempts = [
            lambda kwargs: self.model.encode(input_values, padding_mask, **kwargs),
            lambda kwargs: self.model.encode(input_values, **kwargs),
            lambda kwargs: self.model.encode(audio=input_values, **kwargs),
            lambda kwargs: self.model.encode(input_values),
        ]
        last_error: Exception | None = None
        for kwargs in kwargs_candidates:
            for fn in attempts:
                try:
                    return fn(kwargs)
                except TypeError as exc:
                    last_error = exc
                except Exception as exc:  # pragma: no cover
                    last_error = exc
        if last_error is not None:
            raise RuntimeError(f"S1-DAC encode failed: {last_error}") from last_error
        raise RuntimeError("S1-DAC encode failed with unknown error.")

    @staticmethod
    def _extract_codes(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
                if isinstance(item, dict):
                    for key in (
                        "audio_codes",
                        "codebook_indices",
                        "codes",
                        "quantized_codes",
                    ):
                        if key in item and torch.is_tensor(item[key]):
                            return item[key]
        if isinstance(output, dict):
            for key in (
                "audio_codes",
                "codebook_indices",
                "codes",
                "quantized_codes",
            ):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
        for key in ("audio_codes", "codebook_indices", "codes", "quantized_codes"):
            value = getattr(output, key, None)
            if torch.is_tensor(value):
                return value
        raise RuntimeError("Unable to extract audio codes from S1-DAC encode output.")

    def _to_bqt(self, codes: torch.Tensor, num_quantizers: int) -> torch.Tensor:
        if codes.ndim == 2:
            # Likely [Q, T] or [T, Q]
            if codes.shape[0] <= max(self.max_codebooks, 32):
                codes = codes.unsqueeze(0)  # -> [B=1, Q, T]
            else:
                codes = codes.unsqueeze(0).transpose(1, 2)  # -> [B=1, Q, T]
        if codes.ndim != 3:
            raise RuntimeError(f"Unsupported S1-DAC code tensor shape: {tuple(codes.shape)}")

        q_axis = 1
        if codes.shape[1] < num_quantizers and codes.shape[2] >= num_quantizers:
            q_axis = 2
        elif (
            codes.shape[2] <= max(self.max_codebooks, 32)
            and codes.shape[1] > max(self.max_codebooks, 32)
        ):
            q_axis = 2

        if q_axis == 2:
            codes = codes.transpose(1, 2)

        if codes.shape[1] < num_quantizers:
            raise RuntimeError(
                f"S1-DAC produced {codes.shape[1]} codebooks, need {num_quantizers}."
            )
        return codes[:, :num_quantizers, :].long()

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
        num_quantizers: int,
    ) -> SimpleNamespace:
        outputs = self._call_encode(input_values, padding_mask, num_quantizers)
        codes = self._extract_codes(outputs)
        codes_bqt = self._to_bqt(codes, num_quantizers)
        return SimpleNamespace(audio_codes=codes_bqt)

    @staticmethod
    def _extract_audio_values(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
                if isinstance(item, dict):
                    for key in (
                        "audio_values",
                        "audio",
                        "audios",
                        "waveform",
                        "reconstructed_audio",
                    ):
                        if key in item and torch.is_tensor(item[key]):
                            return item[key]
        if isinstance(output, dict):
            for key in (
                "audio_values",
                "audio",
                "audios",
                "waveform",
                "reconstructed_audio",
            ):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
        for key in ("audio_values", "audio", "audios", "waveform", "reconstructed_audio"):
            value = getattr(output, key, None)
            if torch.is_tensor(value):
                return value
        raise RuntimeError("Unable to extract decoded audio from S1-DAC output.")

    @staticmethod
    def _normalize_waveform_shape(audio_values: torch.Tensor) -> torch.Tensor:
        # Normalize to [B, C, T] to match existing Mimi decode usage in this repo.
        if audio_values.ndim == 1:
            return audio_values.unsqueeze(0).unsqueeze(0)
        if audio_values.ndim == 2:
            return audio_values.unsqueeze(1)
        if audio_values.ndim == 3:
            if audio_values.shape[1] <= 2:
                return audio_values
            if audio_values.shape[2] <= 2:
                return audio_values.transpose(1, 2)
        return audio_values

    def decode(
        self,
        codes_bqt: torch.Tensor,
        global_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del global_codes
        last_error: Exception | None = None
        candidates = [
            {"audio_codes": codes_bqt},
            {"audio_codes": codes_bqt.transpose(1, 2)},
        ]
        for kwargs in candidates:
            try:
                out = self.model.decode(**kwargs)
                audio_values = self._extract_audio_values(out)
                return self._normalize_waveform_shape(audio_values)
            except TypeError as exc:
                last_error = exc
            except Exception as exc:  # pragma: no cover
                last_error = exc

        for value in (codes_bqt, codes_bqt.transpose(1, 2)):
            try:
                out = self.model.decode(value)
                audio_values = self._extract_audio_values(out)
                return self._normalize_waveform_shape(audio_values)
            except Exception as exc:  # pragma: no cover
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"S1-DAC decode failed: {last_error}") from last_error
        raise RuntimeError("S1-DAC decode failed with unknown error.")


class FishSpeechCodecAdapter(BaseCodecAdapter):
    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        codebook_size: int,
        max_codebooks: int,
        semantic_codebook_size: int = 4096,
    ) -> None:
        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            codebook_size=codebook_size,
            max_codebooks=max_codebooks,
        )
        self.semantic_codebook_size = int(max(semantic_codebook_size, 1))

    @staticmethod
    def _to_bct(input_values: torch.Tensor) -> torch.Tensor:
        audio = input_values
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(1)
        elif audio.ndim == 3 and audio.shape[1] != 1 and audio.shape[2] == 1:
            audio = audio.transpose(1, 2)
        if audio.ndim != 3:
            raise RuntimeError(
                f"Unsupported Fish Speech input shape: {tuple(input_values.shape)}"
            )
        return audio

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
        num_quantizers: int,
    ) -> SimpleNamespace:
        del padding_mask
        audio_bct = self._to_bct(input_values)
        audio_lengths = torch.full(
            (audio_bct.shape[0],),
            int(audio_bct.shape[-1]),
            dtype=torch.long,
            device=audio_bct.device,
        )
        indices, _indices_lens = self.model.encode(
            audio_bct,
            audio_lengths,
            n_quantizers=int(num_quantizers),
        )
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
        if indices.ndim != 3:
            raise RuntimeError(
                f"Unsupported Fish Speech code tensor shape: {tuple(indices.shape)}"
            )

        # Fish S1 emits semantic + residual codebooks; we train on the residual
        # stack to keep a uniform codebook size in tokenizer expansion.
        if indices.shape[1] >= int(num_quantizers) + 1:
            residual_codes = indices[:, 1 : 1 + int(num_quantizers), :]
        elif indices.shape[1] >= int(num_quantizers):
            residual_codes = indices[:, : int(num_quantizers), :]
        else:
            raise RuntimeError(
                f"Fish Speech produced {indices.shape[1]} codebooks, "
                f"need at least {num_quantizers}."
            )
        return SimpleNamespace(audio_codes=residual_codes.long())

    def decode(
        self,
        codes_bqt: torch.Tensor,
        global_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del global_codes
        if codes_bqt.ndim == 2:
            codes_bqt = codes_bqt.unsqueeze(0)
        if codes_bqt.ndim != 3:
            raise RuntimeError(
                f"Unsupported Fish Speech decode shape: {tuple(codes_bqt.shape)}"
            )
        bsz, _, t_len = codes_bqt.shape
        semantic = torch.zeros(
            (bsz, 1, t_len),
            dtype=torch.long,
            device=codes_bqt.device,
        )
        indices = torch.cat([semantic, codes_bqt.long()], dim=1)
        feature_lengths = torch.full(
            (bsz,),
            int(t_len),
            dtype=torch.long,
            device=codes_bqt.device,
        )
        audio_values, _audio_lengths = self.model.decode(indices, feature_lengths)
        return S1DacCodecAdapter._normalize_waveform_shape(audio_values)


class SparkBiCodecAdapter(BaseCodecAdapter):
    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        codebook_size: int,
        max_codebooks: int,
        global_codebook_size: int = 4096,
        global_token_count: int = 32,
    ) -> None:
        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            codebook_size=codebook_size,
            max_codebooks=max_codebooks,
        )
        self.config.global_codebook_size = int(max(global_codebook_size, 1))
        self.config.global_token_count = int(max(global_token_count, 1))

    @staticmethod
    def _normalize_wave_batch(input_values: torch.Tensor) -> torch.Tensor:
        wav = input_values
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 3 and wav.shape[1] == 1:
            wav = wav[:, 0, :]
        elif wav.ndim == 3 and wav.shape[2] == 1:
            wav = wav[:, :, 0]
        if wav.ndim != 2:
            raise RuntimeError(
                f"Unsupported Spark input shape: {tuple(input_values.shape)}"
            )
        return wav.float()

    def _build_ref_wav(self, wav_bt: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_ref_clip"):
            ref_rows: list[torch.Tensor] = []
            for i in range(int(wav_bt.shape[0])):
                ref_np = self.model.get_ref_clip(
                    wav_bt[i].detach().cpu().numpy().astype("float32")
                )
                ref_rows.append(torch.as_tensor(ref_np, dtype=torch.float32))
            return torch.stack(ref_rows, dim=0)
        return wav_bt

    @staticmethod
    def _normalize_semantic_tokens(tokens: torch.Tensor | Any) -> torch.Tensor:
        sem = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        if sem.ndim == 1:
            sem = sem.unsqueeze(0)
        if sem.ndim == 3:
            if sem.shape[1] == 1:
                sem = sem[:, 0, :]
            elif sem.shape[2] == 1:
                sem = sem[:, :, 0]
        if sem.ndim != 2:
            raise RuntimeError(f"Unsupported Spark semantic token shape: {tuple(sem.shape)}")
        return sem.long()

    @staticmethod
    def _normalize_global_tokens(tokens: torch.Tensor | Any) -> torch.Tensor:
        glob = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        if glob.ndim == 1:
            glob = glob.unsqueeze(0)
        if glob.ndim == 3:
            if glob.shape[1] == 1:
                glob = glob[:, 0, :]
            elif glob.shape[2] == 1:
                glob = glob[:, :, 0]
            else:
                glob = glob.reshape(glob.shape[0], -1)
        if glob.ndim != 2:
            raise RuntimeError(f"Unsupported Spark global token shape: {tuple(glob.shape)}")
        return glob.long()

    def _tokenize_batch(self, wav_bt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wav_bt = wav_bt.to(self.device)
        ref_wav = self._build_ref_wav(wav_bt).to(self.device)

        if hasattr(self.model, "tokenize_batch"):
            # Spark's tokenizer path expects a list of 1D float waveforms.
            wav_list = [
                wav_bt[i].detach().cpu().numpy().astype("float32")
                for i in range(int(wav_bt.shape[0]))
            ]
            batch = {"wav": wav_list, "ref_wav": ref_wav}
            outputs = self.model.tokenize_batch(batch)
            if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
                raise RuntimeError(
                    "Spark tokenize_batch must return (global_tokens, semantic_tokens)."
                )
            global_tokens, semantic_tokens = outputs[0], outputs[1]
        elif (
            hasattr(self.model, "extract_wav2vec2_features")
            and hasattr(self.model, "model")
            and hasattr(self.model.model, "tokenize")
        ):
            batch = {"wav": wav_bt, "ref_wav": ref_wav}
            batch["feat"] = self.model.extract_wav2vec2_features(batch["wav"])
            semantic_tokens, global_tokens = self.model.model.tokenize(batch)
        else:
            raise RuntimeError(
                "Spark tokenizer object does not expose tokenize_batch or model.tokenize."
            )

        semantic_bt = self._normalize_semantic_tokens(semantic_tokens)
        global_bt = self._normalize_global_tokens(global_tokens)
        return semantic_bt, global_bt

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
        num_quantizers: int,
    ) -> SimpleNamespace:
        del padding_mask
        if int(num_quantizers) != 1:
            raise RuntimeError(
                "Spark BiCodec exposes one semantic token stream. "
                f"Expected num_quantizers=1, got {num_quantizers}."
            )

        wav_bt = self._normalize_wave_batch(input_values)
        semantic_bt, global_bt = self._tokenize_batch(wav_bt)
        semantic_bqt = semantic_bt.unsqueeze(1)  # (B, 1, T)
        return SimpleNamespace(audio_codes=semantic_bqt, global_codes=global_bt)

    def decode(
        self,
        codes_bqt: torch.Tensor,
        global_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_codes is None:
            raise RuntimeError(
                "Spark BiCodec decode requires global_codes. "
                "Pass global prompt tokens during decode/inference."
            )

        if codes_bqt.ndim == 2:
            semantic_bt = codes_bqt.long()
        elif codes_bqt.ndim == 3:
            if codes_bqt.shape[1] < 1:
                raise RuntimeError(
                    f"Spark decode expected at least one stream, got {tuple(codes_bqt.shape)}"
                )
            semantic_bt = codes_bqt[:, 0, :].long()
        else:
            raise RuntimeError(
                f"Unsupported Spark semantic code shape: {tuple(codes_bqt.shape)}"
            )

        global_bt = self._normalize_global_tokens(global_codes)
        if global_bt.shape[0] != semantic_bt.shape[0]:
            if global_bt.shape[0] == 1:
                global_bt = global_bt.expand(semantic_bt.shape[0], -1)
            else:
                raise RuntimeError(
                    "Spark decode batch mismatch: "
                    f"semantic batch={semantic_bt.shape[0]} global batch={global_bt.shape[0]}"
                )

        semantic_bt = semantic_bt.to(self.device)
        global_bt = global_bt.to(self.device)
        if hasattr(self.model, "detokenize"):
            audio_np = self.model.detokenize(global_bt, semantic_bt)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "detokenize"):
            audio_np = self.model.model.detokenize(semantic_bt, global_bt.unsqueeze(1))
        else:
            raise RuntimeError(
                "Spark tokenizer object does not expose detokenize or model.detokenize."
            )
        audio_values = torch.as_tensor(audio_np, dtype=torch.float32, device=self.device)
        return S1DacCodecAdapter._normalize_waveform_shape(audio_values)


def _resolve_int(config: AudioCodec, attr: str, fallback: int) -> int:
    raw = int(getattr(config, attr))
    return raw if raw > 0 else int(fallback)


def _guess_spark_global_token_count(model: Any, fallback: int = 32) -> int:
    try:
        speaker_encoder = getattr(getattr(model, "model", model), "speaker_encoder", None)
        perceiver = getattr(speaker_encoder, "perceiver_sampler", None)
        latents = getattr(perceiver, "latents", None)
        if torch.is_tensor(latents) and latents.ndim >= 2:
            return int(latents.shape[0])
    except Exception:
        pass
    return int(fallback)


def _guess_spark_global_codebook_size(model: Any, fallback: int = 4096) -> int:
    try:
        speaker_encoder = getattr(getattr(model, "model", model), "speaker_encoder", None)
        quantizer = getattr(speaker_encoder, "quantizer", None)
        codebook_size = int(getattr(quantizer, "codebook_size", fallback))
        if codebook_size > 0:
            return codebook_size
    except Exception:
        pass
    return int(fallback)


def _build_mimi_codec(
    config: AudioCodec,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any]:
    from transformers import AutoFeatureExtractor, MimiModel

    model_ref = config.model_id.strip() or "kyutai/mimi"
    model = MimiModel.from_pretrained(model_ref).to(device)
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_ref)
    codebook_size = _resolve_int(
        config,
        "codebook_size_override",
        int(getattr(model.config, "codebook_size", 2048)),
    )
    default_max = int(
        getattr(
            model.config,
            "num_quantizers",
            getattr(model.config, "n_codebooks", 8),
        )
    )
    max_codebooks = _resolve_int(config, "max_codebooks", default_max)
    return (
        MimiCodecAdapter(
            model=model,
            feature_extractor=feature_extractor,
            codebook_size=codebook_size,
            max_codebooks=max_codebooks,
        ),
        feature_extractor,
    )


def _build_spark_bicodec_codec(
    config: AudioCodec,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any]:
    model_ref = (
        config.codec_ckpt_path.strip()
        or config.model_id.strip()
        or "/root/spark-tts/pretrained_models/Spark-TTS-0.5B"
    )

    spark_repo_candidates = [
        Path("/root/spark-tts"),
        Path("/root/Spark-TTS"),
        Path("/workspace/Spark-TTS"),
    ]
    env_repo_raw = os.environ.get("SPARK_TTS_REPO", "").strip()
    if env_repo_raw:
        env_repo = Path(env_repo_raw).expanduser()
        spark_repo_candidates.insert(0, env_repo)
    for repo in spark_repo_candidates:
        if repo.exists() and (repo / "sparktts").exists():
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            break

    try:
        spark_module = importlib.import_module("sparktts.models.audio_tokenizer")
        BiCodecTokenizer = getattr(spark_module, "BiCodecTokenizer")
    except Exception as exc:
        raise RuntimeError(
            "Spark BiCodec backend requires the Spark-TTS source package. "
            "Mount the Spark-TTS repo (containing `sparktts/`) and/or set SPARK_TTS_REPO."
        ) from exc

    model_dir = Path(model_ref).expanduser()
    if not model_dir.exists():
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                f"Spark model directory not found at {model_ref} and huggingface_hub unavailable."
            ) from exc
        local_dir = Path(
            tempfile.mkdtemp(prefix="spark_bicodec_repo_", dir="/tmp")
        ).resolve()
        model_dir = Path(
            snapshot_download(
                repo_id=model_ref,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        ).resolve()

    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    model = BiCodecTokenizer(model_dir=model_dir, device=device_obj)

    model_cfg = getattr(model, "config", {}) or {}
    sampling_rate = int(model_cfg.get("sample_rate", 16000))
    feature_extractor = SimpleFeatureExtractor(sampling_rate=sampling_rate)

    default_semantic_codebook_size = int(
        getattr(getattr(getattr(model, "model", None), "quantizer", None), "codebook_size", 4096)
    )
    codebook_size = _resolve_int(
        config,
        "codebook_size_override",
        default_semantic_codebook_size,
    )
    # Spark semantic stream is a single codebook sequence.
    max_codebooks = _resolve_int(config, "max_codebooks", 1)
    if max_codebooks > 1:
        max_codebooks = 1

    default_global_codebook_size = _guess_spark_global_codebook_size(model, fallback=4096)
    default_global_token_count = _guess_spark_global_token_count(model, fallback=32)
    return (
        SparkBiCodecAdapter(
            model=model,
            feature_extractor=feature_extractor,
            codebook_size=codebook_size,
            max_codebooks=max_codebooks,
            global_codebook_size=default_global_codebook_size,
            global_token_count=default_global_token_count,
        ),
        feature_extractor,
    )


def _build_s1_dac_codec(
    config: AudioCodec,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any]:
    from transformers import AutoFeatureExtractor, AutoModel

    def _retry_with_safetensors_alias() -> Any:
        from huggingface_hub import snapshot_download

        local_dir = Path(
            tempfile.mkdtemp(prefix="s1_dac_repo_", dir="/tmp")
        ).resolve()
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_ref,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        ).resolve()
        src = snapshot_path / "pytorch_model.safetensors"
        dst = snapshot_path / "model.safetensors"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        return AutoModel.from_pretrained(
            str(snapshot_path),
            trust_remote_code=trust_remote_code,
        )

    model_ref = (
        config.codec_ckpt_path.strip()
        or config.model_id.strip()
        or "jordand/fish-s1-dac-min"
    )

    trust_remote_code = bool(config.trust_remote_code)
    model = None
    last_error: Exception | None = None
    try:
        from transformers import DacModel

        model = DacModel.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        last_error = exc
        try:
            model = AutoModel.from_pretrained(
                model_ref,
                trust_remote_code=trust_remote_code,
            )
        except Exception as auto_exc:
            last_error = auto_exc
            try:
                # Some redistributed checkpoints expose only
                # `pytorch_model.safetensors` (without `model.safetensors`),
                # which older transformers builds may not auto-resolve.
                model = _retry_with_safetensors_alias()
            except Exception as alias_exc:
                raise RuntimeError(
                    f"Failed to load S1-DAC checkpoint from {model_ref}: {alias_exc}"
                ) from alias_exc
    model = model.to(device)
    model.eval()

    feature_extractor = None
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        sampling_rate = _resolve_int(
            config,
            "sample_rate_override",
            int(getattr(model.config, "sampling_rate", 44100)),
        )
        feature_extractor = SimpleFeatureExtractor(sampling_rate)

    codebook_size = _resolve_int(
        config,
        "codebook_size_override",
        int(getattr(model.config, "codebook_size", 2048)),
    )
    default_max = int(
        getattr(
            model.config,
            "n_codebooks",
            getattr(model.config, "num_codebooks", 10),
        )
    )
    max_codebooks = _resolve_int(config, "max_codebooks", default_max)

    if model is None and last_error is not None:  # pragma: no cover
        raise RuntimeError(f"Failed to load S1-DAC model: {last_error}") from last_error

    return (
        S1DacCodecAdapter(
            model=model,
            feature_extractor=feature_extractor,
            codebook_size=codebook_size,
            max_codebooks=max_codebooks,
        ),
        feature_extractor,
    )


def _build_s1_dac_official_fish_codec(
    config: AudioCodec,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any]:
    import sys

    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    fish_repo_root = Path("/root/fish-speech")
    if not fish_repo_root.exists():
        raise RuntimeError(
            "Official Fish codec backend requires /root/fish-speech to be mounted."
        )
    if str(fish_repo_root) not in sys.path:
        sys.path.insert(0, str(fish_repo_root))
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    model_ref = config.model_id.strip() or "fishaudio/openaudio-s1-mini"
    checkpoint_path = config.codec_ckpt_path.strip()
    checkpoint_is_safetensors = False
    if not checkpoint_path:
        try:
            checkpoint_path = hf_hub_download(repo_id=model_ref, filename="codec.pth")
        except Exception:
            checkpoint_path = hf_hub_download(
                repo_id=model_ref,
                filename="pytorch_model.safetensors",
            )
            checkpoint_is_safetensors = True
    else:
        checkpoint_is_safetensors = checkpoint_path.endswith(".safetensors")

    config_dir = fish_repo_root / "fish_speech" / "configs"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
    ):
        dac_cfg = compose(config_name="modded_dac_vq")
    model = instantiate(dac_cfg)
    if checkpoint_is_safetensors:
        from safetensors.torch import load_file as load_safetensors_file

        state_dict = load_safetensors_file(checkpoint_path)
    else:
        state_dict = torch.load(
            checkpoint_path,
            map_location=str(device),
            mmap=True,
            weights_only=True,
        )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in key for key in state_dict):
        state_dict = {
            key.replace("generator.", ""): value
            for key, value in state_dict.items()
            if "generator." in key
        }
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval().to(device)

    sampling_rate = _resolve_int(
        config,
        "sample_rate_override",
        int(getattr(model, "sample_rate", 44100)),
    )
    feature_extractor = SimpleFeatureExtractor(sampling_rate)

    residual_quantizer = getattr(getattr(model, "quantizer", None), "quantizer", None)
    semantic_quantizer = getattr(
        getattr(model, "quantizer", None), "semantic_quantizer", None
    )
    default_codebook_size = int(getattr(residual_quantizer, "codebook_size", 1024))
    default_max_codebooks = int(getattr(residual_quantizer, "n_codebooks", 9))
    semantic_codebook_size = int(
        getattr(semantic_quantizer, "codebook_size", 4096)
    )

    codebook_size = _resolve_int(
        config,
        "codebook_size_override",
        default_codebook_size,
    )
    max_codebooks = _resolve_int(config, "max_codebooks", default_max_codebooks)

    return (
        FishSpeechCodecAdapter(
            model=model,
            feature_extractor=feature_extractor,
            codebook_size=codebook_size,
            max_codebooks=max_codebooks,
            semantic_codebook_size=semantic_codebook_size,
        ),
        feature_extractor,
    )


def load_audio_codec(
    job_config: JobConfig,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any, CodecRuntimeInfo]:
    cfg = job_config.audio_codec
    adapter, feature_extractor, model_ref, backend, source = build_codec_from_registry(
        cfg, device
    )

    requested_q = int(job_config.model.num_quantizers)
    if requested_q > int(adapter.max_codebooks):
        raise ValueError(
            f"Requested model.num_quantizers={requested_q} exceeds codec max={adapter.max_codebooks} "
            f"for backend={backend}."
        )

    info = CodecRuntimeInfo(
        backend=backend,
        source=source,
        model_ref=model_ref,
        sampling_rate=int(adapter.sampling_rate),
        codebook_size=int(adapter.config.codebook_size),
        max_codebooks=int(adapter.max_codebooks),
        global_codebook_size=int(
            getattr(getattr(adapter, "config", None), "global_codebook_size", 0) or 0
        ),
        global_token_count=int(
            getattr(getattr(adapter, "config", None), "global_token_count", 0) or 0
        ),
    )
    return adapter, feature_extractor, info
