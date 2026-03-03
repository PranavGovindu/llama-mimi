from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from torchtitan.config_manager import AudioCodec, JobConfig


@dataclass
class CodecRuntimeInfo:
    backend: str
    source: str
    model_ref: str
    sampling_rate: int
    codebook_size: int
    max_codebooks: int


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

    def decode(self, codes_bqt: torch.Tensor) -> torch.Tensor:
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

    def decode(self, codes_bqt: torch.Tensor) -> torch.Tensor:
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

    def decode(self, codes_bqt: torch.Tensor) -> torch.Tensor:
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


def _resolve_int(config: AudioCodec, attr: str, fallback: int) -> int:
    raw = int(getattr(config, attr))
    return raw if raw > 0 else int(fallback)


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


def _build_s1_dac_codec(
    config: AudioCodec,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any]:
    from transformers import AutoFeatureExtractor, AutoModel

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
        model = AutoModel.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
        )
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


def load_audio_codec(
    job_config: JobConfig,
    device: torch.device | str,
) -> tuple[BaseCodecAdapter, Any, CodecRuntimeInfo]:
    cfg = job_config.audio_codec
    backend = cfg.backend.strip().lower()
    source = cfg.source.strip().lower()

    if backend == "mimi":
        adapter, feature_extractor = _build_mimi_codec(cfg, device)
        model_ref = cfg.model_id.strip() or "kyutai/mimi"
    elif backend == "s1_dac":
        adapter, feature_extractor = _build_s1_dac_codec(cfg, device)
        model_ref = cfg.codec_ckpt_path.strip() or cfg.model_id.strip() or "jordand/fish-s1-dac-min"
    else:
        raise ValueError(f"Unsupported audio codec backend: {cfg.backend}")

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
    )
    return adapter, feature_extractor, info
