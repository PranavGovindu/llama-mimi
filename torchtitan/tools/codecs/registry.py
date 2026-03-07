from __future__ import annotations

from typing import Any

import torch

from torchtitan.config_manager import AudioCodec


def build_codec_from_registry(
    config: AudioCodec,
    device: torch.device | str,
) -> tuple[Any, Any, str, str, str]:
    """Resolve codec backend/source and return adapter, extractor, model_ref, backend, source.

    Return tuple format:
    - adapter
    - feature_extractor
    - model_ref
    - backend (normalized)
    - source (normalized)
    """

    # Imported lazily to avoid module import cycles.
    from torchtitan.tools.audio_codec import (
        _build_dualcodec_codec,
        _build_mimi_codec,
        _build_spark_bicodec_codec,
        _build_s1_dac_codec,
        _build_s1_dac_official_fish_codec,
    )

    backend = config.backend.strip().lower()
    source = config.source.strip().lower()

    if backend == "mimi":
        adapter, feature_extractor = _build_mimi_codec(config, device)
        model_ref = config.model_id.strip() or "kyutai/mimi"
        return adapter, feature_extractor, model_ref, backend, source

    if backend == "s1_dac":
        if source == "official_fish":
            adapter, feature_extractor = _build_s1_dac_official_fish_codec(
                config, device
            )
            model_ref = (
                config.codec_ckpt_path.strip()
                or config.model_id.strip()
                or "fishaudio/openaudio-s1-mini"
            )
            return adapter, feature_extractor, model_ref, backend, source

        adapter, feature_extractor = _build_s1_dac_codec(config, device)
        model_ref = (
            config.codec_ckpt_path.strip()
            or config.model_id.strip()
            or "jordand/fish-s1-dac-min"
        )
        return adapter, feature_extractor, model_ref, backend, source

    if backend == "spark_bicodec":
        adapter, feature_extractor = _build_spark_bicodec_codec(config, device)
        model_ref = (
            config.codec_ckpt_path.strip()
            or config.model_id.strip()
            or "/root/spark-tts/pretrained_models/Spark-TTS-0.5B"
        )
        return adapter, feature_extractor, model_ref, backend, source

    if backend == "dualcodec":
        adapter, feature_extractor = _build_dualcodec_codec(config, device)
        model_ref = config.model_id.strip() or "12hz_v1"
        return adapter, feature_extractor, model_ref, backend, source

    raise ValueError(f"Unsupported audio codec backend: {config.backend}")
