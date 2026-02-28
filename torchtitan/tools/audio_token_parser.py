import re
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import LogitsProcessor


_AUDIO_TOKEN_RE = re.compile(r"^<(\d+)_(\d+)>$")


def build_audio_code_id_map(vocab: Mapping[str, int]) -> dict[int, tuple[int, int]]:
    code_id_map: dict[int, tuple[int, int]] = {}
    for token, token_id in vocab.items():
        match = _AUDIO_TOKEN_RE.match(token)
        if not match:
            continue
        code_id_map[token_id] = (int(match.group(1)), int(match.group(2)))
    return code_id_map


def filter_tokens_by_attention_mask(
    token_ids: Sequence[int], attention_mask: Sequence[int]
) -> list[int]:
    return [tok for tok, mask in zip(token_ids, attention_mask) if int(mask) == 1]


def get_audio_span_indices(
    token_ids: Sequence[int], audio_start_id: int | None, audio_end_id: int | None
) -> tuple[int, int]:
    start_idx = 0
    if audio_start_id is not None and audio_start_id in token_ids:
        start_idx = token_ids.index(audio_start_id) + 1

    end_idx = len(token_ids)
    if audio_end_id is not None:
        for i in range(start_idx, len(token_ids)):
            if token_ids[i] == audio_end_id:
                end_idx = i
                break

    return start_idx, end_idx


def extract_audio_codes_bqt_from_token_ids(
    token_ids: Sequence[int],
    num_quantizers: int,
    audio_code_id_map: Mapping[int, tuple[int, int]],
    audio_start_id: int | None = None,
    audio_end_id: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> torch.Tensor | None:
    if not audio_code_id_map:
        return None

    if start_idx is None or end_idx is None:
        start_idx, end_idx = get_audio_span_indices(
            token_ids, audio_start_id=audio_start_id, audio_end_id=audio_end_id
        )

    segment = token_ids[start_idx:end_idx]
    vals: list[int] = []
    expected_q = 0
    frame_vals: list[int] = []

    for tok_id in segment:
        mapped = audio_code_id_map.get(tok_id)
        if mapped is None:
            expected_q = 0
            frame_vals.clear()
            continue

        code, q_idx = mapped
        if q_idx == expected_q:
            frame_vals.append(code)
            expected_q += 1
            if expected_q == num_quantizers:
                vals.extend(frame_vals)
                frame_vals.clear()
                expected_q = 0
        elif q_idx == 0:
            frame_vals = [code]
            expected_q = 1
        else:
            expected_q = 0
            frame_vals.clear()

    if not vals:
        return None
    vals = vals[: len(vals) - len(vals) % num_quantizers]
    if not vals:
        return None

    tensor_btq = torch.tensor(vals).reshape(1, -1, num_quantizers)
    return tensor_btq.transpose(1, 2)  # (B, Q, T)


def normalize_waveform_for_logging(audio_np: np.ndarray) -> np.ndarray:
    if audio_np.ndim > 1:
        audio_np = audio_np[0]
    audio_np = np.ascontiguousarray(audio_np.reshape(-1).astype(np.float32))
    return np.clip(audio_np, -1.0, 1.0)


class AllowTokenIdsLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: Sequence[int]):
        # keep only unique non-negative ids
        uniq = sorted({int(i) for i in allowed_token_ids if int(i) >= 0})
        self.allowed_token_ids = torch.tensor(uniq, dtype=torch.long)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.allowed_token_ids.numel() == 0:
            return scores

        allowed = self.allowed_token_ids.to(scores.device)
        allowed = allowed[allowed < scores.size(1)]
        if allowed.numel() == 0:
            return scores

        filtered = torch.full_like(scores, torch.finfo(scores.dtype).min)
        filtered.index_copy_(1, allowed, scores.index_select(1, allowed))
        return filtered
