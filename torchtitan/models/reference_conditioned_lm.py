from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


class ReferenceConditioningEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        prefix_tokens: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_tokens = int(prefix_tokens)
        self.max_seq_len = int(max_seq_len)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max(1, self.max_seq_len), self.hidden_size)
        )
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=max(1, int(num_heads)),
            dim_feedforward=self.hidden_size * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.prefix_queries = nn.Parameter(
            torch.zeros(1, max(1, self.prefix_tokens), self.hidden_size)
        )
        nn.init.normal_(self.prefix_queries, mean=0.0, std=0.02)
        self.prefix_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.prefix_norm = nn.LayerNorm(self.hidden_size)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(
        self,
        ref_embeds: torch.Tensor,
        ref_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(ref_embeds.size(1))
        if seq_len == 0:
            batch = int(ref_embeds.size(0))
            device = ref_embeds.device
            dtype = ref_embeds.dtype
            return (
                torch.zeros(batch, self.prefix_tokens, self.hidden_size, device=device, dtype=dtype),
                torch.zeros(batch, self.hidden_size, device=device, dtype=dtype),
            )

        if seq_len > self.max_seq_len:
            ref_embeds = ref_embeds[:, : self.max_seq_len, :]
            ref_attention_mask = ref_attention_mask[:, : self.max_seq_len]
            seq_len = self.max_seq_len

        hidden = ref_embeds + self.position_embeddings[:, :seq_len, :].to(ref_embeds.dtype)
        key_padding_mask = ref_attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        pooled = _masked_mean(hidden, ref_attention_mask)

        prefix_queries = self.prefix_queries.expand(hidden.size(0), -1, -1).to(hidden.dtype)
        prefix_state, _ = self.prefix_attn(
            prefix_queries,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled_bias = self.global_proj(pooled).unsqueeze(1)
        prefix_state = self.prefix_norm(prefix_queries + prefix_state + pooled_bias)
        return prefix_state, pooled


class GroupedResidualHead(nn.Module):
    def __init__(self, hidden_size: int, num_quantizers: int, codebook_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(
                self.hidden_size,
                max(1, self.num_quantizers - 1) * self.codebook_size,
            ),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden)
        if self.num_quantizers <= 1:
            return logits.new_zeros(*hidden.shape[:-1], 0, self.codebook_size)
        return logits.view(*hidden.shape[:-1], self.num_quantizers - 1, self.codebook_size)


class ReferenceConditionedCausalLM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_quantizers: int,
        codebook_size: int,
        audio_code_id_map: dict[int, tuple[int, int]],
        audio_end_id: int | None,
        enable_reference_conditioning: bool,
        reference_seq_len: int,
        reference_conditioning_dropout: float,
        reference_conditioning_prefix_tokens: int,
        reference_encoder_layers: int,
        reference_encoder_heads: int,
        prediction_topology: str,
        grouped_residual_loss_weight: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.audio_end_id = None if audio_end_id is None else int(audio_end_id)
        self.enable_reference_conditioning = bool(enable_reference_conditioning)
        self.reference_seq_len = int(reference_seq_len)
        self.reference_conditioning_dropout = float(reference_conditioning_dropout)
        self.reference_conditioning_prefix_tokens = int(reference_conditioning_prefix_tokens)
        self.prediction_topology = str(prediction_topology).strip().lower()
        self.grouped_residual_loss_weight = float(grouped_residual_loss_weight)
        self.uses_reference_conditioning = self.enable_reference_conditioning

        hidden_size = int(self.base_model.config.hidden_size)
        vocab_size = int(self.base_model.get_input_embeddings().weight.shape[0])
        self.vocab_size = vocab_size

        token_to_q_index = torch.full((vocab_size,), -1, dtype=torch.long)
        token_to_code = torch.full((vocab_size,), -100, dtype=torch.long)
        token_id_table = torch.full(
            (max(1, self.num_quantizers), max(1, self.codebook_size)),
            -1,
            dtype=torch.long,
        )
        for token_id, (code, q_idx) in audio_code_id_map.items():
            if token_id < 0 or token_id >= vocab_size:
                continue
            if q_idx < 0 or q_idx >= self.num_quantizers:
                continue
            if code < 0 or code >= self.codebook_size:
                continue
            token_to_q_index[token_id] = int(q_idx)
            token_to_code[token_id] = int(code)
            token_id_table[q_idx, code] = int(token_id)
        q0_token_ids = token_id_table[0][token_id_table[0] >= 0]
        if self.audio_end_id is not None:
            q0_token_ids = torch.cat(
                [q0_token_ids, torch.tensor([self.audio_end_id], dtype=torch.long)]
            )
        self.register_buffer("token_to_q_index", token_to_q_index, persistent=False)
        self.register_buffer("token_to_code", token_to_code, persistent=False)
        self.register_buffer("token_id_table", token_id_table, persistent=False)
        self.register_buffer("q0_token_ids", q0_token_ids.unique(sorted=True), persistent=False)

        if self.enable_reference_conditioning:
            num_heads = max(1, min(int(reference_encoder_heads), hidden_size))
            while hidden_size % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            self.reference_encoder = ReferenceConditioningEncoder(
                hidden_size=hidden_size,
                prefix_tokens=self.reference_conditioning_prefix_tokens,
                num_layers=reference_encoder_layers,
                num_heads=num_heads,
                max_seq_len=self.reference_seq_len,
                dropout=0.1,
            )
            self.reference_global_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.reference_encoder = None
            self.reference_global_proj = None

        if self.prediction_topology == "grouped_residual":
            self.grouped_residual_head = GroupedResidualHead(
                hidden_size=hidden_size,
                num_quantizers=self.num_quantizers,
                codebook_size=self.codebook_size,
            )
        else:
            self.grouped_residual_head = None

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def _lookup_q_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        clamped = token_ids.clamp(min=0, max=self.token_to_q_index.numel() - 1)
        q_idx = self.token_to_q_index[clamped]
        return torch.where(token_ids >= 0, q_idx, torch.full_like(q_idx, -1))

    def _lookup_code_values(self, token_ids: torch.Tensor) -> torch.Tensor:
        clamped = token_ids.clamp(min=0, max=self.token_to_code.numel() - 1)
        code_vals = self.token_to_code[clamped]
        return torch.where(token_ids >= 0, code_vals, torch.full_like(code_vals, -100))

    def _build_reference_conditioning(
        self,
        ref_input_ids: torch.Tensor | None,
        ref_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.enable_reference_conditioning:
            return None, None, None
        if ref_input_ids is None or ref_attention_mask is None or self.reference_encoder is None:
            return None, None, None

        batch_size = int(ref_input_ids.size(0))
        hidden_size = int(self.base_model.config.hidden_size)
        prefix = self.base_model.get_input_embeddings().weight.new_zeros(
            batch_size,
            self.reference_conditioning_prefix_tokens,
            hidden_size,
        )
        prefix_mask = ref_attention_mask.new_zeros(
            batch_size,
            self.reference_conditioning_prefix_tokens,
        )
        pooled = self.base_model.get_input_embeddings().weight.new_zeros(batch_size, hidden_size)

        has_ref = ref_attention_mask.sum(dim=1) > 0
        if self.training and self.reference_conditioning_dropout > 0.0:
            keep_ref = (
                torch.rand(batch_size, device=ref_attention_mask.device)
                >= self.reference_conditioning_dropout
            )
            has_ref = has_ref & keep_ref

        if has_ref.any():
            ref_embeds = self.base_model.get_input_embeddings()(ref_input_ids[has_ref])
            prefix_valid, pooled_valid = self.reference_encoder(
                ref_embeds,
                ref_attention_mask[has_ref],
            )
            prefix[has_ref] = prefix_valid
            pooled[has_ref] = pooled_valid
            prefix_mask[has_ref] = 1

        return prefix, prefix_mask, pooled

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        ref_input_ids: torch.Tensor | None,
        ref_attention_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, int]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        prefix, prefix_mask, pooled = self._build_reference_conditioning(
            ref_input_ids,
            ref_attention_mask,
        )
        if prefix is None or prefix_mask is None or pooled is None:
            model_inputs: dict[str, torch.Tensor] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            return model_inputs, labels, 0

        token_embeds = self.base_model.get_input_embeddings()(input_ids)
        token_embeds = token_embeds + self.reference_global_proj(pooled).unsqueeze(1)
        model_inputs = {
            "inputs_embeds": torch.cat([prefix, token_embeds], dim=1),
            "attention_mask": torch.cat([prefix_mask, attention_mask], dim=1),
        }
        if labels is None:
            return model_inputs, None, int(prefix.size(1))

        prefix_labels = torch.full(
            (labels.size(0), prefix.size(1)),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return model_inputs, torch.cat([prefix_labels, labels], dim=1), int(prefix.size(1))

    @staticmethod
    def _compute_flat_loss(logits: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor | None:
        if labels is None:
            return None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    def _compute_grouped_backbone_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if labels is None:
            return None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        active = shift_labels.ne(-100)
        if not active.any():
            return shift_logits.sum() * 0.0

        q_idx = self._lookup_q_indices(shift_labels)
        mask = active & q_idx.eq(0)
        if self.audio_end_id is not None:
            mask = mask | active & shift_labels.eq(self.audio_end_id)
        if not mask.any():
            return shift_logits.sum() * 0.0
        return F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="mean")

    def _compute_grouped_residual_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if labels is None or self.grouped_residual_head is None or self.num_quantizers <= 1:
            return None
        seq_len = int(input_ids.size(1))
        if seq_len <= self.num_quantizers:
            return hidden_states.sum() * 0.0

        window = seq_len - (self.num_quantizers - 1)
        token_q = self._lookup_q_indices(input_ids)
        token_code = self._lookup_code_values(input_ids)
        base_hidden = hidden_states[:, :window, :]
        valid = token_q[:, :window].eq(0)
        targets: list[torch.Tensor] = []
        for offset in range(1, self.num_quantizers):
            q_slice = token_q[:, offset : offset + window]
            label_slice = labels[:, offset : offset + window]
            code_slice = token_code[:, offset : offset + window]
            valid = valid & label_slice.ne(-100) & q_slice.eq(offset) & code_slice.ge(0)
            targets.append(code_slice)

        if not valid.any():
            return base_hidden.sum() * 0.0

        target_codes = torch.stack(targets, dim=2)
        residual_logits = self.grouped_residual_head(base_hidden)
        residual_logits = residual_logits[valid]
        target_codes = target_codes[valid]
        return F.cross_entropy(
            residual_logits.reshape(-1, self.codebook_size),
            target_codes.reshape(-1),
            reduction="mean",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ref_input_ids: torch.Tensor | None = None,
        ref_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (
            not self.enable_reference_conditioning
            and self.prediction_topology == "flat"
            and ref_input_ids is None
        ):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        model_inputs, effective_labels, prefix_len = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ref_input_ids=ref_input_ids,
            ref_attention_mask=ref_attention_mask,
        )
        need_hidden_states = self.prediction_topology == "grouped_residual"
        outputs = self.base_model(
            **model_inputs,
            output_hidden_states=need_hidden_states,
            return_dict=True,
            **kwargs,
        )
        if self.prediction_topology == "grouped_residual":
            backbone_loss = self._compute_grouped_backbone_loss(outputs.logits, effective_labels)
            main_hidden = outputs.hidden_states[-1][:, prefix_len:, :]
            residual_loss = self._compute_grouped_residual_loss(
                hidden_states=main_hidden,
                input_ids=input_ids,
                labels=labels,
            )
            loss = None
            if backbone_loss is not None:
                loss = backbone_loss
                if residual_loss is not None:
                    loss = loss + self.grouped_residual_loss_weight * residual_loss
        else:
            loss = self._compute_flat_loss(outputs.logits, effective_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _restrict_scores(
        self,
        scores: torch.Tensor,
        allowed_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if allowed_token_ids is None or allowed_token_ids.numel() == 0:
            return scores
        allowed = allowed_token_ids.to(scores.device)
        allowed = allowed[(allowed >= 0) & (allowed < scores.size(-1))]
        if allowed.numel() == 0:
            return scores
        filtered = torch.full_like(scores, torch.finfo(scores.dtype).min)
        filtered.index_copy_(1, allowed, scores.index_select(1, allowed))
        return filtered

    @staticmethod
    def _sample_from_scores(
        scores: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        if not do_sample:
            return scores.argmax(dim=-1, keepdim=True)

        temp = float(max(temperature, 1e-5))
        sample_scores = scores / temp
        if top_k > 0 and top_k < sample_scores.size(-1):
            top_values, top_indices = torch.topk(sample_scores, top_k, dim=-1)
            probs = torch.softmax(top_values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return top_indices.gather(-1, sampled)
        probs = torch.softmax(sample_scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _apply_generation_processors(
        self,
        scores: torch.Tensor,
        generated_ids: torch.Tensor,
        *,
        eos_token_id: int | None,
        min_new_tokens: int,
        generated_count: int,
        logits_processor: Any,
        allowed_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self._restrict_scores(scores, allowed_token_ids)
        if eos_token_id is not None and generated_count < int(max(0, min_new_tokens)):
            if 0 <= int(eos_token_id) < scores.size(-1):
                scores[:, int(eos_token_id)] = torch.finfo(scores.dtype).min
        if logits_processor is not None:
            scores = logits_processor(generated_ids, scores)
        finite_mask = torch.isfinite(scores)
        if not finite_mask.any(dim=-1).all():
            fallback = scores.new_full(scores.shape, torch.finfo(scores.dtype).min)
            fallback[..., 0] = 0.0
            scores = torch.where(finite_mask, scores, fallback)
        return scores

    def _prime_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        ref_input_ids: torch.Tensor | None,
        ref_attention_mask: torch.Tensor | None,
        *,
        need_hidden_states: bool,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if self.enable_reference_conditioning and ref_input_ids is not None:
            model_inputs, _, _ = self._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
            )
            full_attention_mask = model_inputs["attention_mask"]
            outputs = self.base_model(
                **model_inputs,
                use_cache=True,
                output_hidden_states=need_hidden_states,
                return_dict=True,
            )
            return outputs, full_attention_mask

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        return outputs, attention_mask

    def _generate_flat(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        ref_input_ids: torch.Tensor | None,
        ref_attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        eos_token_id: int | None,
        logits_processor: Any,
        min_new_tokens: int,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return input_ids

        outputs, cache_attention_mask = self._prime_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ref_input_ids=ref_input_ids,
            ref_attention_mask=ref_attention_mask,
            need_hidden_states=False,
        )
        past_key_values = outputs.past_key_values
        emitted = input_ids
        next_scores = outputs.logits[:, -1, :]

        generated_count = 0
        while generated_count < max_new_tokens:
            next_scores = self._apply_generation_processors(
                next_scores,
                emitted,
                eos_token_id=eos_token_id,
                min_new_tokens=min_new_tokens,
                generated_count=generated_count,
                logits_processor=logits_processor,
            )
            next_token = self._sample_from_scores(
                next_scores,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )
            emitted = torch.cat([emitted, next_token], dim=1)
            generated_count += 1
            if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
                break
            cache_attention_mask = torch.cat(
                [cache_attention_mask, cache_attention_mask.new_ones((cache_attention_mask.size(0), 1))],
                dim=1,
            )
            outputs = self.base_model(
                input_ids=next_token,
                attention_mask=cache_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_scores = outputs.logits[:, -1, :]

        return emitted

    def _generate_grouped(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        ref_input_ids: torch.Tensor | None,
        ref_attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        eos_token_id: int | None,
        logits_processor: Any,
        min_new_tokens: int,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return input_ids

        outputs, cache_attention_mask = self._prime_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ref_input_ids=ref_input_ids,
            ref_attention_mask=ref_attention_mask,
            need_hidden_states=False,
        )
        past_key_values = outputs.past_key_values
        emitted = input_ids
        next_scores = outputs.logits[:, -1, :]

        generated_count = 0
        while generated_count < max_new_tokens:
            remaining = max_new_tokens - generated_count
            if remaining < self.num_quantizers and generated_count < int(max(0, min_new_tokens)):
                break
            allowed_q0 = self.q0_token_ids
            if eos_token_id is not None and remaining < self.num_quantizers:
                allowed_q0 = torch.tensor([int(eos_token_id)], device=next_scores.device)

            next_scores = self._apply_generation_processors(
                next_scores,
                emitted,
                eos_token_id=eos_token_id,
                min_new_tokens=min_new_tokens,
                generated_count=generated_count,
                logits_processor=logits_processor,
                allowed_token_ids=allowed_q0,
            )
            ar_token = self._sample_from_scores(
                next_scores,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )
            emitted = torch.cat([emitted, ar_token], dim=1)
            generated_count += 1
            if eos_token_id is not None and bool((ar_token == int(eos_token_id)).all()):
                break

            cache_attention_mask = torch.cat(
                [cache_attention_mask, cache_attention_mask.new_ones((cache_attention_mask.size(0), 1))],
                dim=1,
            )
            q0_out = self.base_model(
                input_ids=ar_token,
                attention_mask=cache_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = q0_out.past_key_values
            q0_hidden = q0_out.hidden_states[-1][:, -1, :]

            residual_logits = self.grouped_residual_head(q0_hidden.unsqueeze(1)).squeeze(1)
            residual_tokens: list[torch.Tensor] = []
            for q_offset in range(self.num_quantizers - 1):
                code_scores = residual_logits[:, q_offset, :]
                token_ids = self.token_id_table[q_offset + 1].to(code_scores.device)
                valid = token_ids >= 0
                if not valid.all():
                    mask = torch.full_like(code_scores, torch.finfo(code_scores.dtype).min)
                    mask[:, valid] = code_scores[:, valid]
                    code_scores = mask
                sampled_code = self._sample_from_scores(
                    code_scores,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                )
                sampled_token = token_ids.index_select(0, sampled_code.squeeze(-1)).unsqueeze(1)
                residual_tokens.append(sampled_token)

            residual_ids = torch.cat(residual_tokens, dim=1)
            emitted = torch.cat([emitted, residual_ids], dim=1)
            generated_count += residual_ids.size(1)
            cache_attention_mask = torch.cat(
                [
                    cache_attention_mask,
                    cache_attention_mask.new_ones(
                        (cache_attention_mask.size(0), residual_ids.size(1))
                    ),
                ],
                dim=1,
            )
            residual_out = self.base_model(
                input_ids=residual_ids,
                attention_mask=cache_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = residual_out.past_key_values
            next_scores = residual_out.logits[:, -1, :]

        return emitted[:, : input_ids.size(1) + max_new_tokens]

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        ref_input_ids: torch.Tensor | None = None,
        ref_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        logits_processor: Any = None,
        min_new_tokens: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids.size(0) != 1 and (
            self.prediction_topology == "grouped_residual"
            or self.enable_reference_conditioning
        ):
            raise NotImplementedError(
                "Custom cloning-aware generation currently supports batch_size=1."
            )
        if (
            self.prediction_topology == "flat"
            and (not self.enable_reference_conditioning or ref_input_ids is None)
        ):
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                logits_processor=logits_processor,
                min_new_tokens=min_new_tokens,
                **kwargs,
            )

        if self.prediction_topology == "grouped_residual":
            return self._generate_grouped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                logits_processor=logits_processor,
                min_new_tokens=min_new_tokens,
            )

        return self._generate_flat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ref_input_ids=ref_input_ids,
            ref_attention_mask=ref_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
            min_new_tokens=min_new_tokens,
        )
