# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
import torch.nn as nn

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderStopIteration
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.datasets.hf_datasets import build_hf_validation_dataloader
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.models.reference_conditioned_lm import (
    ReferenceConditionedCausalLM,
)
from torchtitan.tools.audio_token_parser import (
    AllowTokenIdsLogitsProcessor,
    build_audio_code_id_map,
    build_spark_global_id_map,
    extract_spark_global_token_ids,
    extract_audio_codes_bqt_from_token_ids,
    filter_tokens_by_attention_mask,
    get_audio_span_indices,
    normalize_waveform_for_logging,
)
from torchtitan.tools.audio_codec import CodecRuntimeInfo, load_audio_codec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.tools.research_eval import summarize_full_eval_rows
from torchtitan.tools.text_norm import extract_language_token, normalize_text_for_eval


def expand_tokenizer_with_unit_tokens(
    tokenizer,
    codebook_size=2048,
    num_quantizers=8,
    language_codes: Optional[list[str]] = None,
    codec_backend: str = "mimi",
    spark_global_codebook_size: int = 4096,
):
    """
    Expand tokenizer vocabulary for codec token streams.
    """
    backend = codec_backend.strip().lower()
    if backend == "spark_bicodec":
        new_tokens = [f"<|bicodec_semantic_{x}|>" for x in range(int(codebook_size))]
        new_tokens.extend(
            [
                f"<|bicodec_global_{x}|>"
                for x in range(int(max(spark_global_codebook_size, 1)))
            ]
        )
        new_tokens.extend(
            [
                "<|task_tts|>",
                "<|start_content|>",
                "<|end_content|>",
                "<|start_global_token|>",
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                "<|end_semantic_token|>",
                "<|start_style_label|>",
                "<|end_style_label|>",
            ]
        )
    else:
        new_tokens = [
            f"<{x}_{i}>" for x in range(codebook_size) for i in range(num_quantizers)
        ]
    existing_tokens = set(tokenizer.get_vocab().keys())
    added_tokens = [tok for tok in new_tokens if tok not in existing_tokens]
    tokenizer.add_tokens(added_tokens)
    # add <audio> and </audio> tokens
    special_tokens = ["<audio>", "</audio>"]
    if language_codes:
        special_tokens.extend(
            [f"<lang_{lang.lower().replace('-', '_')}>" for lang in language_codes]
        )
    tokenizer.add_tokens(special_tokens)
    return tokenizer


def get_nparams_and_flops(model, seq_len: int) -> (int, int):
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )

    config = model.config

    try:
        l = config.num_hidden_layers
        h = config.num_attention_heads
        d = config.hidden_size
    except AttributeError:
        raise ValueError("Model configuration does not have the required attributes.")

    q = d // h  # Query dimension per head
    t = seq_len
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    return nparams, num_flops_per_token


def load_causal_lm_with_fallback(model_name: str, pretrained: bool):
    from transformers import AutoConfig, AutoModelForCausalLM

    preferred_attn = "flash_attention_2"
    fallback_attn = "sdpa"
    try:
        if pretrained:
            return AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=preferred_attn
            )
        config = AutoConfig.from_pretrained(model_name)
        return AutoModelForCausalLM.from_config(
            config, attn_implementation=preferred_attn
        )
    except Exception as e:
        logger.warning(
            f"Failed to load model with {preferred_attn} ({e}); falling back to {fallback_attn}."
        )
        if pretrained:
            return AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=fallback_attn
            )
        config = AutoConfig.from_pretrained(model_name)
        return AutoModelForCausalLM.from_config(
            config, attn_implementation=fallback_attn
        )


def _safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    if not slug:
        return "unknown"
    return slug[:max_len]


def _edit_distance(seq_a: list[Any], seq_b: list[Any]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        cur = [i]
        for j, b in enumerate(seq_b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if a == b else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    # core configs
    job_config: JobConfig
    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec

    # swappable training components in TrainSpec
    dataloader: train_spec_module.BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer
    validator: train_spec_module.BaseValidator
    metrics_processor: train_spec_module.MetricsProcessor

    # non-swappable training components
    checkpointer: CheckpointManager
    ft_manager: FTManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Generator[None, None, None]
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    epoch: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config
        self.loss_ema: Optional[float] = None
        self.grad_norm_ema: Optional[float] = None

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        dist_utils.init_distributed(job_config)
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            world_size=world_size,
        )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallel training path is currently unsupported in this fork."
            )

        world_mesh = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.ft_manager = FTManager(job_config.fault_tolerance)
        dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = train_spec_module.get_train_spec("llama3")

        from transformers import AutoTokenizer

        # build dataloader
        tokenizer = AutoTokenizer.from_pretrained(
            job_config.model.name,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})

        audio_tokenizer, feature_extractor, codec_info = load_audio_codec(
            job_config, self.device
        )
        tokenizer = expand_tokenizer_with_unit_tokens(
        tokenizer,
        codebook_size=codec_info.codebook_size,
        num_quantizers=job_config.model.num_quantizers,
        language_codes=(
            job_config.training.languages
            if job_config.training.language_tokens
            else None
        ),
        codec_backend=codec_info.backend,
        spark_global_codebook_size=(
            codec_info.global_codebook_size
            if codec_info.global_codebook_size > 0
            else codec_info.codebook_size
        ),
    )

        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            feature_extractor=feature_extractor,
            job_config=job_config,
        )
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.feature_extractor = feature_extractor
        self.codec_info: CodecRuntimeInfo | None = codec_info
        logger.info(
            "Audio codec initialized: "
            f"backend={codec_info.backend} source={codec_info.source} "
            f"model_ref={codec_info.model_ref} sr={codec_info.sampling_rate} "
            f"codebook_size={codec_info.codebook_size} "
            f"max_codebooks={codec_info.max_codebooks}"
        )
        vocab = tokenizer.get_vocab()
        self.audio_start_id = vocab.get("<audio>")
        self.audio_end_id = vocab.get("</audio>")
        self.audio_code_id_map = build_audio_code_id_map(vocab)
        self.spark_global_id_map = build_spark_global_id_map(vocab)
        self.spark_global_start_id = vocab.get("<|start_global_token|>")
        self.spark_global_end_id = vocab.get("<|end_global_token|>")
        self.audio_generation_token_ids = sorted(self.audio_code_id_map.keys())
        if self.audio_end_id is not None:
            self.audio_generation_token_ids.append(self.audio_end_id)
        self.generated_audio_unconstrained_seen = False
        self.artifact_root = Path(
            job_config.artifact.dump_root or job_config.job.dump_folder
        )
        self.local_sample_artifacts_dir = self.artifact_root / "sample_artifacts"
        self.run_snapshot_dir = self.artifact_root / "run_snapshot"
        self.asr_pipeline = None
        self.overfit_gate_consecutive_passes = 0
        self.overfit_gate_passed = False
        self.latest_overfit_gate_metrics: dict[str, Any] = {}
        self.full_eval_enabled = False
        self.fixed_preview_batch: dict[str, torch.Tensor] | None = None

        # set the model args from training job configs
        # model_args.update_from_config(job_config, tokenizer)

        logger.info(f"Building {self.train_spec.name}")
        # with torch.device("meta"):
        #     model = self.train_spec.model_cls(model_args)
        model = load_causal_lm_with_fallback(
            job_config.model.name,
            job_config.model.pretrained,
        )

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            # model.tie_weights()

        if (
            job_config.training.enable_reference_conditioning
            or job_config.training.prediction_topology == "grouped_residual"
        ):
            model = ReferenceConditionedCausalLM(
                model,
                num_quantizers=job_config.model.num_quantizers,
                codebook_size=codec_info.codebook_size,
                audio_code_id_map=self.audio_code_id_map,
                audio_end_id=self.audio_end_id,
                enable_reference_conditioning=job_config.training.enable_reference_conditioning,
                reference_seq_len=job_config.training.reference_seq_len,
                reference_conditioning_dropout=job_config.training.reference_conditioning_dropout,
                reference_conditioning_prefix_tokens=job_config.training.reference_conditioning_prefix_tokens,
                reference_encoder_layers=job_config.training.reference_encoder_layers,
                reference_encoder_heads=job_config.training.reference_encoder_heads,
                prediction_topology=job_config.training.prediction_topology,
                grouped_residual_loss_weight=job_config.training.grouped_residual_loss_weight,
            )

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(job_config, parallel_dims)
        color = self.metrics_processor.color
        if torch.distributed.get_rank() == 0:
            self.run_snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._save_run_snapshot()

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(job_config)

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)

            # model.to_empty(device=init_device)
            # with torch.no_grad():
            #     model.init_weights(buffer_device=buffer_device)
            # model.train()
            model.to(device=init_device)
            # with torch.no_grad():
            #     model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config, parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.epoch = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            job_config=job_config,
            sd_adapter=self.train_spec.state_dict_adapter,
            ft_manager=self.ft_manager,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if job_config.validation.enabled:
            assert self.train_spec.build_validator_fn is not None
            assert not parallel_dims.pp_enabled, (
                "pp is enabled but validation doesn't support pipeline parallelism yet"
            )

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=tokenizer,
                audio_tokenizer=audio_tokenizer,
                feature_extractor=feature_extractor,
                parallel_dims=parallel_dims,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
            )

        if (
            job_config.tts_eval.full_pack_enabled
            and torch.distributed.get_rank() == 0
        ):
            self.full_eval_enabled = True

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})."
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderStopIteration() from ex
            data_load_start = time.perf_counter()
            self.metrics_processor.ntokens_since_last_log += batch["input_ids"].numel()
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            yield batch

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        input_ids = input_dict["input_ids"].to(self.device)
        attention_mask = input_dict["attention_mask"].to(self.device)
        labels = input_dict.get("labels", input_ids).to(self.device)
        ref_input_ids = input_dict.get("ref_input_ids")
        if ref_input_ids is not None:
            ref_input_ids = ref_input_ids.to(self.device)
        ref_attention_mask = input_dict.get("ref_attention_mask")
        if ref_attention_mask is not None:
            ref_attention_mask = ref_attention_mask.to(self.device)

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        # inputs = input_dict["input"]
        # optional_context_parallel_ctx = (
        #     dist_utils.create_context_parallel_ctx(
        #         cp_mesh=self.world_mesh["cp"],
        #         cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
        #         cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        #         cp_no_restore_buffers={inputs, labels},
        #         cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
        #     )
        #     if parallel_dims.cp_enabled
        #     else None
        # )
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=self.world_mesh["cp"],
                cp_buffers=[input_ids],
                cp_seq_dims=[1],
                cp_no_restore_buffers={input_ids},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )
        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                # with self.maybe_enable_amp:
                #     pred = model_parts[0](inputs)
                #     loss = self.loss_fn(pred, labels)
                # # need to free to before bwd to avoid peaking memory
                # del pred
                # loss.backward()
                with self.maybe_enable_amp:
                    outputs = model_parts[0](
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        ref_input_ids=ref_input_ids,
                        ref_attention_mask=ref_attention_mask,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps

                del outputs
                loss.backward()

        return loss

    def _write_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _run_git_command(self, args: list[str]) -> str:
        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=20,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _save_run_snapshot(self) -> None:
        if not self.job_config.artifact.save_git_snapshot:
            return
        if torch.distributed.get_rank() != 0:
            return

        snapshot_dir = self.run_snapshot_dir
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Store a fully resolved runtime config for exact reproducibility.
        self._write_json_file(snapshot_dir / "resolved_config.json", self.job_config.to_dict())
        self._write_json_file(
            snapshot_dir / "manifest.json",
            {
                "experiment_id": self.job_config.experiment.id,
                "campaign_id": self.job_config.experiment.campaign_id,
                "phase": self.job_config.experiment.phase,
                "track": self.job_config.experiment.track,
                "axis": self.job_config.experiment.axis,
                "family": self.job_config.experiment.family,
                "stage": self.job_config.experiment.stage,
                "variant": self.job_config.experiment.variant,
                "question": self.job_config.experiment.question,
                "hypothesis": self.job_config.experiment.hypothesis,
                "brief_path": self.job_config.experiment.brief_path,
                "baseline_experiment_id": self.job_config.experiment.baseline_experiment_id,
                "owner": self.job_config.experiment.owner,
                "tags": self.job_config.experiment.tags,
                "description": self.job_config.job.description,
                "created_at_unix": int(time.time()),
            },
        )
        self._write_json_file(
            snapshot_dir / "run_ids.json",
            {
                "wandb_run_id": (
                    getattr(getattr(self.metrics_processor.logger, "wandb", None), "run", None).id
                    if getattr(self.metrics_processor.logger, "wandb", None) is not None
                    and getattr(getattr(self.metrics_processor.logger, "wandb", None), "run", None)
                    is not None
                    else ""
                ),
                "experiment_id": self.job_config.experiment.id,
                "modal_app_id": os.environ.get("MODAL_APP_ID", ""),
            },
        )

        git_sha = self._run_git_command(["git", "rev-parse", "HEAD"])
        git_status = self._run_git_command(["git", "status", "--short"])
        git_branch = self._run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        git_diff = self._run_git_command(["git", "diff"])
        git_untracked = self._run_git_command(
            ["git", "ls-files", "--others", "--exclude-standard"]
        )

        self._write_json_file(
            snapshot_dir / "git.json",
            {
                "sha": git_sha,
                "branch": git_branch,
                "dirty": bool(git_status),
                "status_short": git_status,
                "untracked_files": git_untracked.splitlines() if git_untracked else [],
            },
        )
        if git_diff:
            (snapshot_dir / "git_diff.patch").write_text(git_diff + "\n")
        (snapshot_dir / "git_sha.txt").write_text((git_sha or "") + "\n")
        (snapshot_dir / "git_status.txt").write_text((git_status or "") + "\n")
        (snapshot_dir / "untracked_files.txt").write_text((git_untracked or "") + "\n")

        self._write_json_file(
            snapshot_dir / "env.json",
            {
                "python": ".".join(str(x) for x in tuple(sys.version_info[:3])),
                "torch": getattr(torch, "__version__", ""),
                "cuda_available": torch.cuda.is_available(),
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
                "cwd": os.getcwd(),
                "artifact_root": str(self.artifact_root),
                "dump_folder": self.job_config.job.dump_folder,
                "dataset_path": self.job_config.training.dataset_path,
                "num_quantizers": self.job_config.model.num_quantizers,
            },
        )

        dataset_path = self.job_config.training.dataset_path
        if dataset_path:
            dataset_root = Path(dataset_path)
            for src_name, dst_name in (
                ("dataset_manifest.json", "dataset_manifest.json"),
                ("metadata_single_sample.json", "dataset_single_sample.json"),
            ):
                src_path = dataset_root / src_name
                if src_path.exists():
                    try:
                        shutil.copy2(src_path, snapshot_dir / dst_name)
                    except OSError as exc:
                        logger.warning(
                            f"Failed to copy dataset metadata '{src_path}' into run snapshot: {exc}"
                        )

    def _normalize_text_for_eval(self, text: str, lang_hint: str = "") -> str:
        return normalize_text_for_eval(text, lang_hint=lang_hint)

    def _compute_wer_cer(
        self, prediction: str, reference: str, lang_hint: str = ""
    ) -> tuple[float, float]:
        pred_norm = self._normalize_text_for_eval(prediction, lang_hint=lang_hint)
        ref_norm = self._normalize_text_for_eval(reference, lang_hint=lang_hint)
        pred_words = pred_norm.split()
        ref_words = ref_norm.split()
        wer = (
            float(_edit_distance(pred_words, ref_words)) / float(max(len(ref_words), 1))
            if self.job_config.tts_eval.compute_wer
            else float("nan")
        )
        cer = (
            float(_edit_distance(list(pred_norm), list(ref_norm)))
            / float(max(len(ref_norm), 1))
            if self.job_config.tts_eval.compute_cer
            else float("nan")
        )
        return wer, cer

    def _maybe_build_asr_pipeline(self):
        if self.asr_pipeline is not None:
            return self.asr_pipeline
        if not self.job_config.tts_eval.enabled:
            return None
        try:
            from transformers import pipeline

            self.asr_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.job_config.tts_eval.asr_model_id,
                device=-1,
            )
            return self.asr_pipeline
        except Exception as e:
            logger.warning(f"ASR pipeline init failed; disabling TTS eval for run: {e}")
            self.asr_pipeline = False
            return None

    def _transcribe_audio(self, audio_np: np.ndarray, sample_rate: int) -> str:
        asr = self._maybe_build_asr_pipeline()
        if not asr:
            return ""
        try:
            result = asr({"array": audio_np, "sampling_rate": sample_rate})
            if isinstance(result, dict):
                return str(result.get("text", "")).strip()
            return str(result).strip()
        except Exception as e:
            logger.warning(f"ASR transcription failed at step={self.step}: {e}")
            return ""

    def _get_codebook_stats(self, codes_bqt: torch.Tensor | None) -> dict[str, float]:
        if codes_bqt is None or codes_bqt.ndim != 3:
            return {"frames": 0.0, "coverage_total": 0.0}
        codes_qt = codes_bqt.detach().cpu().to(torch.int64)[0]
        codebook_size = int(self.audio_tokenizer.config.codebook_size)
        frames = int(codes_qt.shape[1]) if codes_qt.ndim == 2 else 0
        unique_total = int(torch.unique(codes_qt).numel()) if frames > 0 else 0
        coverage_total = float(unique_total) / float(max(codebook_size, 1))
        return {"frames": float(frames), "coverage_total": coverage_total}

    def _get_codebook_coverage_per_q(self, codes_bqt: torch.Tensor | None) -> list[float]:
        if codes_bqt is None or codes_bqt.ndim != 3:
            return []
        codes_qt = codes_bqt.detach().cpu().to(torch.int64)[0]
        if codes_qt.ndim != 2 or codes_qt.numel() == 0:
            return []
        codebook_size = int(self.audio_tokenizer.config.codebook_size)
        denom = float(max(codebook_size, 1))
        return [
            float(torch.unique(codes_qt[q_idx]).numel()) / denom
            for q_idx in range(int(codes_qt.shape[0]))
        ]

    def _save_local_audio_artifact(
        self,
        prefix: str,
        sample_idx: int,
        audio_np: np.ndarray,
        sample_rate: int,
    ) -> None:
        if torch.distributed.get_rank() != 0:
            return
        if not self.job_config.artifact.save_local_audio_wav:
            return
        step_dir = (
            self.local_sample_artifacts_dir
            / f"step_{self.step:06d}"
            / f"sample_{sample_idx}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        wav_path = step_dir / f"{prefix}_audio.wav"
        try:
            import soundfile as sf

            sf.write(wav_path, audio_np, sample_rate)
        except Exception as e:
            logger.warning(f"Failed to save local WAV artifact {wav_path}: {e}")

    def _save_local_metrics_json(
        self, sample_idx: int, prefix: str, payload: dict[str, Any]
    ) -> None:
        if torch.distributed.get_rank() != 0:
            return
        step_dir = (
            self.local_sample_artifacts_dir
            / f"step_{self.step:06d}"
            / f"sample_{sample_idx}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_file(step_dir / f"{prefix}_metrics.json", payload)

    def _save_audio_artifact_to_dir(
        self,
        output_dir: Path,
        prefix: str,
        audio_np: np.ndarray,
        sample_rate: int,
    ) -> None:
        if torch.distributed.get_rank() != 0:
            return
        if not self.job_config.artifact.save_local_audio_wav:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / f"{prefix}_audio.wav"
        try:
            import soundfile as sf

            sf.write(wav_path, audio_np, sample_rate)
        except Exception as e:
            logger.warning(f"Failed to save local WAV artifact {wav_path}: {e}")

    def _build_full_eval_step_dir(self) -> Path:
        return self.artifact_root / "full_eval" / f"step_{self.step:06d}"

    def _load_json_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _maybe_log_full_eval_summary(self, summary: dict[str, Any]) -> None:
        if torch.distributed.get_rank() != 0 or not summary:
            return
        metric_mapping = {
            "sample_count": "full_eval/sample_count",
            "generated_count": "full_eval/generated_count",
            "wer_mean": "full_eval/wer_mean",
            "cer_mean": "full_eval/cer_mean",
            "utmos_mean": "full_eval/utmos_mean",
            "dnsmos_p808_mean": "full_eval/dnsmos_p808_mean",
            "dnsmos_ovr_mean": "full_eval/dnsmos_ovr_mean",
            "speaker_similarity_mean": "full_eval/speaker_similarity_mean",
            "salmon_mean": "full_eval/salmon_mean",
            "mel_l1_mean": "full_eval/mel_l1_mean",
            "mel_l2_mean": "full_eval/mel_l2_mean",
            "mel_cosine_mean": "full_eval/mel_cosine_mean",
            "frame_ratio_mean": "full_eval/frame_ratio_mean",
            "malformed_decode_rate": "full_eval/malformed_decode_rate",
            "coverage_q_min_mean": "full_eval/coverage_q_min_mean",
            "coverage_q_abs_diff_max_mean": "full_eval/coverage_q_abs_diff_max_mean",
        }
        metrics: dict[str, Any] = {}
        for src_key, dst_key in metric_mapping.items():
            value = summary.get(src_key)
            if value is None:
                continue
            metrics[dst_key] = value
        if metrics:
            self.metrics_processor.logger.log(metrics, self.step)

    def _maybe_run_rich_full_eval_scoring(self, step_dir: Path) -> dict[str, Any]:
        if torch.distributed.get_rank() != 0:
            return {}
        repo_root = Path(__file__).resolve().parents[1]
        summary_path = step_dir / "summary.json"
        env = dict(os.environ)
        score_cmd = [
            sys.executable,
            "scripts/research/score_tts_eval.py",
            "--run-dir",
            str(self.artifact_root),
            "--step",
            str(self.step),
            "--enable-utmos",
            "--progress-every",
            "10",
        ]
        logger.info(
            "Running rich full-pack scoring at step=%s for %s.",
            self.step,
            step_dir.name,
        )
        subprocess.run(
            score_cmd,
            check=False,
            cwd=repo_root,
            env=env,
        )

        summary = self._load_json_file(summary_path)
        if summary:
            logger.info(
                "Full-pack eval summary at step=%s: WER=%s CER=%s UTMOS=%s DNSMOS_OVR=%s MEL_L1=%s SALMon=%s",
                self.step,
                summary.get("wer_mean"),
                summary.get("cer_mean"),
                summary.get("utmos_mean"),
                summary.get("dnsmos_ovr_mean"),
                summary.get("mel_l1_mean"),
                summary.get("salmon_mean"),
            )
            self._maybe_log_full_eval_summary(summary)
        return summary

    def _resolve_full_pack_eval_every(self) -> int:
        tts_eval_cfg = self.job_config.tts_eval
        if tts_eval_cfg.full_pack_eval_every > 0:
            return tts_eval_cfg.full_pack_eval_every
        if tts_eval_cfg.eval_every > 0:
            return tts_eval_cfg.eval_every
        return self.job_config.training.sample_generate_every

    def _extract_audio_codes_bqt_from_ids(
        self,
        token_ids: list[int],
        num_quantizers: int,
        audio_start_idx: int | None = None,
        audio_end_idx: int | None = None,
    ) -> torch.Tensor | None:
        start_idx = audio_start_idx + 1 if audio_start_idx is not None else None
        end_idx = audio_end_idx if audio_end_idx is not None else None
        return extract_audio_codes_bqt_from_token_ids(
            token_ids=token_ids,
            num_quantizers=num_quantizers,
            audio_code_id_map=self.audio_code_id_map,
            audio_start_id=self.audio_start_id,
            audio_end_id=self.audio_end_id,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def _extract_global_codes_from_ids(
        self,
        token_ids: list[int],
    ) -> torch.Tensor | None:
        if self.job_config.audio_codec.backend.strip().lower() != "spark_bicodec":
            return None
        return extract_spark_global_token_ids(
            token_ids=token_ids,
            spark_global_id_map=self.spark_global_id_map,
            start_global_id=self.spark_global_start_id,
            end_global_id=self.spark_global_end_id,
        )

    def _decode_audio_numpy(
        self,
        codes_bqt: torch.Tensor,
        global_codes: torch.Tensor | None = None,
    ) -> Any:
        audio_values = self.audio_tokenizer.decode(
            codes_bqt.to(self.device),
            global_codes=(
                global_codes.to(self.device)
                if global_codes is not None
                else None
            ),
        )[0]
        return normalize_waveform_for_logging(audio_values.detach().float().cpu().numpy())

    def _clean_utterance_text(self, text: str, max_chars: int = 240) -> str:
        text = text.replace("<audio>", " ").replace("</audio>", " ")
        text = re.sub(r"<lang_[^>]+>", " ", text)
        text = re.sub(r"<\d+_\d+>", " ", text)
        text = re.sub(r"<\|bicodec_semantic_\d+\|>", " ", text)
        text = re.sub(r"<\|bicodec_global_\d+\|>", " ", text)
        text = re.sub(r"<\|[^|>]+\|>", " ", text)
        text = " ".join(text.split())
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    def _decode_utterance_from_token_ids(self, token_ids: list[int]) -> str:
        cut_idx = len(token_ids)
        if self.audio_start_id is not None and self.audio_start_id in token_ids:
            cut_idx = token_ids.index(self.audio_start_id)
        text = self.tokenizer.decode(token_ids[:cut_idx], skip_special_tokens=False)
        return self._clean_utterance_text(text)

    def _resolve_sample_generate_max_new_tokens(
        self, audio_span_start: int, audio_span_end: int
    ) -> int:
        cfg = self.job_config.training
        base_max = max(int(cfg.sample_generate_max_new_tokens), 1)

        if audio_span_end <= audio_span_start:
            return min(base_max, 2048)

        target_audio_tokens = max(audio_span_end - audio_span_start, 0)
        dynamic_cap = int(round(target_audio_tokens * 1.35 + 64))
        dynamic_cap = max(256, min(dynamic_cap, 3072))
        return min(base_max, dynamic_cap)

    def _clone_preview_batch(
        self, batch: dict[str, Any]
    ) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().cpu().clone()
            else:
                cloned[key] = value
        return cloned

    def _extract_reference_generate_kwargs(
        self,
        batch: dict[str, Any],
        index: int,
    ) -> dict[str, torch.Tensor]:
        ref_input_ids = batch.get("ref_input_ids")
        ref_attention_mask = batch.get("ref_attention_mask")
        if ref_input_ids is None or ref_attention_mask is None:
            return {}
        if not torch.is_tensor(ref_input_ids) or not torch.is_tensor(ref_attention_mask):
            return {}
        if ref_input_ids.ndim != 2 or ref_attention_mask.ndim != 2:
            return {}
        return {
            "ref_input_ids": ref_input_ids[index : index + 1].to(self.device),
            "ref_attention_mask": ref_attention_mask[index : index + 1].to(self.device),
        }

    def _get_fixed_preview_batch(self) -> dict[str, Any] | None:
        if self.fixed_preview_batch is not None:
            return self.fixed_preview_batch

        try:
            dataloader = build_hf_validation_dataloader(
                dp_world_size=1,
                dp_rank=0,
                tokenizer=self.tokenizer,
                audio_tokenizer=self.audio_tokenizer,
                feature_extractor=self.feature_extractor,
                job_config=self.job_config,
            )
            iterator = iter(dataloader)
            first_batch = next(iterator, None)
            if first_batch is None:
                return None
            self.fixed_preview_batch = self._clone_preview_batch(first_batch)
            return self.fixed_preview_batch
        except Exception as exc:
            logger.warning(f"Failed to build fixed preview batch: {exc}")
            return None

    def _build_codebook_heatmap_rgb(
        self,
        codes_qt: torch.Tensor,
        codebook_size: int,
    ) -> np.ndarray:
        codes_np = codes_qt.detach().cpu().numpy().astype(np.float32)
        denom = float(max(codebook_size - 1, 1))
        norm = np.clip(codes_np / denom, 0.0, 1.0)  # (Q, T)

        # Lightweight colormap: blue -> cyan -> yellow
        r = norm
        g = 1.0 - np.abs(norm - 0.5) * 1.6
        g = np.clip(g, 0.0, 1.0)
        b = 1.0 - norm * 0.85
        rgb = np.stack([r, g, b], axis=-1)  # (Q, T, 3)

        q_count, t_count = int(norm.shape[0]), int(norm.shape[1])
        row_scale = max(12, 192 // max(q_count, 1))
        col_scale = max(1, min(6, 1200 // max(t_count, 1)))
        rgb = np.repeat(np.repeat(rgb, row_scale, axis=0), col_scale, axis=1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _build_audio_spectrogram_rgb(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        waveform = np.asarray(audio_np, dtype=np.float32).reshape(-1)
        if waveform.size < 256:
            waveform = np.pad(waveform, (0, 256 - waveform.size))

        n_fft = 1024
        hop = 256
        win = np.hanning(n_fft).astype(np.float32)
        if waveform.size < n_fft:
            waveform = np.pad(waveform, (0, n_fft - waveform.size))

        frames = []
        for start in range(0, waveform.size - n_fft + 1, hop):
            x = waveform[start : start + n_fft] * win
            spec = np.fft.rfft(x, n=n_fft)
            frames.append(np.abs(spec))
        if not frames:
            x = waveform[:n_fft] * win
            frames = [np.abs(np.fft.rfft(x, n=n_fft))]

        spec_mag = np.stack(frames, axis=1)  # (F, T)
        spec_db = 20.0 * np.log10(np.maximum(spec_mag, 1e-7))
        spec_db -= float(np.max(spec_db))
        spec_db = np.clip(spec_db, -80.0, 0.0)
        norm = (spec_db + 80.0) / 80.0  # [0,1]

        # Blue -> magenta -> pink style map similar to common speech demos.
        r = np.clip(norm * 1.2, 0.0, 1.0)
        g = np.clip(np.maximum(norm - 0.75, 0.0) * 1.5, 0.0, 1.0)
        b = np.clip(0.25 + norm * 0.9, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)  # (F, T, 3)
        rgb = np.flip(rgb, axis=0)  # low freq at bottom

        # Upscale for readable panel.
        freq_scale = max(1, 384 // max(rgb.shape[0], 1))
        time_scale = max(1, min(6, 1200 // max(rgb.shape[1], 1)))
        rgb = np.repeat(np.repeat(rgb, freq_scale, axis=0), time_scale, axis=1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _maybe_add_codebook_visualizations(
        self,
        media_metrics: dict[str, Any],
        wandb_module: Any | None,
        codes_bqt: torch.Tensor,
        prefix: str,
        sample_idx: int,
        utterance: str = "",
    ) -> None:
        if codes_bqt.ndim != 3:
            return

        codes_qt = codes_bqt.detach().cpu().to(torch.int64)[0]  # (Q, T)
        if codes_qt.ndim != 2 or codes_qt.numel() == 0:
            return

        q_count, t_count = int(codes_qt.shape[0]), int(codes_qt.shape[1])
        codebook_size = int(self.audio_tokenizer.config.codebook_size)
        flat_codes = codes_qt.reshape(-1).numpy()
        heatmap_rgb = self._build_codebook_heatmap_rgb(
            codes_qt=codes_qt, codebook_size=codebook_size
        )

        self._save_local_codebook_artifacts(
            prefix=prefix,
            sample_idx=sample_idx,
            codes_qt=codes_qt,
            codebook_size=codebook_size,
            utterance=utterance,
        )
        if wandb_module is None:
            return

        media_metrics[f"samples/{prefix}_codebook_heatmap_{sample_idx}"] = (
            wandb_module.Image(
                heatmap_rgb,
                caption=(
                    f"{prefix} codebook map step={self.step} sample={sample_idx} "
                    f"(rows=quantizers, cols=frames)"
                ),
            )
        )
        media_metrics[f"samples/{prefix}_codebook_hist_{sample_idx}"] = (
            wandb_module.Histogram(flat_codes)
        )
        media_metrics[f"samples/{prefix}_codebook_frames_{sample_idx}"] = t_count

        unique_total = int(torch.unique(codes_qt).numel())
        media_metrics[f"samples/{prefix}_codebook_unique_total_{sample_idx}"] = (
            unique_total
        )
        media_metrics[f"samples/{prefix}_codebook_coverage_total_{sample_idx}"] = (
            float(unique_total) / float(max(codebook_size, 1))
        )

        max_table_frames = min(t_count, 64)
        codebook_table = wandb_module.Table(
            columns=["frame"] + [f"q{q_idx}" for q_idx in range(q_count)]
        )
        for frame_idx in range(max_table_frames):
            row = [frame_idx]
            row.extend(int(codes_qt[q_idx, frame_idx].item()) for q_idx in range(q_count))
            codebook_table.add_data(*row)
        media_metrics[f"samples/{prefix}_codebook_table_{sample_idx}"] = codebook_table

        q_stats_table = wandb_module.Table(columns=["quantizer", "unique", "coverage"])
        coverage_values: list[float] = []
        unique_values: list[int] = []
        for q_idx in range(q_count):
            unique_q = int(torch.unique(codes_qt[q_idx]).numel())
            coverage_q = float(unique_q) / float(max(codebook_size, 1))
            q_stats_table.add_data(f"q{q_idx}", unique_q, coverage_q)
            unique_values.append(unique_q)
            coverage_values.append(coverage_q)
        media_metrics[f"samples/{prefix}_codebook_qstats_{sample_idx}"] = q_stats_table
        if coverage_values:
            media_metrics[f"samples/{prefix}_codebook_coverage_q_min_{sample_idx}"] = (
                float(np.min(coverage_values))
            )
            media_metrics[f"samples/{prefix}_codebook_coverage_q_mean_{sample_idx}"] = (
                float(np.mean(coverage_values))
            )
            media_metrics[f"samples/{prefix}_codebook_coverage_q_max_{sample_idx}"] = (
                float(np.max(coverage_values))
            )
        if unique_values:
            media_metrics[f"samples/{prefix}_codebook_unique_q_min_{sample_idx}"] = (
                float(np.min(unique_values))
            )
            media_metrics[f"samples/{prefix}_codebook_unique_q_mean_{sample_idx}"] = (
                float(np.mean(unique_values))
            )
            media_metrics[f"samples/{prefix}_codebook_unique_q_max_{sample_idx}"] = (
                float(np.max(unique_values))
            )

    def _save_local_codebook_artifacts(
        self,
        prefix: str,
        sample_idx: int,
        codes_qt: torch.Tensor,
        codebook_size: int,
        utterance: str = "",
    ) -> None:
        if torch.distributed.get_rank() != 0:
            return

        step_dir = (
            self.local_sample_artifacts_dir
            / f"step_{self.step:06d}"
            / f"sample_{sample_idx}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)

        codes_np = codes_qt.detach().cpu().numpy().astype(np.int32)
        q_count, t_count = int(codes_np.shape[0]), int(codes_np.shape[1])
        denom = float(max(codebook_size - 1, 1))
        heatmap_np = (codes_np.astype(np.float32) / denom).astype(np.float32)
        unique_total = int(np.unique(codes_np).size)
        coverage_total = float(unique_total) / float(max(codebook_size, 1))

        codes_csv = step_dir / f"{prefix}_codes_qt.csv"
        heatmap_csv = step_dir / f"{prefix}_heatmap_qt.csv"
        if self.job_config.artifact.save_local_codebook_csv:
            np.savetxt(codes_csv, codes_np, delimiter=",", fmt="%d")
            np.savetxt(heatmap_csv, heatmap_np, delimiter=",", fmt="%.6f")

        summary_md = step_dir / f"{prefix}_summary.md"
        if not self.job_config.artifact.save_local_markdown:
            return
        lines = [
            f"# {prefix} codebook summary",
            "",
            f"- step: {self.step}",
            f"- sample: {sample_idx}",
            f"- quantizers: {q_count}",
            f"- frames: {t_count}",
            f"- unique_total: {unique_total}",
            f"- coverage_total: {coverage_total:.6f}",
        ]
        if utterance:
            lines.append(f"- utterance: {utterance}")

        lines.extend(
            [
                "",
                "## Per-quantizer stats",
                "",
                "| quantizer | unique | coverage |",
                "|---|---:|---:|",
            ]
        )
        for q_idx in range(q_count):
            unique_q = int(np.unique(codes_np[q_idx]).size)
            coverage_q = float(unique_q) / float(max(codebook_size, 1))
            lines.append(f"| q{q_idx} | {unique_q} | {coverage_q:.6f} |")

        lines.extend(
            [
                "",
                "## Files",
                "",
            ]
        )
        if self.job_config.artifact.save_local_codebook_csv:
            lines.append(f"- codes CSV (QxT): `{codes_csv.name}`")
            lines.append(f"- heatmap CSV (QxT normalized): `{heatmap_csv.name}`")
        summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _maybe_log_sample_media(self, input_dict: dict[str, torch.Tensor]) -> None:
        cfg = self.job_config.training
        if cfg.sample_generate_every <= 0:
            return
        if self.step % cfg.sample_generate_every != 0:
            return
        if torch.distributed.get_rank() != 0:
            return

        wandb_module = getattr(self.metrics_processor.logger, "wandb", None)
        model = self.model_parts[0]
        num_quantizers = self.job_config.model.num_quantizers
        sample_rate = self.feature_extractor.sampling_rate
        preview_input_dict = self._get_fixed_preview_batch() or input_dict
        bsz = min(
            cfg.sample_generate_num_samples,
            int(preview_input_dict["input_ids"].shape[0]),
        )
        media_metrics: dict[str, Any] = {}
        tts_eval_cfg = self.job_config.tts_eval
        tts_eval_every = (
            tts_eval_cfg.eval_every if tts_eval_cfg.eval_every > 0 else cfg.sample_generate_every
        )
        run_tts_eval = (
            tts_eval_cfg.enabled and tts_eval_every > 0 and self.step % tts_eval_every == 0
        )

        model_was_training = model.training
        model.eval()
        try:
            for i in range(bsz):
                input_ids_row = preview_input_dict["input_ids"][i].tolist()
                attention_mask_row = preview_input_dict["attention_mask"][i].tolist()
                input_ids_row = filter_tokens_by_attention_mask(
                    input_ids_row, attention_mask_row
                )
                valid_len = len(input_ids_row)
                if valid_len == 0:
                    continue

                audio_span_start, audio_span_end = get_audio_span_indices(
                    input_ids_row, self.audio_start_id, self.audio_end_id
                )
                if (
                    audio_span_start > 0
                    and self.audio_start_id is not None
                    and input_ids_row[audio_span_start - 1] == self.audio_start_id
                ):
                    prompt_ids = input_ids_row[:audio_span_start]
                else:
                    prompt_ids = input_ids_row[: min(valid_len, 128)]
                if len(prompt_ids) == 0:
                    continue

                prompt_text = self.tokenizer.decode(
                    prompt_ids, skip_special_tokens=False
                )
                prompt_tensor = (
                    torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
                    .unsqueeze(0)
                )
                prompt_has_audio_start = (
                    self.audio_start_id is not None
                    and len(prompt_ids) > 0
                    and prompt_ids[-1] == self.audio_start_id
                )
                resolved_max_new_tokens = self._resolve_sample_generate_max_new_tokens(
                    audio_span_start=audio_span_start,
                    audio_span_end=audio_span_end,
                )
                reference_generate_kwargs = self._extract_reference_generate_kwargs(
                    preview_input_dict,
                    i,
                )

                def _generate_ids(constrained: bool) -> list[int]:
                    from transformers import LogitsProcessorList

                    input_ids_for_generate = prompt_tensor
                    use_audio_only = bool(
                        (constrained or cfg.sample_generate_restrict_audio_vocab)
                        and self.audio_generation_token_ids
                    )
                    if use_audio_only and self.audio_start_id is not None and not prompt_has_audio_start:
                        constrained_prompt_ids = [*prompt_ids, self.audio_start_id]
                        input_ids_for_generate = (
                            torch.tensor(
                                constrained_prompt_ids,
                                dtype=torch.long,
                                device=self.device,
                            )
                            .unsqueeze(0)
                        )

                    generate_kwargs: dict[str, Any] = {
                        "input_ids": input_ids_for_generate,
                        "max_new_tokens": resolved_max_new_tokens,
                        "do_sample": cfg.sample_generate_do_sample,
                        "eos_token_id": self.audio_end_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                    }
                    generate_kwargs.update(reference_generate_kwargs)
                    if cfg.overfit_num_samples > 0:
                        generate_kwargs["max_time"] = 45.0
                    if cfg.sample_generate_do_sample:
                        generate_kwargs.update(
                            {
                                "temperature": cfg.sample_generate_temperature,
                                "top_k": cfg.sample_generate_top_k,
                            }
                        )
                    if use_audio_only:
                        # Require at least a few audio tokens so constrained
                        # preview logging doesn't terminate immediately with
                        # an empty <audio></audio> span.
                        if resolved_max_new_tokens > 1:
                            min_audio_tokens = max(int(num_quantizers) * 4, 16)
                            generate_kwargs["min_new_tokens"] = min(
                                min_audio_tokens,
                                resolved_max_new_tokens - 1,
                            )
                        generate_kwargs["logits_processor"] = LogitsProcessorList(
                            [
                                AllowTokenIdsLogitsProcessor(
                                    self.audio_generation_token_ids
                                )
                            ]
                        )

                    with torch.no_grad():
                        generated_ids = model.generate(**generate_kwargs)[0]
                    return generated_ids.detach().cpu().tolist()

                target_text = self.tokenizer.decode(
                    input_ids_row, skip_special_tokens=False
                )
                input_utterance = self._decode_utterance_from_token_ids(input_ids_row)
                sample_language = extract_language_token(target_text)
                normalized_reference = self._normalize_text_for_eval(
                    input_utterance, lang_hint=sample_language
                )

                media_metrics[f"samples/prompt_{i}"] = prompt_text
                media_metrics[f"samples/utterance_{i}"] = input_utterance
                if sample_language:
                    media_metrics[f"samples/language_{i}"] = sample_language
                if cfg.log_target_media:
                    media_metrics[f"samples/target_text_{i}"] = target_text[:1200]
                    media_metrics[f"samples/target_utterance_{i}"] = input_utterance
                media_metrics[f"samples/max_new_tokens_effective_{i}"] = int(
                    resolved_max_new_tokens
                )
                media_metrics[f"samples/generate_status_unconstrained_{i}"] = "pending"
                if cfg.sample_generate_dual_mode:
                    media_metrics[f"samples/generate_status_constrained_{i}"] = "pending"

                target_audio_np = None
                generated_audio_np = None
                constrained_audio_np = None
                target_stats = {"frames": 0.0, "coverage_total": 0.0}
                unconstrained_stats = {"frames": 0.0, "coverage_total": 0.0}
                constrained_stats = {"frames": 0.0, "coverage_total": 0.0}
                generated_text = ""
                generated_utterance = input_utterance

                tgt_codes = self._extract_audio_codes_bqt_from_ids(
                    input_ids_row,
                    num_quantizers,
                    audio_start_idx=audio_span_start - 1
                    if audio_span_start > 0
                    and self.audio_start_id is not None
                    and input_ids_row[audio_span_start - 1] == self.audio_start_id
                    else None,
                    audio_end_idx=audio_span_end,
                )
                target_global_codes = self._extract_global_codes_from_ids(input_ids_row)
                if tgt_codes is not None:
                    try:
                        target_audio_np = self._decode_audio_numpy(
                            tgt_codes,
                            global_codes=target_global_codes,
                        )
                        self._save_local_audio_artifact(
                            prefix="target",
                            sample_idx=i,
                            audio_np=target_audio_np,
                            sample_rate=sample_rate,
                        )
                        if wandb_module is not None and cfg.log_target_media:
                            media_metrics[f"samples/target_audio_{i}"] = wandb_module.Audio(
                                target_audio_np,
                                sample_rate=sample_rate,
                                caption=(
                                    f"target step={self.step} sample={i} "
                                    f"utterance='{input_utterance}'"
                                ),
                            )
                        if cfg.log_target_media:
                            self._maybe_add_codebook_visualizations(
                                media_metrics=media_metrics,
                                wandb_module=wandb_module,
                                codes_bqt=tgt_codes,
                                prefix="target",
                                sample_idx=i,
                                utterance=input_utterance,
                            )
                        target_stats = self._get_codebook_stats(tgt_codes)
                    except Exception as e:
                        logger.warning(
                            f"Target audio decode failed at step={self.step}, sample={i}: {e}"
                        )

                generated_ids_row: list[int] | None = None
                try:
                    generated_ids_row = _generate_ids(constrained=False)
                except Exception as e:
                    logger.warning(
                        f"Unconstrained generation failed at step={self.step}, sample={i}: {e}"
                    )
                    media_metrics[f"samples/generate_status_unconstrained_{i}"] = (
                        f"error: {str(e)[:200]}"
                    )

                if generated_ids_row is not None:
                    generated_text = self.tokenizer.decode(
                        generated_ids_row, skip_special_tokens=False
                    )
                    media_metrics[f"samples/generated_text_{i}"] = generated_text[:1200]
                    if cfg.log_unconstrained_named_media:
                        media_metrics[f"samples/generated_text_unconstrained_{i}"] = (
                            generated_text[:1200]
                        )
                    generated_utterance = self._decode_utterance_from_token_ids(
                        generated_ids_row
                    )
                    if not generated_utterance:
                        generated_utterance = input_utterance
                    if cfg.log_unconstrained_named_media:
                        media_metrics[f"samples/generated_utterance_unconstrained_{i}"] = (
                            generated_utterance
                        )

                    generated_span_start, generated_span_end = get_audio_span_indices(
                        generated_ids_row, self.audio_start_id, self.audio_end_id
                    )
                    gen_codes = self._extract_audio_codes_bqt_from_ids(
                        generated_ids_row,
                        num_quantizers,
                        audio_start_idx=generated_span_start - 1
                        if generated_span_start > 0
                        and self.audio_start_id is not None
                        and generated_ids_row[generated_span_start - 1]
                        == self.audio_start_id
                        else None,
                        audio_end_idx=generated_span_end,
                    )
                else:
                    gen_codes = None

                if gen_codes is not None:
                    try:
                        generated_global_codes = None
                        if generated_ids_row is not None:
                            generated_global_codes = self._extract_global_codes_from_ids(
                                generated_ids_row
                            )
                        if generated_global_codes is None:
                            generated_global_codes = target_global_codes
                        generated_audio_np = self._decode_audio_numpy(
                            gen_codes,
                            global_codes=generated_global_codes,
                        )
                        self._save_local_audio_artifact(
                            prefix="generated_unconstrained",
                            sample_idx=i,
                            audio_np=generated_audio_np,
                            sample_rate=sample_rate,
                        )
                        if wandb_module is not None:
                            media_metrics[f"samples/generated_audio_{i}"] = wandb_module.Audio(
                                generated_audio_np,
                                sample_rate=sample_rate,
                                caption=(
                                    f"generated step={self.step} sample={i} "
                                    f"utterance='{generated_utterance}'"
                                ),
                            )
                            if cfg.log_unconstrained_named_media:
                                media_metrics[f"samples/generated_audio_unconstrained_{i}"] = (
                                    media_metrics[f"samples/generated_audio_{i}"]
                                )
                            else:
                                # Keep legacy dashboard keys populated while
                                # still logging a minimal canonical key set.
                                media_metrics[f"samples/generated_audio_unconstrained_{i}"] = (
                                    media_metrics[f"samples/generated_audio_{i}"]
                                )
                            spec_rgb = self._build_audio_spectrogram_rgb(
                                generated_audio_np,
                                sample_rate=sample_rate,
                            )
                            media_metrics[f"samples/generated_audio_spectrogram_{i}"] = (
                                wandb_module.Image(
                                    spec_rgb,
                                    caption=(
                                        f"generated spectrogram step={self.step} "
                                        f"sample={i} sr={sample_rate}"
                                    ),
                                )
                            )
                        generated_codebook_prefix = (
                            "generated_unconstrained"
                            if cfg.log_unconstrained_named_media
                            else "generated"
                        )
                        self._maybe_add_codebook_visualizations(
                            media_metrics=media_metrics,
                            wandb_module=wandb_module,
                            codes_bqt=gen_codes,
                            prefix=generated_codebook_prefix,
                            sample_idx=i,
                            utterance=generated_utterance,
                        )
                        unconstrained_stats = self._get_codebook_stats(gen_codes)
                        self.generated_audio_unconstrained_seen = True
                        media_metrics[f"samples/generate_status_unconstrained_{i}"] = "ok"
                    except Exception as e:
                        logger.warning(
                            f"Generated audio decode failed at step={self.step}, sample={i}: {e}"
                        )
                        media_metrics[f"samples/generate_status_unconstrained_{i}"] = (
                            f"decode_error: {str(e)[:200]}"
                        )
                elif generated_ids_row is not None:
                    media_metrics[f"samples/generate_status_unconstrained_{i}"] = (
                        "no_audio_span"
                    )

                constrained_utterance = input_utterance
                if cfg.sample_generate_dual_mode:
                    generated_ids_constrained: list[int] | None = None
                    try:
                        generated_ids_constrained = _generate_ids(constrained=True)
                    except Exception as e:
                        logger.warning(
                            f"Constrained generation failed at step={self.step}, sample={i}: {e}"
                        )
                        media_metrics[f"samples/generate_status_constrained_{i}"] = (
                            f"error: {str(e)[:200]}"
                        )

                    if generated_ids_constrained is not None:
                        generated_text_constrained = self.tokenizer.decode(
                            generated_ids_constrained, skip_special_tokens=False
                        )
                        constrained_utterance = self._decode_utterance_from_token_ids(
                            generated_ids_constrained
                        )
                        if not constrained_utterance:
                            constrained_utterance = input_utterance
                        media_metrics[f"samples/generated_text_constrained_{i}"] = (
                            generated_text_constrained[:1200]
                        )
                        media_metrics[f"samples/generated_utterance_constrained_{i}"] = (
                            constrained_utterance
                        )
                        constrained_span_start, constrained_span_end = (
                            get_audio_span_indices(
                                generated_ids_constrained,
                                self.audio_start_id,
                                self.audio_end_id,
                            )
                        )
                        constrained_codes = self._extract_audio_codes_bqt_from_ids(
                            generated_ids_constrained,
                            num_quantizers,
                            audio_start_idx=constrained_span_start - 1
                            if constrained_span_start > 0
                            and self.audio_start_id is not None
                            and generated_ids_constrained[constrained_span_start - 1]
                            == self.audio_start_id
                            else None,
                            audio_end_idx=constrained_span_end,
                        )
                    else:
                        constrained_codes = None

                    if constrained_codes is not None:
                        try:
                            constrained_global_codes = None
                            if generated_ids_constrained is not None:
                                constrained_global_codes = self._extract_global_codes_from_ids(
                                    generated_ids_constrained
                                )
                            if constrained_global_codes is None:
                                constrained_global_codes = target_global_codes
                            constrained_audio_np = self._decode_audio_numpy(
                                constrained_codes,
                                global_codes=constrained_global_codes,
                            )
                            self._save_local_audio_artifact(
                                prefix="generated_constrained",
                                sample_idx=i,
                                audio_np=constrained_audio_np,
                                sample_rate=sample_rate,
                            )
                            if wandb_module is not None:
                                media_metrics[f"samples/generated_audio_constrained_{i}"] = (
                                    wandb_module.Audio(
                                        constrained_audio_np,
                                        sample_rate=sample_rate,
                                        caption=(
                                            "generated constrained "
                                            f"step={self.step} sample={i} "
                                            f"utterance='{constrained_utterance}'"
                                        ),
                                    )
                                )
                            self._maybe_add_codebook_visualizations(
                                media_metrics=media_metrics,
                                wandb_module=wandb_module,
                                codes_bqt=constrained_codes,
                                prefix="generated_constrained",
                                sample_idx=i,
                                utterance=constrained_utterance,
                            )
                            constrained_stats = self._get_codebook_stats(constrained_codes)
                            media_metrics[f"samples/generate_status_constrained_{i}"] = "ok"
                        except Exception as e:
                            logger.warning(
                                "Constrained generated audio decode failed at "
                                f"step={self.step}, sample={i}: {e}"
                            )
                            media_metrics[f"samples/generate_status_constrained_{i}"] = (
                                f"decode_error: {str(e)[:200]}"
                            )
                    elif generated_ids_constrained is not None:
                        media_metrics[f"samples/generate_status_constrained_{i}"] = (
                            "no_audio_span"
                        )

                asr_unconstrained_text = ""
                asr_constrained_text = ""
                wer_unconstrained = float("nan")
                cer_unconstrained = float("nan")
                wer_constrained = float("nan")
                cer_constrained = float("nan")

                if run_tts_eval and generated_audio_np is not None:
                    asr_unconstrained_text = self._transcribe_audio(
                        generated_audio_np, sample_rate
                    )
                    if asr_unconstrained_text:
                        wer_unconstrained, cer_unconstrained = self._compute_wer_cer(
                            asr_unconstrained_text,
                            input_utterance,
                            lang_hint=sample_language,
                        )
                        media_metrics[f"samples/generated_asr_text_unconstrained_{i}"] = (
                            asr_unconstrained_text[:240]
                        )
                        if self.job_config.tts_eval.compute_wer:
                            media_metrics[f"samples/generated_wer_unconstrained_{i}"] = (
                                wer_unconstrained
                            )
                        if self.job_config.tts_eval.compute_cer:
                            media_metrics[f"samples/generated_cer_unconstrained_{i}"] = (
                                cer_unconstrained
                            )

                if run_tts_eval and constrained_audio_np is not None:
                    asr_constrained_text = self._transcribe_audio(
                        constrained_audio_np, sample_rate
                    )
                    if asr_constrained_text:
                        wer_constrained, cer_constrained = self._compute_wer_cer(
                            asr_constrained_text,
                            input_utterance,
                            lang_hint=sample_language,
                        )
                        media_metrics[f"samples/generated_asr_text_constrained_{i}"] = (
                            asr_constrained_text[:240]
                        )
                        if self.job_config.tts_eval.compute_wer:
                            media_metrics[f"samples/generated_wer_constrained_{i}"] = (
                                wer_constrained
                            )
                        if self.job_config.tts_eval.compute_cer:
                            media_metrics[f"samples/generated_cer_constrained_{i}"] = (
                                cer_constrained
                            )

                sample_payload = {
                    "step": self.step,
                    "sample_idx": i,
                    "language": sample_language,
                    "utterance": input_utterance,
                    "reference_normalized": normalized_reference,
                    "target_frames": int(target_stats["frames"]),
                    "unconstrained_frames": int(unconstrained_stats["frames"]),
                    "constrained_frames": int(constrained_stats["frames"]),
                    "target_coverage_total": float(target_stats["coverage_total"]),
                    "unconstrained_coverage_total": float(
                        unconstrained_stats["coverage_total"]
                    ),
                    "constrained_coverage_total": float(constrained_stats["coverage_total"]),
                    "asr_text_unconstrained": asr_unconstrained_text,
                    "asr_text_constrained": asr_constrained_text,
                }
                if asr_unconstrained_text:
                    sample_payload["asr_text_unconstrained_normalized"] = (
                        self._normalize_text_for_eval(
                            asr_unconstrained_text, lang_hint=sample_language
                        )
                    )
                if asr_constrained_text:
                    sample_payload["asr_text_constrained_normalized"] = (
                        self._normalize_text_for_eval(
                            asr_constrained_text, lang_hint=sample_language
                        )
                    )
                if self.job_config.tts_eval.compute_wer and not np.isnan(wer_unconstrained):
                    sample_payload["wer_unconstrained"] = float(wer_unconstrained)
                if self.job_config.tts_eval.compute_cer and not np.isnan(cer_unconstrained):
                    sample_payload["cer_unconstrained"] = float(cer_unconstrained)
                if self.job_config.tts_eval.compute_wer and not np.isnan(wer_constrained):
                    sample_payload["wer_constrained"] = float(wer_constrained)
                if self.job_config.tts_eval.compute_cer and not np.isnan(cer_constrained):
                    sample_payload["cer_constrained"] = float(cer_constrained)
                self._save_local_metrics_json(i, "sample_eval", sample_payload)

                if i == 0:
                    gate_cfg = self.job_config.overfit_gate
                    generated_frames = float(unconstrained_stats["frames"])
                    target_frames = float(target_stats["frames"])
                    frame_ratio = (
                        generated_frames / max(target_frames, 1.0)
                        if generated_frames > 0 and target_frames > 0
                        else 0.0
                    )
                    coverage_abs_diff = abs(
                        float(unconstrained_stats["coverage_total"])
                        - float(target_stats["coverage_total"])
                    )
                    target_cov_q = self._get_codebook_coverage_per_q(tgt_codes)
                    unconstrained_cov_q = self._get_codebook_coverage_per_q(gen_codes)
                    q_coverage_min = (
                        float(np.min(unconstrained_cov_q))
                        if unconstrained_cov_q
                        else 0.0
                    )
                    q_coverage_abs_diff_max = 0.0
                    if target_cov_q and unconstrained_cov_q:
                        q_count = min(len(target_cov_q), len(unconstrained_cov_q))
                        q_coverage_abs_diff_max = float(
                            max(
                                abs(unconstrained_cov_q[q_idx] - target_cov_q[q_idx])
                                for q_idx in range(q_count)
                            )
                        )
                    wer_pass = True
                    cer_pass = True
                    if self.job_config.tts_eval.compute_wer:
                        wer_pass = (
                            (not run_tts_eval)
                            or np.isnan(wer_unconstrained)
                            or wer_unconstrained <= gate_cfg.wer_max
                        )
                    if self.job_config.tts_eval.compute_cer:
                        cer_pass = (
                            (not run_tts_eval)
                            or np.isnan(cer_unconstrained)
                            or cer_unconstrained <= gate_cfg.cer_max
                        )
                    frame_ratio_pass = (
                        generated_frames > 0
                        and target_frames > 0
                        and gate_cfg.frame_ratio_min <= frame_ratio <= gate_cfg.frame_ratio_max
                    )
                    coverage_pass = (
                        generated_frames > 0
                        and target_frames > 0
                        and coverage_abs_diff <= gate_cfg.coverage_abs_diff_max
                    )
                    coverage_q_min_pass = (
                        generated_frames > 0
                        and q_coverage_min >= gate_cfg.coverage_q_min
                    )
                    coverage_q_diff_pass = (
                        generated_frames > 0
                        and target_frames > 0
                        and q_coverage_abs_diff_max <= gate_cfg.coverage_q_abs_diff_max
                    )
                    audio_seen = generated_frames > 0
                    overall_pass = (
                        audio_seen
                        and wer_pass
                        and cer_pass
                        and frame_ratio_pass
                        and coverage_pass
                        and coverage_q_min_pass
                        and coverage_q_diff_pass
                    )

                    if overall_pass:
                        self.overfit_gate_consecutive_passes += 1
                    else:
                        self.overfit_gate_consecutive_passes = 0
                    if (
                        self.overfit_gate_consecutive_passes
                        >= gate_cfg.min_consecutive_passes
                    ):
                        self.overfit_gate_passed = True

                    gate_payload = {
                        "step": self.step,
                        "language": sample_language,
                        "utterance": input_utterance,
                        "audio_seen": bool(audio_seen),
                        "wer_pass": bool(wer_pass),
                        "cer_pass": bool(cer_pass),
                        "frame_ratio_pass": bool(frame_ratio_pass),
                        "coverage_pass": bool(coverage_pass),
                        "coverage_q_min_pass": bool(coverage_q_min_pass),
                        "coverage_q_diff_pass": bool(coverage_q_diff_pass),
                        "overall_pass": bool(overall_pass),
                        "consecutive_passes": int(self.overfit_gate_consecutive_passes),
                        "gate_passed_ever": bool(self.overfit_gate_passed),
                        "frame_ratio": float(frame_ratio),
                        "coverage_abs_diff": float(coverage_abs_diff),
                        "coverage_q_min": float(q_coverage_min),
                        "coverage_q_abs_diff_max": float(q_coverage_abs_diff_max),
                    }
                    if self.job_config.tts_eval.compute_wer and not np.isnan(wer_unconstrained):
                        gate_payload["wer_unconstrained"] = float(wer_unconstrained)
                    if self.job_config.tts_eval.compute_cer and not np.isnan(cer_unconstrained):
                        gate_payload["cer_unconstrained"] = float(cer_unconstrained)
                    self.latest_overfit_gate_metrics = gate_payload
                    self._save_local_metrics_json(i, "gate", gate_payload)

                    media_metrics["gates/unconstrained_audio_seen"] = int(audio_seen)
                    media_metrics["gates/wer_pass"] = int(wer_pass)
                    media_metrics["gates/cer_pass"] = int(cer_pass)
                    media_metrics["gates/frame_ratio_pass"] = int(frame_ratio_pass)
                    media_metrics["gates/coverage_pass"] = int(coverage_pass)
                    media_metrics["gates/coverage_q_min_pass"] = int(coverage_q_min_pass)
                    media_metrics["gates/coverage_q_diff_pass"] = int(coverage_q_diff_pass)
                    media_metrics["gates/overall_pass"] = int(overall_pass)
                    media_metrics["gates/consecutive_passes"] = int(
                        self.overfit_gate_consecutive_passes
                    )
                    media_metrics["gates/frame_ratio"] = float(frame_ratio)
                    media_metrics["gates/coverage_abs_diff"] = float(coverage_abs_diff)
                    media_metrics["gates/coverage_q_min"] = float(q_coverage_min)
                    media_metrics["gates/coverage_q_abs_diff_max"] = float(
                        q_coverage_abs_diff_max
                    )
                    if "wer_unconstrained" in gate_payload:
                        media_metrics["gates/wer_unconstrained"] = gate_payload[
                            "wer_unconstrained"
                        ]
                    if "cer_unconstrained" in gate_payload:
                        media_metrics["gates/cer_unconstrained"] = gate_payload[
                            "cer_unconstrained"
                        ]
        except Exception as e:
            logger.warning(f"Sample media logging skipped due to generation error: {e}")
        finally:
            if model_was_training:
                model.train()

        if media_metrics:
            if self.job_config.training.minimal_media_logging:
                # Keep only the essential qualitative panels in W&B.
                keep_prefixes = (
                    "samples/utterance_",
                    "samples/target_utterance_",
                    "samples/generated_audio_",
                    "samples/target_audio_",
                    "samples/generated_audio_unconstrained_",
                    "samples/generated_audio_constrained_",
                    "samples/generated_utterance_",
                    "samples/generated_audio_spectrogram_",
                    "samples/generate_status_",
                )
                media_metrics = {
                    k: v
                    for k, v in media_metrics.items()
                    if k.startswith(keep_prefixes)
                }

            core_aliases = {
                "samples/utterance_0": "core/utterance_0",
                "samples/target_utterance_0": "core/target_utterance_0",
                "samples/generated_audio_0": "core/generated_audio_0",
                "samples/generated_utterance_unconstrained_0": "core/generated_utterance_unconstrained_0",
                "samples/generated_utterance_constrained_0": "core/generated_utterance_constrained_0",
                "samples/target_audio_0": "core/target_audio_0",
                "samples/generated_audio_unconstrained_0": "core/generated_audio_unconstrained_0",
                "samples/generated_audio_constrained_0": "core/generated_audio_constrained_0",
                "samples/generated_audio_spectrogram_0": "core/generated_audio_spectrogram_0",
                "samples/generated_wer_unconstrained_0": "core/generated_wer_unconstrained_0",
                "samples/generated_cer_unconstrained_0": "core/generated_cer_unconstrained_0",
                "samples/generated_wer_constrained_0": "core/generated_wer_constrained_0",
                "samples/generated_cer_constrained_0": "core/generated_cer_constrained_0",
                "samples/generate_status_unconstrained_0": "core/generate_status_unconstrained_0",
                "samples/max_new_tokens_effective_0": "core/max_new_tokens_effective_0",
            }
            for src_key, dst_key in core_aliases.items():
                if src_key in media_metrics:
                    media_metrics[dst_key] = media_metrics[src_key]
            self.metrics_processor.logger.log(media_metrics, self.step)

    @torch.no_grad()
    def _maybe_run_full_pack_eval(self) -> None:
        if not self.full_eval_enabled:
            return
        eval_every = self._resolve_full_pack_eval_every()
        if eval_every <= 0 or self.step % eval_every != 0:
            return
        if torch.distributed.get_rank() != 0:
            return

        model = self.model_parts[0]
        cfg = self.job_config.training
        tts_eval_cfg = self.job_config.tts_eval
        sample_rate = self.feature_extractor.sampling_rate
        num_quantizers = self.job_config.model.num_quantizers
        max_samples = tts_eval_cfg.full_pack_max_samples
        step_dir = self._build_full_eval_step_dir()
        sample_rows: list[dict[str, Any]] = []
        sample_index = 0

        dataloader = build_hf_validation_dataloader(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            feature_extractor=self.feature_extractor,
            job_config=self.job_config,
        )

        model_was_training = model.training
        model.eval()
        logger.info(
            "Running full-pack TTS eval at "
            f"step={self.step} on validation split (max_samples={max_samples})."
        )
        try:
            stop = False
            for input_dict in dataloader:
                bsz = int(input_dict["input_ids"].shape[0])
                for i in range(bsz):
                    if max_samples > 0 and sample_index >= max_samples:
                        stop = True
                        break

                    input_ids_row = input_dict["input_ids"][i].tolist()
                    attention_mask_row = input_dict["attention_mask"][i].tolist()
                    input_ids_row = filter_tokens_by_attention_mask(
                        input_ids_row, attention_mask_row
                    )
                    if not input_ids_row:
                        continue

                    audio_span_start, audio_span_end = get_audio_span_indices(
                        input_ids_row, self.audio_start_id, self.audio_end_id
                    )
                    if (
                        audio_span_start > 0
                        and self.audio_start_id is not None
                        and input_ids_row[audio_span_start - 1] == self.audio_start_id
                    ):
                        prompt_ids = input_ids_row[:audio_span_start]
                    else:
                        prompt_ids = input_ids_row
                    if not prompt_ids:
                        continue

                    prompt_tensor = (
                        torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
                        .unsqueeze(0)
                    )
                    reference_generate_kwargs = self._extract_reference_generate_kwargs(
                        input_dict,
                        i,
                    )
                    prompt_has_audio_start = (
                        self.audio_start_id is not None
                        and len(prompt_ids) > 0
                        and prompt_ids[-1] == self.audio_start_id
                    )
                    resolved_max_new_tokens = self._resolve_sample_generate_max_new_tokens(
                        audio_span_start=audio_span_start,
                        audio_span_end=audio_span_end,
                    )

                    def _generate_ids() -> list[int]:
                        from transformers import LogitsProcessorList

                        use_audio_only = bool(
                            cfg.sample_generate_restrict_audio_vocab
                            and self.audio_generation_token_ids
                        )
                        input_ids_for_generate = prompt_tensor
                        if use_audio_only and not prompt_has_audio_start and self.audio_start_id is not None:
                            constrained_prompt_ids = [*prompt_ids, self.audio_start_id]
                            input_ids_for_generate = (
                                torch.tensor(
                                    constrained_prompt_ids,
                                    dtype=torch.long,
                                    device=self.device,
                                )
                                .unsqueeze(0)
                            )
                        generate_kwargs: dict[str, Any] = {
                            "input_ids": input_ids_for_generate,
                            "max_new_tokens": resolved_max_new_tokens,
                            "do_sample": cfg.sample_generate_do_sample,
                            "eos_token_id": self.audio_end_id,
                            "pad_token_id": self.tokenizer.pad_token_id,
                        }
                        generate_kwargs.update(reference_generate_kwargs)
                        if cfg.sample_generate_do_sample:
                            generate_kwargs.update(
                                {
                                    "temperature": cfg.sample_generate_temperature,
                                    "top_k": cfg.sample_generate_top_k,
                                }
                            )
                        if use_audio_only:
                            if resolved_max_new_tokens > 1:
                                min_audio_tokens = max(int(num_quantizers) * 4, 16)
                                generate_kwargs["min_new_tokens"] = min(
                                    min_audio_tokens,
                                    resolved_max_new_tokens - 1,
                                )
                            generate_kwargs["logits_processor"] = LogitsProcessorList(
                                [
                                    AllowTokenIdsLogitsProcessor(
                                        self.audio_generation_token_ids
                                    )
                                ]
                            )
                        generated_ids = model.generate(**generate_kwargs)[0]
                        return generated_ids.detach().cpu().tolist()

                    target_text = self.tokenizer.decode(
                        input_ids_row, skip_special_tokens=False
                    )
                    input_utterance = self._decode_utterance_from_token_ids(input_ids_row)
                    sample_language = extract_language_token(target_text)
                    normalized_reference = self._normalize_text_for_eval(
                        input_utterance, lang_hint=sample_language
                    )
                    sample_dir = step_dir / f"sample_{sample_index:05d}"
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    tgt_codes = self._extract_audio_codes_bqt_from_ids(
                        input_ids_row,
                        num_quantizers,
                        audio_start_idx=audio_span_start - 1
                        if audio_span_start > 0
                        and self.audio_start_id is not None
                        and input_ids_row[audio_span_start - 1] == self.audio_start_id
                        else None,
                        audio_end_idx=audio_span_end,
                    )
                    target_audio_np = None
                    target_stats = {"frames": 0.0, "coverage_total": 0.0}
                    target_cov_q: list[float] = []
                    if tgt_codes is not None:
                        try:
                            target_audio_np = self._decode_audio_numpy(tgt_codes)
                            self._save_audio_artifact_to_dir(
                                sample_dir,
                                prefix="target",
                                audio_np=target_audio_np,
                                sample_rate=sample_rate,
                            )
                            target_stats = self._get_codebook_stats(tgt_codes)
                            target_cov_q = self._get_codebook_coverage_per_q(tgt_codes)
                        except Exception as e:
                            logger.warning(
                                "Target audio decode failed in full eval at "
                                f"step={self.step}, sample={sample_index}: {e}"
                            )

                    generated_audio_np = None
                    generated_stats = {"frames": 0.0, "coverage_total": 0.0}
                    generated_cov_q: list[float] = []
                    generated_ids_row: list[int] | None = None
                    generation_status = "pending"
                    asr_text = ""
                    wer = None
                    cer = None
                    generated_utterance = input_utterance

                    try:
                        generated_ids_row = _generate_ids()
                    except Exception as e:
                        generation_status = f"error: {str(e)[:200]}"
                        logger.warning(
                            "Full eval generation failed at "
                            f"step={self.step}, sample={sample_index}: {e}"
                        )

                    if generated_ids_row is not None:
                        generated_utterance = self._decode_utterance_from_token_ids(
                            generated_ids_row
                        )
                        if not generated_utterance:
                            generated_utterance = input_utterance
                        generated_span_start, generated_span_end = get_audio_span_indices(
                            generated_ids_row, self.audio_start_id, self.audio_end_id
                        )
                        gen_codes = self._extract_audio_codes_bqt_from_ids(
                            generated_ids_row,
                            num_quantizers,
                            audio_start_idx=generated_span_start - 1
                            if generated_span_start > 0
                            and self.audio_start_id is not None
                            and generated_ids_row[generated_span_start - 1]
                            == self.audio_start_id
                            else None,
                            audio_end_idx=generated_span_end,
                        )
                    else:
                        gen_codes = None

                    if gen_codes is not None:
                        try:
                            generated_audio_np = self._decode_audio_numpy(gen_codes)
                            self._save_audio_artifact_to_dir(
                                sample_dir,
                                prefix="generated",
                                audio_np=generated_audio_np,
                                sample_rate=sample_rate,
                            )
                            generated_stats = self._get_codebook_stats(gen_codes)
                            generated_cov_q = self._get_codebook_coverage_per_q(gen_codes)
                            generation_status = "ok"
                        except Exception as e:
                            generation_status = f"decode_error: {str(e)[:200]}"
                            logger.warning(
                                "Generated audio decode failed in full eval at "
                                f"step={self.step}, sample={sample_index}: {e}"
                            )
                    elif generated_ids_row is not None:
                        generation_status = "no_audio_span"

                    if tts_eval_cfg.enabled and generated_audio_np is not None:
                        asr_text = self._transcribe_audio(generated_audio_np, sample_rate)
                        if asr_text:
                            wer_value, cer_value = self._compute_wer_cer(
                                asr_text,
                                input_utterance,
                                lang_hint=sample_language,
                            )
                            if self.job_config.tts_eval.compute_wer and not np.isnan(wer_value):
                                wer = float(wer_value)
                            if self.job_config.tts_eval.compute_cer and not np.isnan(cer_value):
                                cer = float(cer_value)

                    coverage_q_min = (
                        float(np.min(generated_cov_q)) if generated_cov_q else None
                    )
                    target_frames = float(target_stats["frames"])
                    generated_frames = float(generated_stats["frames"])
                    frame_ratio = None
                    if target_frames > 0 and generated_frames > 0:
                        frame_ratio = generated_frames / max(target_frames, 1.0)
                    coverage_q_abs_diff_max = None
                    if target_cov_q and generated_cov_q:
                        q_count = min(len(target_cov_q), len(generated_cov_q))
                        coverage_q_abs_diff_max = float(
                            max(
                                abs(generated_cov_q[q_idx] - target_cov_q[q_idx])
                                for q_idx in range(q_count)
                            )
                        )

                    sample_payload: dict[str, Any] = {
                        "step": self.step,
                        "sample_idx": sample_index,
                        "language": sample_language,
                        "utterance": input_utterance,
                        "reference_normalized": normalized_reference,
                        "generated_utterance": generated_utterance,
                        "generation_status": generation_status,
                        "target_frames": int(target_frames),
                        "generated_frames": int(generated_frames),
                        "frame_ratio": frame_ratio,
                        "target_coverage_total": float(target_stats["coverage_total"]),
                        "generated_coverage_total": float(
                            generated_stats["coverage_total"]
                        ),
                        "target_coverage_q": target_cov_q,
                        "generated_coverage_q": generated_cov_q,
                        "coverage_q_min": coverage_q_min,
                        "coverage_q_abs_diff_max": coverage_q_abs_diff_max,
                        "asr_text": asr_text,
                    }
                    if wer is not None:
                        sample_payload["wer"] = wer
                    if cer is not None:
                        sample_payload["cer"] = cer
                    if asr_text:
                        sample_payload["asr_text_normalized"] = self._normalize_text_for_eval(
                            asr_text, lang_hint=sample_language
                        )
                    self._write_json_file(sample_dir / "sample_metrics.json", sample_payload)
                    sample_rows.append(sample_payload)
                    sample_index += 1
                    if sample_index % 10 == 0:
                        logger.info(
                            "Full-pack TTS eval progress at step=%s: processed %s/%s samples.",
                            self.step,
                            sample_index,
                            max_samples if max_samples > 0 else "all",
                        )

                if stop:
                    break

            summary = summarize_full_eval_rows(sample_rows)
            summary.update(
                {
                    "step": self.step,
                    "eval_pack": "validation",
                    "sample_rate": sample_rate,
                    "num_quantizers": num_quantizers,
                    "max_samples": max_samples,
                }
            )
            self._write_json_file(step_dir / "summary.json", summary)
            self._maybe_run_rich_full_eval_scoring(step_dir)
        finally:
            if model_was_training:
                model.train()

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        accumulated_losses = []
        last_input_dict = None
        # If data runs out during gradient accumulation, that
        # entire step will not be executed.
        for microbatch in range(self.gradient_accumulation_steps):
            input_dict = next(data_iterator)
            last_input_dict = input_dict
            loss = self.forward_backward_step(input_dict)
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=(
                parallel_dims.world_mesh["pp"] if parallel_dims.pp_enabled else None
            ),
            ep_dense_params_mesh_ndim=(
                parallel_dims.dense_params_mesh_ndim
                if parallel_dims.ep_enabled
                else None
            ),
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # Sample generation cadence is controlled by training.sample_generate_every.
        # Keep it independent from metrics.log_freq/should_log so strict overfit
        # deadlines are based on actual generation attempts.
        if last_input_dict is not None:
            self._maybe_log_sample_media(last_input_dict)

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            ft_pg = self.ft_manager.loss_sync_pg
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, parallel_dims.world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss, parallel_dims.world_mesh["dp_cp"], ft_pg),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()

        lr = self.optimizers.optimizers[0].param_groups[0]["lr"]
        ema_beta = self.job_config.training.metrics_ema_beta
        if self.loss_ema is None:
            self.loss_ema = float(global_avg_loss)
            self.grad_norm_ema = float(grad_norm.item())
        else:
            self.loss_ema = ema_beta * self.loss_ema + (1.0 - ema_beta) * float(
                global_avg_loss
            )
            self.grad_norm_ema = ema_beta * self.grad_norm_ema + (
                1.0 - ema_beta
            ) * float(grad_norm.item())

        extra_metrics = {
            "loss_metrics/global_avg_loss_ema": self.loss_ema,
            "grad_norm_ema": self.grad_norm_ema,
            "core/train_loss": float(global_avg_loss),
            "core/train_loss_ema": self.loss_ema,
            "core/grad_norm_ema": self.grad_norm_ema,
        }

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            lr,
            self.epoch,
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}.")

        with (
            maybe_enable_profiling(job_config, global_step=self.step) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config, global_step=self.step
            ) as memory_profiler,
            maybe_semi_sync_training(
                job_config.fault_tolerance,
                ft_manager=self.ft_manager,
                model_parts=self.model_parts,
                optimizer=self.optimizers,
            ),
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderStopIteration:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # Run validation if validator is available
                if (
                    self.job_config.validation.enabled
                    and self.validator.should_validate(self.step)
                ):
                    self.validator.validate(self.model_parts, self.step)

                require_audio = (
                    job_config.overfit_gate.require_unconstrained_audio
                    or job_config.training.overfit_require_generated_audio
                )
                audio_deadline_step = (
                    job_config.overfit_gate.audio_deadline_step
                    if job_config.overfit_gate.audio_deadline_step > 0
                    else job_config.training.overfit_generated_audio_deadline_step
                )
                if (
                    require_audio
                    and audio_deadline_step > 0
                    and self.step >= audio_deadline_step
                    and not self.generated_audio_unconstrained_seen
                ):
                    raise RuntimeError(
                        "Strict overfit failed: no unconstrained generated audio was "
                        f"logged by step {self.step}."
                    )

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # Save resumable state before heavy full-pack eval so long runs do not
                # lose progress if checkpoint-side scoring stalls or is interrupted.
                self._maybe_run_full_pack_eval()

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.parallel_dims.world_mesh,
                    )

        require_audio = (
            job_config.overfit_gate.require_unconstrained_audio
            or job_config.training.overfit_require_generated_audio
        )
        if require_audio and not self.generated_audio_unconstrained_seen:
            raise RuntimeError(
                "Strict overfit failed: unconstrained generation never produced decodable audio."
            )
        if job_config.overfit_gate.fail_on_no_pass and not self.overfit_gate_passed:
            raise RuntimeError(
                "Strict overfit failed: objective overfit gates never reached pass state."
            )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def validation(self, data_loader):
        # clear_gpu_memory()
        model = self.model_parts[0]
        model.eval()
        local_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels", batch["input_ids"]).to(self.device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss.item()
                    local_loss += loss * input_ids.size(0)
                    total_tokens += input_ids.size(0)

                del outputs, input_ids, attention_mask

        local_loss = torch.tensor([local_loss], device=self.device)
        total_tokens = torch.tensor([total_tokens], device=self.device)
        import torch.distributed as dist

        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            validation_loss = (local_loss / total_tokens).item()
            self.metrics_processor.logger.log(
                {"validation/loss": validation_loss}, self.step
            )
            logger.info(
                f"{self.metrics_processor.color.green}loss: {validation_loss:.4f}{self.metrics_processor.color.reset}"
            )

        model.train()
        # clear_gpu_memory()

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metrics_processor:
            self.metrics_processor.close()


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[Trainer] = None

    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert int(os.environ["WORLD_SIZE"]) == 1, (
                "Must create seed checkpoint using a single device, to disable sharding."
            )
            assert config.checkpoint.enable_checkpoint, (
                "Must enable checkpointing when creating a seed checkpoint."
            )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")
