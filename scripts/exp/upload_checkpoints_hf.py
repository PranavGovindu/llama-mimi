#!/usr/bin/env python3
import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.train import expand_tokenizer_with_unit_tokens


def _discover_steps(folder: Path) -> list[Path]:
    out = []
    for p in folder.glob("step-*"):
        if p.is_dir():
            try:
                int(p.name.split("step-")[-1])
            except Exception:
                continue
            out.append(p)
    return sorted(out, key=lambda p: int(p.name.split("step-")[-1]))


def _step_num(step_dir: Path) -> int:
    return int(step_dir.name.split("step-")[-1])


def _is_checkpoint_ready(step_dir: Path) -> bool:
    has_dcp = (step_dir / ".metadata").exists() and any(
        p.name.endswith(".distcp") for p in step_dir.glob("*.distcp")
    )
    has_hf = (step_dir / "model.safetensors.index.json").exists() or any(
        p.name == "model.safetensors" for p in step_dir.glob("model*.safetensors")
    )
    return has_dcp or has_hf


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _build_converter(
    model_name: str,
    num_quantizers: int,
    codebook_size: int,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ModelWrapper]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer = expand_tokenizer_with_unit_tokens(
        tokenizer,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
    )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    wrapped = ModelWrapper(model)
    return model, tokenizer, wrapped


def _convert_step_to_hf(
    step_dir: Path,
    export_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    wrapped_model: ModelWrapper,
    num_quantizers: int,
    export_dtype: torch.dtype,
) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    dcp.load(wrapped_model.state_dict(), checkpoint_id=str(step_dir))
    model.config.num_quantizers = int(num_quantizers)

    state_dict = model.state_dict()
    if export_dtype != torch.float32:
        state_dict = {k: v.to(export_dtype) for k, v in state_dict.items()}

    model.save_pretrained(
        str(export_dir),
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(str(export_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload step checkpoints to Hugging Face repo.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--idle-exit-seconds", type=int, default=600)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--upload-format",
        default="hf_pretrained",
        choices=["hf_pretrained", "raw"],
        help="hf_pretrained converts each DCP step to save_pretrained files before upload.",
    )
    parser.add_argument("--model-name", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--num-quantizers", type=int, default=1)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument(
        "--export-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--converted-root", default="")
    parser.add_argument("--keep-converted", action="store_true")
    parser.add_argument(
        "--upload-every",
        type=int,
        default=200,
        help=(
            "Only upload checkpoints where step %% upload_every == 0. "
            "Set 0 or negative to upload every discovered step."
        ),
    )
    parser.add_argument(
        "--upload-latest-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before idle exit, upload the latest ready checkpoint once even if "
            "it does not match --upload-every."
        ),
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    model = None
    tokenizer = None
    wrapped_model = None
    converted_root = (
        Path(args.converted_root)
        if args.converted_root.strip()
        else ckpt_dir.parent / f"{ckpt_dir.name}_hf_exports"
    )
    export_dtype = _resolve_dtype(args.export_dtype)
    if args.upload_format == "hf_pretrained":
        model, tokenizer, wrapped_model = _build_converter(
            model_name=args.model_name,
            num_quantizers=int(args.num_quantizers),
            codebook_size=int(args.codebook_size),
        )
        converted_root.mkdir(parents=True, exist_ok=True)

    uploaded: set[str] = set()
    last_new = time.time()

    def _upload_step(step_dir: Path, reason: str = "") -> bool:
        upload_folder = step_dir
        converted_step_dir = None
        try:
            if args.upload_format == "hf_pretrained":
                assert model is not None and tokenizer is not None and wrapped_model is not None
                converted_step_dir = converted_root / step_dir.name
                _convert_step_to_hf(
                    step_dir=step_dir,
                    export_dir=converted_step_dir,
                    model=model,
                    tokenizer=tokenizer,
                    wrapped_model=wrapped_model,
                    num_quantizers=int(args.num_quantizers),
                    export_dtype=export_dtype,
                )
                upload_folder = converted_step_dir

            api.upload_folder(
                folder_path=str(upload_folder),
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo=step_dir.name,
                commit_message=f"upload {step_dir.name}",
            )
            uploaded.add(step_dir.name)
            suffix = f" ({reason})" if reason else ""
            print(
                f"uploaded {step_dir.name} ({args.upload_format}) -> {args.repo_id}{suffix}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                f"failed upload for {step_dir.name}: {exc}",
                flush=True,
            )
            return False
        finally:
            if (
                converted_step_dir is not None
                and converted_step_dir.exists()
                and not args.keep_converted
            ):
                shutil.rmtree(converted_step_dir, ignore_errors=True)

    while True:
        steps = _discover_steps(ckpt_dir)
        new_any = False
        for step_dir in steps:
            if step_dir.name in uploaded:
                continue
            if not _is_checkpoint_ready(step_dir):
                continue
            step_num = _step_num(step_dir)
            if args.upload_every > 0 and step_num % args.upload_every != 0:
                continue
            if _upload_step(step_dir):
                new_any = True

        if new_any:
            last_new = time.time()
        if time.time() - last_new > args.idle_exit_seconds:
            if args.upload_latest_on_exit:
                latest_ready = None
                for step_dir in steps:
                    if step_dir.name in uploaded:
                        continue
                    if not _is_checkpoint_ready(step_dir):
                        continue
                    if latest_ready is None or _step_num(step_dir) > _step_num(latest_ready):
                        latest_ready = step_dir
                if latest_ready is not None and _upload_step(latest_ready, reason="latest_on_exit"):
                    # We intentionally exit after a final upload attempt to avoid
                    # keeping stale uploader loops around after training completion.
                    pass
            print("no new checkpoints; exiting uploader", flush=True)
            break
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
