#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path


def _run_dir(repo_root: Path, experiment_id: str) -> Path:
    run_dir = repo_root / "experiments" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_modal_app_id(output: str) -> str:
    m = re.findall(r"(ap-[A-Za-z0-9]+)", output)
    return m[-1] if m else ""


def _load_run_defaults(repo_root: Path, config_file: str) -> dict[str, object]:
    raw = tomllib.loads((repo_root / config_file).read_text(encoding="utf-8"))
    model_name = str(raw.get("model", {}).get("name", "model")).split("/")[-1]
    dataset_name = str(raw.get("training", {}).get("dataset", "dataset"))
    seq_len = int(raw.get("training", {}).get("seq_len", 2048))
    pretrained = bool(raw.get("model", {}).get("pretrained", True))
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "pretrained": pretrained,
    }


def _resolve_run_name(
    model_name: str,
    dataset_name: str,
    num_quantizers: int,
    seq_len: int,
    pretrained: bool,
    experiment_id: str,
) -> str:
    run_name = (
        f"{model_name}_{dataset_name}"
        f"-q{num_quantizers}"
        f"-s{seq_len}"
        f"{'-random' if not pretrained else ''}"
    )
    if experiment_id:
        run_name = f"{run_name}-{experiment_id}"
    return run_name


def _build_hf_repo_id(
    q: int,
    interval: int,
    repo_prefix: str,
    repo_template: str,
) -> str:
    if repo_template:
        return repo_template.format(q=q, interval=interval)
    if repo_prefix:
        return f"{repo_prefix}-q{q}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Q5/Q6/Q7/Q8 one-sample overfit matrix. "
            "Supports local multi-GPU and modal detached modes."
        )
    )
    parser.add_argument("--mode", choices=["local", "modal"], default="modal")
    parser.add_argument("--config", default="config/tinyaya_q8_download_overfit_1sample.toml")
    parser.add_argument("--modal-cli", default="modal")
    parser.add_argument("--modal-entry", default="modal/app.py::train")
    parser.add_argument("--modal-path", default="overfit_download_q8")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--quantizers", default="5,6,7,8")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--experiment-prefix", default="exp-codebook-matrix")
    parser.add_argument("--phase", default="overfit_codebook_matrix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overfit-num-samples", type=int, default=1)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--keep-latest-k", type=int, default=2)
    parser.add_argument("--checkpoint-folder-prefix", default="checkpoint")
    parser.add_argument("--hf-repo-prefix", default="")
    parser.add_argument(
        "--hf-repo-template",
        default="",
        help="Template supports {q} and {interval}, e.g. rumik-ai/q{q}-{interval}-aya",
    )
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument(
        "--hf-collection-slug",
        default="",
        help="Existing HF collection slug to auto-add each model repo into.",
    )
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="codebook-matrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--torchrun-bin", default="torchrun")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    defaults = _load_run_defaults(repo_root, args.config)
    quantizers = [int(x.strip()) for x in args.quantizers.split(",") if x.strip()]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(quantizers) != len(gpus):
        raise SystemExit("quantizers and gpus must have equal length")

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    wandb_group = args.wandb_group.strip() or f"{args.experiment_prefix}-{ts}"
    launched: list[dict[str, object]] = []

    for q, gpu in zip(quantizers, gpus):
        exp_id = f"{args.experiment_prefix}-{ts}-q{q}"
        run_dir = _run_dir(repo_root, exp_id)
        log_path = run_dir / "launch.log"
        checkpoint_folder = f"{args.checkpoint_folder_prefix}_{exp_id}"
        run_name = _resolve_run_name(
            model_name=str(defaults["model_name"]),
            dataset_name=str(defaults["dataset_name"]),
            num_quantizers=q,
            seq_len=int(defaults["seq_len"]),
            pretrained=bool(defaults["pretrained"]),
            experiment_id=exp_id,
        )
        checkpoint_dir_local = repo_root / "outputs" / run_name / checkpoint_folder
        checkpoint_dir_modal = Path("/vol/outputs") / run_name / checkpoint_folder
        hf_repo_id = _build_hf_repo_id(
            q=q,
            interval=args.checkpoint_interval,
            repo_prefix=args.hf_repo_prefix.strip(),
            repo_template=args.hf_repo_template.strip(),
        )
        tag_items = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tag_items.extend(
            [f"q{q}", f"seed{args.seed}", f"steps{args.steps}", f"ckpt{args.checkpoint_interval}"]
        )
        wandb_tags = ",".join(dict.fromkeys(tag_items))

        uploader_pid = None
        uploader_log = ""
        modal_app_id = ""
        launch_returncode = 0

        if args.mode == "local":
            cmd = [
                args.torchrun_bin,
                "--nproc_per_node=1",
                "-m",
                "torchtitan.train",
                "--job.config_file",
                args.config,
                "--experiment.id",
                exp_id,
                "--experiment.phase",
                args.phase,
                "--training.steps",
                str(args.steps),
                "--model.num_quantizers",
                str(q),
                "--training.seed",
                str(args.seed),
                "--training.overfit_num_samples",
                str(args.overfit_num_samples),
                "--training.minimal_media_logging",
                "true",
                "--training.log_target_media",
                "false",
                "--training.log_unconstrained_named_media",
                "false",
                "--checkpoint.enable_checkpoint",
                "true",
                "--checkpoint.interval",
                str(args.checkpoint_interval),
                "--checkpoint.last_save_in_hf",
                "true",
                "--checkpoint.keep_latest_k",
                str(args.keep_latest_k),
                "--checkpoint.folder",
                checkpoint_folder,
            ]
            if args.deterministic:
                cmd.extend(["--training.deterministic", "true"])
            if args.dataset_path:
                cmd.extend(["--training.dataset_path", args.dataset_path])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["WANDB_RUN_GROUP"] = wandb_group
            env["WANDB_TAGS"] = wandb_tags

            proc = None
            if not args.dry_run:
                with log_path.open("w", encoding="utf-8") as logf:
                    proc = subprocess.Popen(  # noqa: S603
                        cmd,
                        cwd=str(repo_root),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                if hf_repo_id:
                    uploader_log_path = run_dir / "hf_upload.log"
                    upload_cmd = [
                        args.python_bin,
                        "scripts/exp/upload_checkpoints_hf.py",
                        "--checkpoint-dir",
                        str(checkpoint_dir_local),
                        "--repo-id",
                        hf_repo_id,
                    ]
                    if args.hf_private:
                        upload_cmd.append("--private")
                    with uploader_log_path.open("w", encoding="utf-8") as upf:
                        uploader_proc = subprocess.Popen(  # noqa: S603
                            upload_cmd,
                            cwd=str(repo_root),
                            stdout=upf,
                            stderr=subprocess.STDOUT,
                            env=env,
                        )
                    uploader_pid = uploader_proc.pid
                    uploader_log = str(uploader_log_path.relative_to(repo_root))
            pid = proc.pid if proc is not None else None

        else:
            cmd = [
                *shlex.split(args.modal_cli),
                "run",
                "--detach",
                args.modal_entry,
                "--path",
                args.modal_path,
                "--experiment-id",
                exp_id,
                "--steps",
                str(args.steps),
                "--num-quantizers",
                str(q),
                "--seed",
                str(args.seed),
                "--overfit-num-samples",
                str(args.overfit_num_samples),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
                "--checkpoint-keep-latest-k",
                str(args.keep_latest_k),
                "--checkpoint-folder",
                checkpoint_folder,
                "--wandb-group",
                wandb_group,
                "--wandb-tags",
                wandb_tags,
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            if args.dataset_path:
                cmd.extend(["--dataset-path", args.dataset_path])
            if hf_repo_id:
                cmd.extend(["--hf-repo-id", hf_repo_id])
            if hf_repo_id and args.hf_collection_slug.strip():
                cmd.extend(["--hf-collection-slug", args.hf_collection_slug.strip()])
            if args.hf_private:
                cmd.append("--hf-repo-private")

            pid = None
            if not args.dry_run:
                proc_res = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy(),
                )
                launch_returncode = int(proc_res.returncode)
                output = proc_res.stdout
                log_path.write_text(output, encoding="utf-8")
                modal_app_id = _parse_modal_app_id(output)
                if launch_returncode != 0:
                    print(
                        f"[warn] modal launch failed for q={q} rc={launch_returncode}",
                        file=sys.stderr,
                    )
            else:
                log_path.write_text("(dry-run)\n", encoding="utf-8")

        launched.append(
            {
                "mode": args.mode,
                "experiment_id": exp_id,
                "quantizers": q,
                "gpu_label": gpu,
                "seed": args.seed,
                "wandb_group": wandb_group,
                "wandb_tags": wandb_tags,
                "pid": pid,
                "modal_app_id": modal_app_id,
                "launch_returncode": launch_returncode,
                "uploader_pid": uploader_pid,
                "hf_repo_id": hf_repo_id,
                "hf_collection_slug": args.hf_collection_slug.strip(),
                "checkpoint_folder": checkpoint_folder,
                "checkpoint_dir_local": str(checkpoint_dir_local),
                "checkpoint_dir_modal": str(checkpoint_dir_modal),
                "run_name": run_name,
                "command": cmd,
                "command_str": " ".join(shlex.quote(c) for c in cmd),
                "log": str(log_path.relative_to(repo_root)),
                "uploader_log": uploader_log,
                "launched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )

    out = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "wandb_group": wandb_group,
        "hf_collection_slug": args.hf_collection_slug.strip(),
        "runs": launched,
    }
    out_path = repo_root / "experiments" / "runs" / f"{args.experiment_prefix}-{ts}-matrix.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
