#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
from pathlib import Path


def _run_dir(repo_root: Path, experiment_id: str) -> Path:
    run_dir = repo_root / "experiments" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch parallel local overfit runs for Q5/Q6/Q7/Q8 with fixed GPU mapping."
    )
    parser.add_argument("--config", default="config/tinyaya_q8_download_overfit_1sample.toml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--quantizers", default="5,6,7,8")
    parser.add_argument("--gpus", default="5,6,7,8")
    parser.add_argument("--experiment-prefix", default="exp-codebook-matrix")
    parser.add_argument("--phase", default="overfit_codebook_matrix")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--hf-repo-prefix", default="")
    parser.add_argument(
        "--hf-repo-template",
        default="",
        help="Template for HF repo id, supports {q} and {interval}, e.g. rumik-ai/q{q}-{interval}-aya",
    )
    parser.add_argument("--hf-private", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    quantizers = [int(x.strip()) for x in args.quantizers.split(",") if x.strip()]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(quantizers) != len(gpus):
        raise SystemExit("quantizers and gpus must have equal length")

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    launched: list[dict] = []

    for q, gpu in zip(quantizers, gpus):
        exp_id = f"{args.experiment_prefix}-{ts}-q{q}"
        run_dir = _run_dir(repo_root, exp_id)
        log_path = run_dir / "launch.log"

        cmd = [
            "torchrun",
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
            "--training.minimal_media_logging",
            "true",
            "--checkpoint.enable_checkpoint",
            "true",
            "--checkpoint.interval",
            str(args.checkpoint_interval),
            "--checkpoint.last_save_in_hf",
            "true",
            "--checkpoint.keep_latest_k",
            "2",
            "--checkpoint.folder",
            f"checkpoint_q{q}_matrix",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(repo_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )

        uploader_pid = None
        uploader_log = ""
        hf_repo_id = ""
        if args.hf_repo_template:
            hf_repo_id = args.hf_repo_template.format(
                q=q,
                interval=args.checkpoint_interval,
            )
        elif args.hf_repo_prefix:
            hf_repo_id = f"{args.hf_repo_prefix}-q{q}"
        if hf_repo_id:
            checkpoint_dir = repo_root / "outputs" / f"checkpoint_q{q}_matrix"
            uploader_log_path = run_dir / "hf_upload.log"
            upload_cmd = [
                "python",
                "scripts/exp/upload_checkpoints_hf.py",
                "--checkpoint-dir",
                str(checkpoint_dir),
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

        launched.append(
            {
                "experiment_id": exp_id,
                "quantizers": q,
                "gpu": gpu,
                "pid": proc.pid,
                "uploader_pid": uploader_pid,
                "hf_repo_id": hf_repo_id,
                "command": cmd,
                "command_str": " ".join(shlex.quote(c) for c in cmd),
                "log": str(log_path.relative_to(repo_root)),
                "uploader_log": uploader_log,
                "launched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )

    out = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": launched,
    }
    out_path = repo_root / "experiments" / "runs" / f"{args.experiment_prefix}-{ts}-matrix.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
