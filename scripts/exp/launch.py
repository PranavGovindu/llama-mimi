#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
from pathlib import Path


def _timestamp_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("exp-%Y%m%d-%H%M%S")


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def _git_info(repo_root: Path) -> dict:
    def _git(args: list[str]) -> str:
        code, out = _run(["git", *args], repo_root)
        if code != 0:
            return ""
        return out.strip()

    return {
        "sha": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git(["status", "--short"]),
        "dirty": bool(_git(["status", "--short"])),
    }


def _parse_ids(log_text: str) -> dict:
    wandb_url = ""
    wandb_run_id = ""
    modal_app_id = ""
    m_url = re.findall(r"https://wandb\.ai/\S+/runs/[a-zA-Z0-9]+", log_text)
    if m_url:
        wandb_url = m_url[-1]
    m = re.findall(r"/runs/([a-zA-Z0-9]+)", wandb_url or "")
    if m:
        wandb_run_id = m[-1]
    m2 = re.findall(r"(ap-[A-Za-z0-9]+)", log_text)
    if m2:
        modal_app_id = m2[-1]
    return {
        "wandb_run_id": wandb_run_id,
        "wandb_url": wandb_url,
        "modal_app_id": modal_app_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch reproducible TinyAya experiments.")
    parser.add_argument("--mode", choices=["local", "modal"], required=True)
    parser.add_argument("--config", default="config/tinyaya_q1_fleurs_overfit_1sample_strict.toml")
    parser.add_argument("--modal-path", default="overfit_strict")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--phase", default="adhoc")
    parser.add_argument("--question", default="")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--owner", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run modal jobs detached so local disconnect does not stop training.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exp_id = args.experiment_id.strip() or _timestamp_id()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    run_dir = repo_root / "experiments" / "runs" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "local":
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
            "--experiment.question",
            args.question,
            "--experiment.hypothesis",
            args.hypothesis,
            "--experiment.owner",
            args.owner,
        ]
    else:
        cmd = [
            "modal",
            "run",
            "--detach" if args.detach else "",
            "modal/app.py::train",
            "--path",
            args.modal_path,
            "--experiment-id",
            exp_id,
        ]
        cmd = [c for c in cmd if c]
        if args.steps > 0:
            cmd.extend(["--steps", str(args.steps)])

    metadata = {
        "experiment_id": exp_id,
        "mode": args.mode,
        "phase": args.phase,
        "question": args.question,
        "hypothesis": args.hypothesis,
        "owner": args.owner,
        "tags": tags,
        "config": args.config,
        "modal_path": args.modal_path,
        "steps": args.steps,
        "detach": args.detach,
        "command": cmd,
        "command_str": " ".join(shlex.quote(c) for c in cmd),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": _git_info(repo_root),
    }
    (run_dir / "manifest_start.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    code, output = _run(cmd, repo_root)
    (run_dir / "launch.log").write_text(output)
    parsed_ids = _parse_ids(output)
    (run_dir / "run_ids.json").write_text(json.dumps(parsed_ids, indent=2) + "\n")

    final_meta = {
        **metadata,
        **parsed_ids,
        "exit_code": code,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (run_dir / "manifest_end.json").write_text(
        json.dumps(final_meta, indent=2, sort_keys=True) + "\n"
    )
    print(output)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
