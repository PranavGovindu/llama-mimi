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


def _infer_codec(
    explicit_codec: str,
    config: str,
    modal_path: str,
    experiment_id: str,
) -> str:
    if explicit_codec.strip():
        return explicit_codec.strip()
    haystack = " ".join(
        [
            config.lower(),
            modal_path.lower(),
            experiment_id.lower(),
        ]
    )
    if "qwen" in haystack:
        return "qwen_codec"
    if "spark" in haystack or "bicodec" in haystack:
        return "spark_bicodec"
    if "dualcodec" in haystack or "12hz" in haystack or "25hz" in haystack:
        return "dualcodec"
    if "s1" in haystack or "dac" in haystack:
        return "s1_dac"
    return "mimi"


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


def _parse_override_items(items: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for raw in items:
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"override must be KEY=VALUE, got: {raw!r}")
        key, value_raw = item.split("=", 1)
        key = key.strip()
        value_raw = value_raw.strip()
        if not key:
            raise SystemExit(f"override key is empty in: {raw!r}")
        try:
            value = json.loads(value_raw)
        except json.JSONDecodeError:
            value = value_raw
        overrides[key] = value
    return overrides


def _format_override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch reproducible TinyAya TTS Lab experiments.")
    parser.add_argument("--mode", choices=["local", "modal"], required=True)
    parser.add_argument(
        "--config",
        default="recipes/tinyaya/mimi/overfit/tinyaya_q1_fleurs_overfit_1sample_strict.toml",
    )
    parser.add_argument("--modal-path", default="overfit_strict")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--phase", default="adhoc")
    parser.add_argument("--stage", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--track", default="")
    parser.add_argument("--axis", default="")
    parser.add_argument("--family", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--brief-path", default="")
    parser.add_argument("--baseline-experiment-id", default="")
    parser.add_argument("--owner", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--codec", default="")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable KEY=VALUE override passed through to torchtitan.train.",
    )
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
    overrides = _parse_override_items(args.override)
    codec = _infer_codec(
        explicit_codec=args.codec,
        config=args.config,
        modal_path=args.modal_path,
        experiment_id=exp_id,
    )

    run_dir = repo_root / "experiments" / "runs" / codec / exp_id
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
            "--experiment.campaign_id",
            args.campaign_id,
            "--experiment.phase",
            args.phase,
            "--experiment.stage",
            args.stage,
            "--experiment.variant",
            args.variant,
            "--experiment.track",
            args.track,
            "--experiment.axis",
            args.axis,
            "--experiment.family",
            args.family,
            "--experiment.question",
            args.question,
            "--experiment.hypothesis",
            args.hypothesis,
            "--experiment.brief_path",
            args.brief_path,
            "--experiment.baseline_experiment_id",
            args.baseline_experiment_id,
            "--experiment.owner",
            args.owner,
        ]
        if args.tags:
            cmd.extend(["--experiment.tags", args.tags])
        for key, value in overrides.items():
            cmd.extend([f"--{key}", _format_override_value(value)])
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
        if args.campaign_id:
            cmd.extend(["--campaign-id", args.campaign_id])
        if args.phase:
            cmd.extend(["--phase", args.phase])
        if args.stage:
            cmd.extend(["--stage", args.stage])
        if args.variant:
            cmd.extend(["--variant", args.variant])
        if args.track:
            cmd.extend(["--track", args.track])
        if args.axis:
            cmd.extend(["--axis", args.axis])
        if args.family:
            cmd.extend(["--family", args.family])
        if args.question:
            cmd.extend(["--question", args.question])
        if args.hypothesis:
            cmd.extend(["--hypothesis", args.hypothesis])
        if args.brief_path:
            cmd.extend(["--brief-path", args.brief_path])
        if args.baseline_experiment_id:
            cmd.extend(["--baseline-experiment-id", args.baseline_experiment_id])
        if args.owner:
            cmd.extend(["--owner", args.owner])
        if args.tags:
            cmd.extend(["--tags", args.tags])
        if args.steps > 0:
            cmd.extend(["--steps", str(args.steps)])
        if overrides:
            cmd.extend(["--overrides-json", json.dumps(overrides, ensure_ascii=False)])

    metadata = {
        "experiment_id": exp_id,
        "campaign_id": args.campaign_id,
        "mode": args.mode,
        "phase": args.phase,
        "stage": args.stage,
        "variant": args.variant,
        "track": args.track,
        "axis": args.axis,
        "family": args.family,
        "question": args.question,
        "hypothesis": args.hypothesis,
        "brief_path": args.brief_path,
        "baseline_experiment_id": args.baseline_experiment_id,
        "owner": args.owner,
        "codec": codec,
        "tags": tags,
        "config": args.config,
        "modal_path": args.modal_path,
        "steps": args.steps,
        "overrides": overrides,
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
