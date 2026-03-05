#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _infer_codec(
    explicit_codec: str,
    start: dict,
    end: dict,
    experiment_id: str,
) -> str:
    if explicit_codec.strip():
        return explicit_codec.strip()
    for key in ("codec",):
        value = (start.get(key) or end.get(key) or "").strip()
        if value:
            return value
    haystack = " ".join(
        [
            str(start.get("config", "")).lower(),
            str(start.get("modal_path", "")).lower(),
            str(experiment_id).lower(),
        ]
    )
    if "s1" in haystack or "dac" in haystack:
        return "s1_dac"
    return "mimi"


def _resolve_run_dir(repo_root: Path, experiment_id: str, codec_hint: str) -> Path:
    candidates: list[Path] = []
    if codec_hint.strip():
        candidates.append(repo_root / "experiments" / "runs" / codec_hint / experiment_id)
    candidates.append(repo_root / "experiments" / "runs" / experiment_id)

    runs_root = repo_root / "experiments" / "runs"
    if runs_root.exists():
        for codec_dir in runs_root.iterdir():
            if not codec_dir.is_dir():
                continue
            candidates.append(codec_dir / experiment_id)

    for path in candidates:
        if path.exists():
            return path
    # Default target for new records.
    return repo_root / "experiments" / "runs" / (codec_hint.strip() or "mimi") / experiment_id


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize run registry entry.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--codec", default="")
    parser.add_argument(
        "--status", default="completed", choices=["running", "stopped", "failed", "completed"]
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--wandb-url", default="")
    parser.add_argument("--modal-app-id", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = _resolve_run_dir(repo_root, args.experiment_id, args.codec)
    start = _load_json(run_dir / "manifest_start.json")
    end = _load_json(run_dir / "manifest_end.json")
    ids = _load_json(run_dir / "run_ids.json")

    codec = _infer_codec(args.codec, start, end, args.experiment_id)
    if not run_dir.exists():
        run_dir = repo_root / "experiments" / "runs" / codec / args.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "date": dt.datetime.utcnow().strftime("%Y-%m-%d"),
        "experiment_id": args.experiment_id,
        "status": args.status,
        "codec": codec,
        "mode": start.get("mode", ""),
        "phase": start.get("phase", ""),
        "question": start.get("question", ""),
        "hypothesis": start.get("hypothesis", ""),
        "owner": start.get("owner", ""),
        "tags": start.get("tags", []),
        "config": start.get("config", ""),
        "modal_path": start.get("modal_path", ""),
        "git_sha": (start.get("git") or {}).get("sha", ""),
        "wandb_run_id": args.wandb_run_id or ids.get("wandb_run_id", "") or end.get("wandb_run_id", ""),
        "wandb_url": args.wandb_url or ids.get("wandb_url", "") or end.get("wandb_url", ""),
        "modal_app_id": args.modal_app_id or ids.get("modal_app_id", "") or end.get("modal_app_id", ""),
        "notes": args.notes,
    }

    per_codec_index = repo_root / "experiments" / "runs" / codec / "index.jsonl"
    _append_jsonl(per_codec_index, row)

    # Maintain aggregate index for dashboards/backward compatibility.
    agg_index_path = repo_root / "experiments" / "runs" / "index.jsonl"
    _append_jsonl(agg_index_path, row)

    print(f"appended codec index: {per_codec_index}")
    print(f"appended aggregate index: {agg_index_path}")
    subprocess.run(
        ["python", "scripts/exp/render_log.py"],
        cwd=str(repo_root),
        check=False,
    )


if __name__ == "__main__":
    main()
