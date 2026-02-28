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


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize run registry entry.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--status", default="completed", choices=["running", "stopped", "failed", "completed"]
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--wandb-url", default="")
    parser.add_argument("--modal-app-id", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / "experiments" / "runs" / args.experiment_id
    start = _load_json(run_dir / "manifest_start.json")
    end = _load_json(run_dir / "manifest_end.json")
    ids = _load_json(run_dir / "run_ids.json")

    row = {
        "date": dt.datetime.utcnow().strftime("%Y-%m-%d"),
        "experiment_id": args.experiment_id,
        "status": args.status,
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

    index_path = repo_root / "experiments" / "runs" / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"appended {index_path}")
    subprocess.run(
        ["python", "scripts/exp/render_log.py"],
        cwd=str(repo_root),
        check=False,
    )


if __name__ == "__main__":
    main()
