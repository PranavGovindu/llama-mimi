from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirror a Hugging Face dataset repo locally for training.dataset_path."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. Pranavz/...")
    parser.add_argument("--local-dir", required=True, help="Target local directory")
    parser.add_argument("--revision", default=None, help="Optional HF revision")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=[],
        help="Optional allow pattern. Repeatable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = args.allow_patterns or [
        "README.md",
        "dataset_manifest.json",
        "train/**/*.parquet",
        "train/*.parquet",
        "*.json",
        "*.md",
    ]

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        revision=args.revision,
    )
    payload = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "snapshot_path": snapshot_path,
        "allow_patterns": allow_patterns,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
