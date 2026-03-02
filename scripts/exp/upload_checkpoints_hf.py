#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload step checkpoints to Hugging Face repo.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--idle-exit-seconds", type=int, default=600)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    uploaded: set[str] = set()
    last_new = time.time()

    while True:
        steps = _discover_steps(ckpt_dir)
        new_any = False
        for step_dir in steps:
            if step_dir.name in uploaded:
                continue
            if not (step_dir / ".metadata").exists() and not (step_dir / "model.safetensors.index.json").exists():
                continue
            api.upload_folder(
                folder_path=str(step_dir),
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo=step_dir.name,
                commit_message=f"upload {step_dir.name}",
            )
            uploaded.add(step_dir.name)
            new_any = True
            print(f"uploaded {step_dir.name} -> {args.repo_id}", flush=True)

        if new_any:
            last_new = time.time()
        if time.time() - last_new > args.idle_exit_seconds:
            print("no new checkpoints; exiting uploader", flush=True)
            break
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
