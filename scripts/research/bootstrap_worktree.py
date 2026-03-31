#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path


def _run(
    cmd: list[str],
    cwd: Path,
    input_text: str | None = None,
    *,
    strip: bool = True,
) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip() if strip else proc.stdout


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _branch_exists(repo_root: Path, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a dedicated research worktree seeded from the current dirty state."
    )
    parser.add_argument(
        "--worktree-path",
        default="",
        help="Destination path. Defaults to <repo-parent>/<repo-name>-research.",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="Branch name for the research worktree. Defaults to research/<timestamp>.",
    )
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument(
        "--copy-untracked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy untracked files from the current worktree into the new worktree.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = args.branch.strip() or f"research/{timestamp}"
    worktree_path = Path(args.worktree_path).expanduser()
    if not args.worktree_path.strip():
        worktree_path = repo_root.parent / f"{repo_root.name}-research"
    worktree_path = worktree_path.resolve()

    tracked_patch = _run(["git", "diff", "--binary"], repo_root, strip=False)
    staged_patch = _run(["git", "diff", "--cached", "--binary"], repo_root, strip=False)
    untracked_paths = []
    if args.copy_untracked:
        raw_untracked = _run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            repo_root,
        )
        untracked_paths = [p for p in raw_untracked.splitlines() if p.strip()]

    payload = {
        "repo_root": str(repo_root),
        "worktree_path": str(worktree_path),
        "branch": branch,
        "base_ref": args.base_ref,
        "copy_untracked": args.copy_untracked,
        "tracked_patch_bytes": len(tracked_patch.encode("utf-8")),
        "staged_patch_bytes": len(staged_patch.encode("utf-8")),
        "untracked_paths": untracked_paths,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if worktree_path.exists():
        raise SystemExit(f"worktree path already exists: {worktree_path}")

    worktree_cmd = ["git", "worktree", "add"]
    if _branch_exists(repo_root, branch):
        worktree_cmd.extend([str(worktree_path), branch])
    else:
        worktree_cmd.extend(["-b", branch, str(worktree_path), args.base_ref])
    _run(worktree_cmd, repo_root)

    try:
        if staged_patch:
            _run(["git", "apply", "--allow-empty"], worktree_path, input_text=staged_patch)
        if tracked_patch:
            _run(["git", "apply", "--allow-empty"], worktree_path, input_text=tracked_patch)
        for rel_path in untracked_paths:
            src = repo_root / rel_path
            dst = worktree_path / rel_path
            if src.exists():
                _copy_path(src, dst)
    except Exception:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=str(repo_root),
            check=False,
        )
        raise

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
