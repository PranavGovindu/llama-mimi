#!/usr/bin/env python3
"""Physically migrate experiment runs into codec-specific folders and rebuild indexes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _infer_codec_from_text(*values: str) -> str:
    haystack = " ".join(v.lower() for v in values if v)
    if "spark" in haystack or "bicodec" in haystack:
        return "spark_bicodec"
    if "s1" in haystack or "dac" in haystack:
        return "s1_dac"
    return "mimi"


def _infer_codec_for_run(run_path: Path) -> str:
    start = _load_json(run_path / "manifest_start.json")
    end = _load_json(run_path / "manifest_end.json")
    ids = _load_json(run_path / "run_ids.json")
    return _infer_codec_from_text(
        run_path.name,
        str(start.get("codec", "")),
        str(end.get("codec", "")),
        str(start.get("config", "")),
        str(start.get("modal_path", "")),
        str(start.get("phase", "")),
        str(start.get("question", "")),
        str(start.get("tags", "")),
        str(ids),
    )


def _move_item(src: Path, dst: Path, execute: bool) -> str:
    if dst.exists():
        return "target_exists"
    if not execute:
        return "planned"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return "moved"


def _is_codec_root(entry: Path) -> bool:
    if not entry.is_dir():
        return False
    if entry.name in {"__pycache__"}:
        return False
    # Any first-level run folder that already has an index is treated as a codec root.
    return (entry / "index.jsonl").exists()


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _abs(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base / path


def _rewrite_indexes(runs_root: Path, execute: bool) -> dict[str, int]:
    agg_path = runs_root / "index.jsonl"
    rows = _load_jsonl(agg_path)
    per_codec: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        codec = str(row.get("codec", "")).strip()
        if not codec:
            codec = _infer_codec_from_text(
                str(row.get("experiment_id", "")),
                str(row.get("config", "")),
                str(row.get("modal_path", "")),
                str(row.get("phase", "")),
                str(row.get("tags", "")),
            )
        row["codec"] = codec
        per_codec.setdefault(codec, []).append(row)

    if execute:
        for codec, codec_rows in per_codec.items():
            codec_index = runs_root / codec / "index.jsonl"
            codec_index.parent.mkdir(parents=True, exist_ok=True)
            codec_index.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in codec_rows)
                + ("\n" if codec_rows else ""),
                encoding="utf-8",
            )

        # Keep aggregate for backward compatibility.
        merged: list[dict[str, Any]] = []
        for codec in sorted(per_codec.keys()):
            merged.extend(per_codec[codec])
        agg_path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in merged)
            + ("\n" if merged else ""),
            encoding="utf-8",
        )

    return {k: len(v) for k, v in sorted(per_codec.items())}


def _rollback(report_path: Path, repo_root: Path, execute: bool) -> None:
    report = _load_json(report_path)
    moves = report.get("moves", [])
    for item in moves:
        if item.get("status") not in {"moved", "planned"}:
            continue
        src = _abs(str(item["target"]), repo_root) if item.get("target") else None
        dst = _abs(str(item["source"]), repo_root) if item.get("source") else None
        if not src or not dst:
            continue
        if execute and src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"rolled back {src} -> {dst}")
        elif not execute:
            print(f"would rollback {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate experiment runs to codec layout.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rollback-manifest", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "experiments" / "runs"
    report_path = runs_root / "migration_report.json"

    if args.rollback_manifest.strip():
        manifest = Path(args.rollback_manifest.strip())
        _rollback(manifest, repo_root=repo_root, execute=args.execute)
        return

    codec_roots = {
        entry.name
        for entry in runs_root.iterdir()
        if _is_codec_root(entry)
    }
    moves: list[dict[str, Any]] = []

    # Move run directories.
    for entry in sorted(runs_root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.name in codec_roots:
            continue
        if entry.name in {"index.jsonl", "migration_report.json"}:
            continue

        if entry.is_dir():
            # Migrate only legacy experiment run directories (exp-*) at root level.
            if _is_codec_root(entry) or not entry.name.startswith("exp-"):
                continue
            codec = _infer_codec_for_run(entry)
            target = runs_root / codec / entry.name
            status = _move_item(entry, target, execute=args.execute)
            moves.append(
                {
                    "type": "dir",
                    "source": _rel(entry, repo_root),
                    "target": _rel(target, repo_root),
                    "codec": codec,
                    "status": status,
                }
            )
        elif entry.is_file() and entry.name.endswith("-matrix.json"):
            codec = _infer_codec_from_text(entry.name)
            target = runs_root / codec / entry.name
            status = _move_item(entry, target, execute=args.execute)
            moves.append(
                {
                    "type": "file",
                    "source": _rel(entry, repo_root),
                    "target": _rel(target, repo_root),
                    "codec": codec,
                    "status": status,
                }
            )

    index_counts = _rewrite_indexes(runs_root, execute=args.execute)

    report = {
        "execute": bool(args.execute),
        "moves": moves,
        "index_counts": index_counts,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.execute:
        report_path.write_text(report_text, encoding="utf-8")
        print(f"wrote migration report -> {report_path}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
