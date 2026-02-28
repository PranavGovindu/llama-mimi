#!/usr/bin/env python3
import json
from pathlib import Path


def load_rows(index_path: Path) -> list[dict]:
    rows = []
    if not index_path.exists():
        return rows
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Experiment Log",
        "",
        "Auto-generated from `experiments/runs/index.jsonl`.",
        "",
    ]
    for row in reversed(rows):
        lines.extend(
            [
                f"## {row.get('date', '')} - {row.get('experiment_id', '')}",
                f"- status: {row.get('status', '')}",
                f"- phase: {row.get('phase', '')}",
                f"- mode: {row.get('mode', '')}",
                f"- config: {row.get('config', '')}",
                f"- modal_path: {row.get('modal_path', '')}",
                f"- wandb_run_id: {row.get('wandb_run_id', '')}",
                f"- wandb_url: {row.get('wandb_url', '')}",
                f"- modal_app_id: {row.get('modal_app_id', '')}",
                f"- notes: {row.get('notes', '')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    index_path = repo_root / "experiments" / "runs" / "index.jsonl"
    out_path = repo_root / "experiments" / "EXPERIMENT_LOG.md"
    rows = load_rows(index_path)
    out_path.write_text(render_markdown(rows))
    print(f"rendered {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
