#!/usr/bin/env python3
import json
from pathlib import Path


def load_rows(index_path: Path) -> list[dict]:
    rows = []
    if not index_path.exists():
        return rows
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def render_markdown(rows: list[dict], source_label: str) -> str:
    def _v(row: dict, key: str) -> str:
        value = row.get(key, "")
        text = str(value).strip()
        return text if text else "-"

    lines = [
        "# Experiment Log",
        "",
        f"Auto-generated from `{source_label}`.",
        "",
    ]
    for row in reversed(rows):
        lines.extend(
            [
                f"## {_v(row, 'date')} - {_v(row, 'experiment_id')}",
                f"- codec: {_v(row, 'codec')}",
                f"- status: {_v(row, 'status')}",
                f"- phase: {_v(row, 'phase')}",
                f"- mode: {_v(row, 'mode')}",
                f"- config: {_v(row, 'config')}",
                f"- modal_path: {_v(row, 'modal_path')}",
                f"- wandb_run_id: {_v(row, 'wandb_run_id')}",
                f"- wandb_url: {_v(row, 'wandb_url')}",
                f"- modal_app_id: {_v(row, 'modal_app_id')}",
                f"- notes: {_v(row, 'notes')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _collect_codec_rows(runs_root: Path) -> dict[str, list[dict]]:
    per_codec: dict[str, list[dict]] = {}
    for path in sorted(runs_root.glob("*/index.jsonl")):
        codec = path.parent.name
        per_codec[codec] = load_rows(path)
    return per_codec


def _merge_rows(per_codec: dict[str, list[dict]]) -> list[dict]:
    by_exp: dict[str, dict] = {}
    for codec, rows in per_codec.items():
        for row in rows:
            exp_id = str(row.get("experiment_id", "")).strip()
            if not exp_id:
                continue
            merged = {**row}
            merged.setdefault("codec", codec)
            by_exp[exp_id] = merged
    merged_rows = list(by_exp.values())
    merged_rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("experiment_id", ""))))
    return merged_rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "experiments" / "runs"
    agg_index_path = runs_root / "index.jsonl"

    per_codec = _collect_codec_rows(runs_root)
    if not per_codec:
        rows = load_rows(agg_index_path)
        out_path = repo_root / "experiments" / "EXPERIMENT_LOG.md"
        out_path.write_text(render_markdown(rows, "experiments/runs/index.jsonl"), encoding="utf-8")
        print(f"rendered {len(rows)} rows -> {out_path}")
        return

    merged_rows = _merge_rows(per_codec)
    agg_index_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in merged_rows) + ("\n" if merged_rows else ""),
        encoding="utf-8",
    )

    out_path = repo_root / "experiments" / "EXPERIMENT_LOG.md"
    out_path.write_text(
        render_markdown(merged_rows, "experiments/runs/<codec>/index.jsonl"),
        encoding="utf-8",
    )

    for codec, rows in sorted(per_codec.items()):
        codec_log_path = repo_root / "experiments" / f"EXPERIMENT_LOG_{codec.upper()}.md"
        codec_log_path.write_text(
            render_markdown(rows, f"experiments/runs/{codec}/index.jsonl"),
            encoding="utf-8",
        )

    print(f"rendered {len(merged_rows)} merged rows -> {out_path}")
    print(f"wrote aggregate index -> {agg_index_path}")


if __name__ == "__main__":
    main()
