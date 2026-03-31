#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torchtitan.tools.research_eval import write_tsv


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


FIELDNAMES = [
    "experiment_id",
    "codec",
    "campaign_id",
    "phase",
    "stage",
    "variant",
    "track",
    "axis",
    "family",
    "baseline_experiment_id",
    "seed",
    "step",
    "eval_pack",
    "config_path",
    "dataset_manifest_hash",
    "eval_manifest_hash",
    "seq_len",
    "quantizers",
    "max_audio_seconds",
    "lr",
    "warmup_steps",
    "global_batch_size",
    "local_batch_size",
    "target_audio_tokens_per_update",
    "train_dataset",
    "train_dataset_path",
    "validation_dataset",
    "validation_split",
    "wer_mean",
    "cer_mean",
    "utmos_mean",
    "dnsmos_p808_mean",
    "dnsmos_ovr_mean",
    "speaker_similarity_mean",
    "salmon_mean",
    "mel_l1_mean",
    "mel_l2_mean",
    "mel_cosine_mean",
    "sample_count",
    "generated_count",
    "malformed_decode_rate",
    "frame_ratio_mean",
    "target_coverage_total_mean",
    "generated_coverage_total_mean",
    "coverage_q_min_mean",
    "coverage_q_abs_diff_max_mean",
    "delta_wer",
    "delta_cer",
    "delta_utmos",
    "delta_dnsmos_ovr",
    "delta_mel_l1",
    "gate_pass",
    "rank",
    "decision",
]


def _sort_numeric(value: Any, *, reverse: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("-inf") if reverse else float("inf")
    return -number if reverse else number


def _abs_duration_error(value: Any) -> float:
    try:
        return abs(float(value) - 1.0)
    except (TypeError, ValueError):
        return float("inf")


def _maybe_target_audio_tokens_per_update(cfg: dict[str, Any]) -> int | str:
    training = cfg.get("training", {})
    global_batch_size = int(training.get("global_batch_size", 0) or 0)
    max_audio_seconds = int(training.get("max_audio_seconds", 0) or 0)
    quantizers = int(cfg.get("model", {}).get("num_quantizers", 0) or 0)
    if global_batch_size <= 0 or max_audio_seconds <= 0 or quantizers <= 0:
        return ""
    return int(global_batch_size * max_audio_seconds * 12.5 * quantizers)


def _dataset_hashes(run_dir: Path, validation_split: str) -> tuple[str, str]:
    dataset_manifest = _load_json(run_dir / "run_snapshot" / "dataset_manifest.json")
    dataset_hash = str(dataset_manifest.get("fingerprint_sha256", "") or "").strip()
    split_hashes = dataset_manifest.get("sample_id_sha256_by_split", {})
    eval_hash = ""
    if isinstance(split_hashes, dict):
        eval_hash = str(split_hashes.get(validation_split, "") or "").strip()
    return dataset_hash, eval_hash


def _summary_rows_for_run(run_dir: Path, codec: str) -> list[dict[str, Any]]:
    start = _load_json(run_dir / "manifest_start.json")
    resolved = _load_json(run_dir / "run_snapshot" / "resolved_config.json")
    training = resolved.get("training", {})
    optimizer = resolved.get("optimizer", {})
    lr_scheduler = resolved.get("lr_scheduler", {})
    validation = resolved.get("validation", {})
    validation_split = str(validation.get("split", "validation") or "validation")
    dataset_manifest_hash, eval_manifest_hash = _dataset_hashes(run_dir, validation_split)
    step_dirs = sorted((run_dir / "full_eval").glob("step_*/summary.json"))
    rows: list[dict[str, Any]] = []
    for summary_path in step_dirs:
        summary = _load_json(summary_path)
        if not summary:
            continue
        sample_count = int(summary.get("sample_count", 0) or 0)
        generated_count = int(summary.get("generated_count", 0) or 0)
        malformed_rate = summary.get("malformed_decode_rate", "")
        gate_pass = ""
        if sample_count > 0:
            gate_pass = bool(generated_count == sample_count and float(malformed_rate or 1.0) == 0.0)
        rows.append(
            {
                "experiment_id": start.get("experiment_id", run_dir.name),
                "codec": codec,
                "campaign_id": start.get("campaign_id", ""),
                "phase": start.get("phase", ""),
                "stage": start.get("stage", ""),
                "variant": start.get("variant", ""),
                "track": start.get("track", ""),
                "axis": start.get("axis", ""),
                "family": start.get("family", ""),
                "baseline_experiment_id": start.get("baseline_experiment_id", ""),
                "seed": training.get("seed", ""),
                "step": summary.get("step", ""),
                "eval_pack": summary.get("eval_pack", "validation"),
                "config_path": start.get("config", ""),
                "dataset_manifest_hash": dataset_manifest_hash,
                "eval_manifest_hash": eval_manifest_hash,
                "seq_len": training.get("seq_len", ""),
                "quantizers": resolved.get("model", {}).get("num_quantizers", ""),
                "max_audio_seconds": training.get("max_audio_seconds", ""),
                "lr": optimizer.get("lr", ""),
                "warmup_steps": lr_scheduler.get("warmup_steps", ""),
                "global_batch_size": training.get("global_batch_size", ""),
                "local_batch_size": training.get("local_batch_size", ""),
                "target_audio_tokens_per_update": _maybe_target_audio_tokens_per_update(resolved),
                "train_dataset": training.get("dataset", ""),
                "train_dataset_path": training.get("dataset_path", ""),
                "validation_dataset": validation.get("dataset", ""),
                "validation_split": validation_split,
                "wer_mean": summary.get("wer_mean", ""),
                "cer_mean": summary.get("cer_mean", ""),
                "utmos_mean": summary.get("utmos_mean", ""),
                "dnsmos_p808_mean": summary.get("dnsmos_p808_mean", ""),
                "dnsmos_ovr_mean": summary.get("dnsmos_ovr_mean", ""),
                "speaker_similarity_mean": summary.get("speaker_similarity_mean", ""),
                "salmon_mean": summary.get("salmon_mean", ""),
                "mel_l1_mean": summary.get("mel_l1_mean", ""),
                "mel_l2_mean": summary.get("mel_l2_mean", ""),
                "mel_cosine_mean": summary.get("mel_cosine_mean", ""),
                "sample_count": sample_count,
                "generated_count": generated_count,
                "malformed_decode_rate": malformed_rate,
                "frame_ratio_mean": summary.get("frame_ratio_mean", ""),
                "target_coverage_total_mean": summary.get("target_coverage_total_mean", ""),
                "generated_coverage_total_mean": summary.get("generated_coverage_total_mean", ""),
                "coverage_q_min_mean": summary.get("coverage_q_min_mean", ""),
                "coverage_q_abs_diff_max_mean": summary.get("coverage_q_abs_diff_max_mean", ""),
                "delta_wer": "",
                "delta_cer": "",
                "delta_utmos": "",
                "delta_dnsmos_ovr": "",
                "delta_mel_l1": "",
                "gate_pass": gate_pass,
                "rank": "",
                "decision": "",
            }
        )
    return rows


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("campaign_id", "")).strip(),
            str(row.get("stage", "")).strip(),
            str(row.get("step", "")).strip(),
            str(row.get("eval_pack", "")).strip(),
        )
        groups.setdefault(key, []).append(row)

    for _, group_rows in groups.items():
        group_rows.sort(
            key=lambda row: (
                _sort_numeric(row.get("wer_mean")),
                _sort_numeric(row.get("cer_mean")),
                _sort_numeric(row.get("malformed_decode_rate")),
                _abs_duration_error(row.get("frame_ratio_mean")),
                _sort_numeric(row.get("dnsmos_ovr_mean"), reverse=True),
                _sort_numeric(row.get("utmos_mean"), reverse=True),
                _sort_numeric(row.get("mel_l1_mean")),
                _sort_numeric(row.get("mel_cosine_mean"), reverse=True),
                _sort_numeric(row.get("salmon_mean"), reverse=True),
                _sort_numeric(row.get("dnsmos_p808_mean"), reverse=True),
                _sort_numeric(row.get("coverage_q_abs_diff_max_mean")),
                str(row.get("experiment_id", "")),
            )
        )
        for index, row in enumerate(group_rows, start=1):
            row["rank"] = index


def _assign_anchor_deltas(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("campaign_id", "")).strip(),
            str(row.get("stage", "")).strip(),
            str(row.get("step", "")).strip(),
            str(row.get("eval_pack", "")).strip(),
        )
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        anchor_rows = [row for row in group_rows if str(row.get("variant", "")).strip() == "anchor"]
        if not anchor_rows:
            continue

        def _mean_for(key: str) -> float | None:
            values: list[float] = []
            for row in anchor_rows:
                try:
                    values.append(float(row.get(key)))
                except (TypeError, ValueError):
                    continue
            if not values:
                return None
            return sum(values) / len(values)

        anchor_wer = _mean_for("wer_mean")
        anchor_cer = _mean_for("cer_mean")
        anchor_utmos = _mean_for("utmos_mean")
        anchor_dnsmos = _mean_for("dnsmos_ovr_mean")
        anchor_mel_l1 = _mean_for("mel_l1_mean")

        for row in group_rows:
            for field_name, anchor_value, row_key in (
                ("delta_wer", anchor_wer, "wer_mean"),
                ("delta_cer", anchor_cer, "cer_mean"),
                ("delta_utmos", anchor_utmos, "utmos_mean"),
                ("delta_dnsmos_ovr", anchor_dnsmos, "dnsmos_ovr_mean"),
                ("delta_mel_l1", anchor_mel_l1, "mel_l1_mean"),
            ):
                if anchor_value is None:
                    row[field_name] = ""
                    continue
                try:
                    row[field_name] = float(row.get(row_key)) - anchor_value
                except (TypeError, ValueError):
                    row[field_name] = ""


def _collect_rows(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    per_codec: dict[str, list[dict[str, Any]]] = {}
    for codec_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        codec = codec_dir.name
        rows: list[dict[str, Any]] = []
        for run_dir in sorted(path for path in codec_dir.iterdir() if path.is_dir()):
            rows.extend(_summary_rows_for_run(run_dir, codec))
        if rows:
            _assign_ranks(rows)
            _assign_anchor_deltas(rows)
            rows.sort(key=lambda row: (str(row.get("experiment_id", "")), int(row.get("step", 0) or 0)))
            per_codec[codec] = rows
    return per_codec


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full-eval TSV leaderboards.")
    parser.add_argument("--runs-root", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runs_root = (
        Path(args.runs_root).resolve()
        if args.runs_root
        else repo_root / "experiments" / "runs"
    )
    per_codec = _collect_rows(runs_root)

    merged_rows: list[dict[str, Any]] = []
    for codec, rows in per_codec.items():
        merged_rows.extend(rows)
        write_tsv(runs_root / codec / "results.tsv", rows, FIELDNAMES)

    merged_rows.sort(key=lambda row: (str(row.get("codec", "")), str(row.get("experiment_id", "")), int(row.get("step", 0) or 0)))
    write_tsv(repo_root / "experiments" / "results.tsv", merged_rows, FIELDNAMES)
    print(json.dumps({"rows": len(merged_rows), "per_codec": {k: len(v) for k, v in per_codec.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
