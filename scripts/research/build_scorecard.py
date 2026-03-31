#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torchtitan.tools.research_eval import metric_summary, safe_mean


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        row = _load_json(path)
        if row:
            rows.append(row)
    return rows


def _resolve_run_dir(repo_root: Path, experiment_id: str) -> Path | None:
    runs_root = repo_root / "experiments" / "runs"
    for codec_dir in sorted(runs_root.iterdir() if runs_root.exists() else []):
        candidate = codec_dir / experiment_id
        if candidate.exists():
            return candidate
    return None


def _load_latest_full_eval_summary(run_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
    if run_dir is None:
        return {}, None
    candidates = sorted((run_dir / "full_eval").glob("step_*/summary.json"))
    if not candidates:
        return {}, None
    summary_path = candidates[-1]
    return _load_json(summary_path), summary_path


def _summarize_legacy_sample_eval(eval_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_language: dict[str, dict[str, list[float] | int]] = {}
    unconstrained_decode_count = 0
    constrained_decode_count = 0
    for row in eval_rows:
        language = str(row.get("language", "")).strip() or "unknown"
        bucket = per_language.setdefault(
            language,
            {
                "samples": 0,
                "unconstrained_decode_count": 0,
                "constrained_decode_count": 0,
                "wer_unconstrained": [],
                "cer_unconstrained": [],
                "wer_constrained": [],
                "cer_constrained": [],
            },
        )
        bucket["samples"] = int(bucket["samples"]) + 1
        if int(row.get("unconstrained_frames", 0)) > 0:
            unconstrained_decode_count += 1
            bucket["unconstrained_decode_count"] = int(bucket["unconstrained_decode_count"]) + 1
        if int(row.get("constrained_frames", 0)) > 0:
            constrained_decode_count += 1
            bucket["constrained_decode_count"] = int(bucket["constrained_decode_count"]) + 1
        for key in (
            "wer_unconstrained",
            "cer_unconstrained",
            "wer_constrained",
            "cer_constrained",
        ):
            if key in row:
                bucket[key].append(float(row[key]))

    formatted_per_language: dict[str, dict[str, Any]] = {}
    for lang, payload in per_language.items():
        formatted_per_language[lang] = {
            "samples": payload["samples"],
            "unconstrained_decode_count": payload["unconstrained_decode_count"],
            "constrained_decode_count": payload["constrained_decode_count"],
            "wer_unconstrained_mean": safe_mean(payload["wer_unconstrained"]),
            "cer_unconstrained_mean": safe_mean(payload["cer_unconstrained"]),
            "wer_constrained_mean": safe_mean(payload["wer_constrained"]),
            "cer_constrained_mean": safe_mean(payload["cer_constrained"]),
        }

    content = {
        "samples_with_eval": len(eval_rows),
        "unconstrained_decode_count": unconstrained_decode_count,
        "constrained_decode_count": constrained_decode_count,
        "wer_unconstrained": metric_summary(eval_rows, "wer_unconstrained"),
        "cer_unconstrained": metric_summary(eval_rows, "cer_unconstrained"),
        "wer_constrained": metric_summary(eval_rows, "wer_constrained"),
        "cer_constrained": metric_summary(eval_rows, "cer_constrained"),
    }
    return content, formatted_per_language


def _build_markdown(scorecard: dict[str, Any]) -> str:
    metadata = scorecard.get("metadata", {})
    gate = scorecard.get("gate_summary", {})
    content = scorecard.get("content", {})
    naturalness = scorecard.get("naturalness", {})
    diagnostics = scorecard.get("diagnostics", {})
    lines = [
        f"# Scorecard: {metadata.get('experiment_id', '-')}",
        "",
        "## Metadata",
        f"- campaign_id: {metadata.get('campaign_id', '-')}",
        f"- codec: {metadata.get('codec', '-')}",
        f"- phase: {metadata.get('phase', '-')}",
        f"- stage: {metadata.get('stage', '-')}",
        f"- variant: {metadata.get('variant', '-')}",
        f"- track: {metadata.get('track', '-')}",
        f"- axis: {metadata.get('axis', '-')}",
        f"- family: {metadata.get('family', '-')}",
        f"- config_path: {metadata.get('config_path', '-')}",
        f"- seq_len: {metadata.get('seq_len', '-')}",
        f"- quantizers: {metadata.get('quantizers', '-')}",
        f"- validation_split: {metadata.get('validation_split', '-')}",
        f"- selected_step: {metadata.get('selected_step', '-')}",
        f"- verdict_suggestion: {scorecard.get('verdict_suggestion', '-')}",
        "",
        "## Content",
        f"- samples: {content.get('sample_count', content.get('samples_with_eval', 0))}",
        f"- generated_count: {content.get('generated_count', content.get('unconstrained_decode_count', 0))}",
        f"- wer_mean: {content.get('wer_mean', content.get('wer_unconstrained', {}))}",
        f"- cer_mean: {content.get('cer_mean', content.get('cer_unconstrained', {}))}",
        "",
        "## Naturalness",
        f"- utmos_mean: {naturalness.get('utmos_mean', '-')}",
        f"- dnsmos_p808_mean: {naturalness.get('dnsmos_p808_mean', '-')}",
        f"- dnsmos_ovr_mean: {naturalness.get('dnsmos_ovr_mean', '-')}",
        f"- speaker_similarity_mean: {naturalness.get('speaker_similarity_mean', '-')}",
        f"- salmon_mean: {naturalness.get('salmon_mean', '-')}",
        "",
        "## Diagnostics",
        f"- mel_l1_mean: {diagnostics.get('mel_l1_mean', '-')}",
        f"- mel_l2_mean: {diagnostics.get('mel_l2_mean', '-')}",
        f"- mel_cosine_mean: {diagnostics.get('mel_cosine_mean', '-')}",
        f"- frame_ratio_mean: {diagnostics.get('frame_ratio_mean', '-')}",
        f"- malformed_decode_rate: {diagnostics.get('malformed_decode_rate', '-')}",
        "",
        "## Languages",
    ]
    for lang, lang_payload in sorted(scorecard.get("per_language", {}).items()):
        lines.append(f"- {lang}: {lang_payload}")
    lines.extend(
        [
            "",
            "## Codec Health",
            f"- target_coverage_total_mean: {scorecard.get('codec_health', {}).get('target_coverage_total_mean', {})}",
            f"- generated_coverage_total_mean: {scorecard.get('codec_health', {}).get('generated_coverage_total_mean', {})}",
            f"- coverage_q_min_mean: {scorecard.get('codec_health', {}).get('coverage_q_min_mean', {})}",
            f"- coverage_q_abs_diff_max_mean: {scorecard.get('codec_health', {}).get('coverage_q_abs_diff_max_mean', {})}",
            "",
            "## Gate Summary",
            f"- gate_pass: {gate.get('gate_pass', '-')}",
            f"- steps_with_gate: {gate.get('steps_with_gate', 0)}",
            f"- first_pass_step: {gate.get('first_pass_step', '-')}",
            f"- last_gate: {gate.get('last_gate', {})}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable run scorecard from local artifacts.")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--dump-root", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None and args.experiment_id:
        run_dir = _resolve_run_dir(repo_root, args.experiment_id)
    dump_root = Path(args.dump_root).resolve() if args.dump_root else None
    if dump_root is None and run_dir is not None and (run_dir / "sample_artifacts").exists():
        dump_root = run_dir

    start = _load_json(run_dir / "manifest_start.json") if run_dir else {}
    end = _load_json(run_dir / "manifest_end.json") if run_dir else {}
    snapshot_manifest = _load_json(dump_root / "run_snapshot" / "manifest.json") if dump_root else {}
    resolved_config = _load_json(dump_root / "run_snapshot" / "resolved_config.json") if dump_root else {}
    dataset_manifest = _load_json(dump_root / "run_snapshot" / "dataset_manifest.json") if dump_root else {}
    full_eval_summary, full_eval_summary_path = _load_latest_full_eval_summary(run_dir)

    gate_rows = (
        _load_rows(sorted((dump_root / "sample_artifacts").glob("step_*/sample_0/gate_metrics.json")))
        if dump_root and (dump_root / "sample_artifacts").exists()
        else []
    )

    if full_eval_summary:
        content = {
            "sample_count": int(full_eval_summary.get("sample_count", 0) or 0),
            "generated_count": int(full_eval_summary.get("generated_count", 0) or 0),
            "wer_mean": full_eval_summary.get("wer_mean"),
            "cer_mean": full_eval_summary.get("cer_mean"),
        }
        per_language = full_eval_summary.get("per_language", {})
        codec_health = {
            "target_coverage_total_mean": full_eval_summary.get("target_coverage_total_mean"),
            "generated_coverage_total_mean": full_eval_summary.get("generated_coverage_total_mean"),
            "coverage_q_min_mean": full_eval_summary.get("coverage_q_min_mean"),
            "coverage_q_abs_diff_max_mean": full_eval_summary.get("coverage_q_abs_diff_max_mean"),
        }
        naturalness = {
            "utmos_mean": full_eval_summary.get("utmos_mean"),
            "dnsmos_p808_mean": full_eval_summary.get("dnsmos_p808_mean"),
            "dnsmos_ovr_mean": full_eval_summary.get("dnsmos_ovr_mean"),
            "speaker_similarity_mean": full_eval_summary.get("speaker_similarity_mean"),
            "salmon_mean": full_eval_summary.get("salmon_mean"),
        }
        diagnostics = {
            "mel_l1_mean": full_eval_summary.get("mel_l1_mean"),
            "mel_l2_mean": full_eval_summary.get("mel_l2_mean"),
            "mel_cosine_mean": full_eval_summary.get("mel_cosine_mean"),
            "frame_ratio_mean": full_eval_summary.get("frame_ratio_mean"),
            "malformed_decode_rate": full_eval_summary.get("malformed_decode_rate"),
            "metric_availability": full_eval_summary.get("metric_availability", {}),
        }
        malformed_rate = full_eval_summary.get("malformed_decode_rate")
        gate_summary = {
            "gate_pass": bool(
                content["sample_count"] > 0
                and content["generated_count"] == content["sample_count"]
                and float(malformed_rate if malformed_rate is not None else 1.0) == 0.0
            ),
            "steps_with_gate": len(gate_rows),
            "first_pass_step": None,
            "last_gate": gate_rows[-1] if gate_rows else {},
        }
        mean_wer = full_eval_summary.get("wer_mean")
        mean_cer = full_eval_summary.get("cer_mean")
        if content["sample_count"] == 0:
            verdict = "invalid"
        elif content["generated_count"] == 0:
            verdict = "negative"
        elif gate_summary["gate_pass"]:
            verdict = "positive"
        elif mean_wer is not None and mean_cer is not None:
            verdict = "inconclusive"
        else:
            verdict = "negative"
        selected_step = full_eval_summary.get("step")
    else:
        eval_rows = (
            _load_rows(sorted((dump_root / "sample_artifacts").glob("step_*/sample_0/sample_eval_metrics.json")))
            if dump_root and (dump_root / "sample_artifacts").exists()
            else []
        )
        content, per_language = _summarize_legacy_sample_eval(eval_rows)
        codec_health = {
            "target_coverage_total_mean": metric_summary(eval_rows, "target_coverage_total").get("mean"),
            "generated_coverage_total_mean": metric_summary(eval_rows, "unconstrained_coverage_total").get("mean"),
            "coverage_q_min_mean": None,
            "coverage_q_abs_diff_max_mean": None,
        }
        naturalness = {
            "utmos_mean": None,
            "dnsmos_p808_mean": None,
            "dnsmos_ovr_mean": None,
            "speaker_similarity_mean": None,
            "salmon_mean": None,
        }
        diagnostics = {
            "mel_l1_mean": None,
            "mel_l2_mean": None,
            "mel_cosine_mean": None,
            "frame_ratio_mean": None,
            "malformed_decode_rate": None,
            "metric_availability": {},
        }
        first_pass_step = None
        passed_steps = [int(row.get("step", 0)) for row in gate_rows if row.get("overall_pass")]
        if passed_steps:
            first_pass_step = min(passed_steps)
        mean_wer = safe_mean(
            [float(row["wer_unconstrained"]) for row in eval_rows if "wer_unconstrained" in row]
        )
        mean_cer = safe_mean(
            [float(row["cer_unconstrained"]) for row in eval_rows if "cer_unconstrained" in row]
        )
        gate_summary = {
            "steps_with_gate": len(gate_rows),
            "first_pass_step": first_pass_step,
            "last_gate": gate_rows[-1] if gate_rows else {},
            "gate_pass": bool(first_pass_step is not None),
        }
        if not eval_rows and not gate_rows:
            verdict = "invalid"
        elif content.get("unconstrained_decode_count", 0) == 0:
            verdict = "negative"
        elif first_pass_step is not None:
            verdict = "positive"
        elif mean_wer is not None and mean_cer is not None and mean_wer <= 0.35 and mean_cer <= 0.18:
            verdict = "positive"
        else:
            verdict = "inconclusive"
        selected_step = None

    metadata = {
        "experiment_id": start.get("experiment_id") or snapshot_manifest.get("experiment_id") or args.experiment_id,
        "campaign_id": start.get("campaign_id") or snapshot_manifest.get("campaign_id", ""),
        "codec": start.get("codec", ""),
        "phase": start.get("phase") or snapshot_manifest.get("phase", ""),
        "stage": start.get("stage") or snapshot_manifest.get("stage", ""),
        "variant": start.get("variant") or snapshot_manifest.get("variant", ""),
        "track": start.get("track") or snapshot_manifest.get("track", ""),
        "axis": start.get("axis") or snapshot_manifest.get("axis", ""),
        "family": start.get("family") or snapshot_manifest.get("family", ""),
        "question": start.get("question") or snapshot_manifest.get("question", ""),
        "hypothesis": start.get("hypothesis") or snapshot_manifest.get("hypothesis", ""),
        "brief_path": start.get("brief_path") or snapshot_manifest.get("brief_path", ""),
        "baseline_experiment_id": start.get("baseline_experiment_id") or snapshot_manifest.get("baseline_experiment_id", ""),
        "owner": start.get("owner") or snapshot_manifest.get("owner", ""),
        "config_path": start.get("config", ""),
        "seq_len": resolved_config.get("training", {}).get("seq_len", ""),
        "quantizers": resolved_config.get("model", {}).get("num_quantizers", ""),
        "validation_split": resolved_config.get("validation", {}).get("split", "validation"),
        "dataset_manifest_hash": dataset_manifest.get("fingerprint_sha256", ""),
        "run_dir": str(run_dir) if run_dir else "",
        "dump_root": str(dump_root) if dump_root else "",
        "selected_step": selected_step,
        "full_eval_summary_path": str(full_eval_summary_path) if full_eval_summary_path else "",
        "exit_code": end.get("exit_code", None),
    }

    scorecard = {
        "metadata": metadata,
        "verdict_suggestion": verdict,
        "content": content,
        "naturalness": naturalness,
        "diagnostics": diagnostics,
        "per_language": per_language,
        "codec_health": codec_health,
        "gate_summary": gate_summary,
    }

    output_root = run_dir or dump_root or repo_root / "research"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "scorecard.md").write_text(_build_markdown(scorecard), encoding="utf-8")
    print(json.dumps(scorecard, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
