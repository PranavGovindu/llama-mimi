#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gate/sample metrics from local overfit artifacts.")
    parser.add_argument("--dump-root", required=True, help="Run output root that contains sample_artifacts")
    args = parser.parse_args()

    sample_root = Path(args.dump_root) / "sample_artifacts"
    if not sample_root.exists():
        raise SystemExit(f"missing sample_artifacts under {args.dump_root}")

    gate_rows: list[dict] = []
    eval_rows: list[dict] = []
    for gate_path in sorted(sample_root.glob("step_*/sample_0/gate_metrics.json")):
        row = _load_json(gate_path)
        if row:
            gate_rows.append(row)
    for eval_path in sorted(sample_root.glob("step_*/sample_0/sample_eval_metrics.json")):
        row = _load_json(eval_path)
        if row:
            eval_rows.append(row)

    result = {
        "steps_with_gate": len(gate_rows),
        "steps_with_eval": len(eval_rows),
        "last_gate": gate_rows[-1] if gate_rows else {},
        "best_gate_step": None,
        "best_eval_step": None,
    }

    passed = [r for r in gate_rows if r.get("overall_pass")]
    if passed:
        result["best_gate_step"] = min(int(r.get("step", 0)) for r in passed)

    scored = []
    for row in eval_rows:
        score = 0.0
        if "wer_unconstrained" in row:
            score += float(row["wer_unconstrained"])
        if "cer_unconstrained" in row:
            score += float(row["cer_unconstrained"])
        scored.append((score, int(row.get("step", 0))))
    if scored:
        result["best_eval_step"] = min(scored)[1]

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
