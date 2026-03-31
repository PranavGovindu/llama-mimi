#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _upsert_jsonl(path: Path, key: str, row: dict) -> None:
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [existing for existing in rows if existing.get(key) != row.get(key)]
    rows.append(row)
    rows.sort(key=lambda item: str(item.get(key, "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a structured review memo for one experiment.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--label",
        required=True,
        choices=["invalid", "negative", "inconclusive", "positive"],
    )
    parser.add_argument(
        "--decision",
        required=True,
        choices=["deepen", "pause", "retire", "reproduce", "rerun"],
    )
    parser.add_argument("--summary", default="")
    parser.add_argument("--next-action", default="")
    parser.add_argument("--owner", default="")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--scorecard", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scorecard_path = Path(args.scorecard).resolve() if args.scorecard else None
    if scorecard_path is None:
        scorecard_path = repo_root / "research" / "scorecard.json"
    scorecard = _load_json(scorecard_path)
    metadata = scorecard.get("metadata", {})
    campaign_id = args.campaign_id.strip() or metadata.get("campaign_id", "")
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    review = {
        "review_id": args.experiment_id,
        "experiment_id": args.experiment_id,
        "campaign_id": campaign_id,
        "label": args.label,
        "decision": args.decision,
        "summary": args.summary,
        "next_action": args.next_action,
        "owner": args.owner or metadata.get("owner", ""),
        "scorecard_path": str(scorecard_path),
        "created_at_utc": created_at,
    }

    lines = [
        f"# Review Memo: {args.experiment_id}",
        "",
        f"- label: `{args.label}`",
        f"- decision: `{args.decision}`",
        f"- campaign_id: `{campaign_id or '-'}`",
        f"- owner: `{review['owner'] or '-'}`",
        f"- created_at_utc: `{created_at}`",
        "",
        "## Summary",
        args.summary or "-",
        "",
        "## Evidence",
        f"- scorecard: `{scorecard_path}`",
        f"- verdict_suggestion: `{scorecard.get('verdict_suggestion', '-')}`",
        f"- content: `{scorecard.get('content', {})}`",
        f"- gate_summary: `{scorecard.get('gate_summary', {})}`",
        "",
        "## Next Action",
        args.next_action or "-",
        "",
    ]

    reviews_root = repo_root / "research" / "reviews"
    reviews_root.mkdir(parents=True, exist_ok=True)
    review_json_path = reviews_root / f"{args.experiment_id}.json"
    review_md_path = reviews_root / f"{args.experiment_id}.md"
    review_json_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_md_path.write_text("\n".join(lines), encoding="utf-8")
    _upsert_jsonl(reviews_root / "index.jsonl", "review_id", review)

    if campaign_id:
        campaign_reviews_root = repo_root / "research" / "campaigns" / campaign_id / "reviews"
        campaign_reviews_root.mkdir(parents=True, exist_ok=True)
        (campaign_reviews_root / f"{args.experiment_id}.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
