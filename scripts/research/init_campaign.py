#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _seed_directions() -> list[dict]:
    return [
        {
            "direction_id": "semantic_global_tokens",
            "family": "semantic_global_tokens",
            "status": "queued",
            "question": "How strong is a semantic-stream plus global-token baseline for EN/HI reference-conditioned expressive TTS?",
        },
        {
            "direction_id": "semantic_grouped_residual",
            "family": "semantic_grouped_residual",
            "status": "queued",
            "question": "Does grouped prediction over a low-rate semantic-first codec outperform flat token modeling for EN/HI expressive TTS?",
        },
        {
            "direction_id": "single_stream_large_codebook",
            "family": "single_stream_large_codebook",
            "status": "queued",
            "question": "Can a strong single-stream tokenizer stay competitive without grouped residual prediction?",
        },
        {
            "direction_id": "factorized_attribute_codec",
            "family": "factorized_attribute_codec",
            "status": "queued",
            "question": "Is a factorized content/prosody/timbre representation worth the added engineering cost for TinyAya-core TTS?",
        },
    ]


def _brief_text(campaign: dict) -> str:
    lines = [
        f"# Campaign Brief: {campaign['campaign_id']}",
        "",
        "## Goal",
        campaign["goal"],
        "",
        "## Product Target",
        f"- {campaign['product_target']}",
        "",
        "## Core Constraints",
        f"- TinyAya remains the core text-to-speech backbone: `{campaign['backbone_constraint']}`",
        f"- Primary languages: `{', '.join(campaign['languages'])}`",
        f"- Approval boundary: `{campaign['approval_boundary']}`",
        "",
        "## Research Rules",
        "- Treat the current repo as lab infrastructure and prior evidence, not as the thesis.",
        "- Research representation, architecture, loss, evaluation, conditioning, and observability from scratch.",
        "- Advance directions only on cumulative evidence, not on single-run excitement.",
        "",
        "## Initial Workstreams",
        "- Evaluation stack and bilingual normalization",
        "- Codec / tokenizer audit",
        "- TinyAya architecture-family audit",
        "- Loss / conditioning audit",
        "- Expressivity audit",
        "",
    ]
    return "\n".join(lines)


def _workstream_brief(title: str, question: str, axis: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- axis: `{axis}`",
            f"- question: {question}",
            "- success_criteria:",
            "  - clear comparison setup",
            "  - explicit kill criteria",
            "  - reusable scorecard evidence",
            "- notes:",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a tracked research campaign.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--owner", default="")
    parser.add_argument(
        "--goal",
        default="Build the best TinyAya-core EN/HI expressive reference-conditioned TTS system.",
    )
    parser.add_argument("--languages", nargs="+", default=["en", "hi"])
    parser.add_argument("--product-target", default="expressive_ref_tts")
    parser.add_argument("--approval-boundary", default="promoted_main_runs")
    parser.add_argument("--backbone-constraint", default="tinyaya_core")
    parser.add_argument("--status", default="active")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    research_root = repo_root / "research"
    campaigns_root = research_root / "campaigns"
    campaign_dir = campaigns_root / args.campaign_id
    if campaign_dir.exists():
        raise SystemExit(f"campaign already exists: {campaign_dir}")

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    campaign = {
        "campaign_id": args.campaign_id,
        "owner": args.owner,
        "goal": args.goal,
        "languages": [lang.strip().lower().replace("-", "_") for lang in args.languages],
        "product_target": args.product_target,
        "approval_boundary": args.approval_boundary,
        "backbone_constraint": args.backbone_constraint,
        "status": args.status,
        "current_champion": "",
        "current_direction": "",
        "created_at_utc": created_at,
    }

    for rel_dir in ("briefs", "reviews", "scorecards", "notes"):
        (campaign_dir / rel_dir).mkdir(parents=True, exist_ok=True)
    _write_json(campaign_dir / "campaign.json", campaign)
    (campaign_dir / "brief.md").write_text(_brief_text(campaign) + "\n", encoding="utf-8")
    _write_json(campaign_dir / "directions.json", {"directions": _seed_directions()})
    (campaign_dir / "briefs" / "00-eval-stack.md").write_text(
        _workstream_brief(
            "Evaluation Stack",
            "What bilingual EN/HI scorecard best captures content, language fidelity, speaker consistency, and expressiveness?",
            "evaluation",
        )
        + "\n",
        encoding="utf-8",
    )
    (campaign_dir / "briefs" / "01-codec-audit.md").write_text(
        _workstream_brief(
            "Codec Audit",
            "Which representation families deserve deeper TinyAya integration after small EN/HI encode-decode and pilot training audits?",
            "representation",
        )
        + "\n",
        encoding="utf-8",
    )
    (campaign_dir / "briefs" / "02-model-family-audit.md").write_text(
        _workstream_brief(
            "Model Family Audit",
            "What TinyAya-centered model family best matches the strongest candidate representations?",
            "architecture",
        )
        + "\n",
        encoding="utf-8",
    )
    (campaign_dir / "briefs" / "03-loss-audit.md").write_text(
        _workstream_brief(
            "Loss Audit",
            "Which objective mix improves expressive TTS quality without compromising EN/HI content accuracy?",
            "loss",
        )
        + "\n",
        encoding="utf-8",
    )
    (campaign_dir / "briefs" / "04-expressivity-audit.md").write_text(
        _workstream_brief(
            "Expressivity Audit",
            "What conditioning and evaluation setup best captures emotion and prosody gains for EN/HI reference-conditioned TTS?",
            "conditioning",
        )
        + "\n",
        encoding="utf-8",
    )

    _upsert_jsonl(
        campaigns_root / "index.jsonl",
        "campaign_id",
        {
            "campaign_id": args.campaign_id,
            "owner": args.owner,
            "goal": args.goal,
            "languages": campaign["languages"],
            "product_target": args.product_target,
            "approval_boundary": args.approval_boundary,
            "backbone_constraint": args.backbone_constraint,
            "status": args.status,
            "created_at_utc": created_at,
        },
    )
    print(json.dumps(campaign, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
