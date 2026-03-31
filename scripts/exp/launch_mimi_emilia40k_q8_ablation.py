#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en.toml"
DEFAULT_MODAL_PATH = "mimi/ablation_emilia40k_q8_s4096_en"
DEFAULT_CAMPAIGN_ID = "emilia40k-wave1"
DEFAULT_DATASET_PATH = "/vol/data/emilia_en40k_mimi_q8"
DEFAULT_DATASET_ID = "amphion/Emilia-Dataset"
DEFAULT_DATA_FILES = "Emilia/EN/*.tar"
DEFAULT_OWNER = "codex"


@dataclass(frozen=True)
class SweepVariant:
    name: str
    axis: str
    description: str
    overrides: dict[str, object]
    warmup_ratio: float = 0.04


SCREEN_VARIANTS: tuple[SweepVariant, ...] = (
    SweepVariant(
        name="anchor",
        axis="anchor",
        description="Baseline config for all English-only Q8/4096 comparisons.",
        overrides={
            "optimizer.lr": 5e-5,
            "training.global_batch_size": 4,
            "training.max_audio_seconds": 16,
        },
        warmup_ratio=0.04,
    ),
    SweepVariant(
        name="lr_2e-5",
        axis="lr",
        description="Lower learning-rate challenger.",
        overrides={"optimizer.lr": 2e-5},
    ),
    SweepVariant(
        name="lr_1e-4",
        axis="lr",
        description="Higher learning-rate challenger.",
        overrides={"optimizer.lr": 1e-4},
    ),
    SweepVariant(
        name="gbs_2",
        axis="global_batch_size",
        description="Smaller effective update size.",
        overrides={"training.global_batch_size": 2},
    ),
    SweepVariant(
        name="gbs_8",
        axis="global_batch_size",
        description="Larger effective update size.",
        overrides={"training.global_batch_size": 8},
    ),
    SweepVariant(
        name="maxsec_12",
        axis="max_audio_seconds",
        description="Shorter audio crop challenger.",
        overrides={"training.max_audio_seconds": 12},
    ),
    SweepVariant(
        name="maxsec_20",
        axis="max_audio_seconds",
        description="Longer audio crop challenger.",
        overrides={"training.max_audio_seconds": 20},
    ),
    SweepVariant(
        name="warmup_2pct",
        axis="warmup_ratio",
        description="Short warmup challenger.",
        overrides={},
        warmup_ratio=0.02,
    ),
    SweepVariant(
        name="warmup_8pct",
        axis="warmup_ratio",
        description="Long warmup challenger.",
        overrides={},
        warmup_ratio=0.08,
    ),
)

VARIANT_BY_NAME = {variant.name: variant for variant in SCREEN_VARIANTS}
ANCHOR = VARIANT_BY_NAME["anchor"]

STAGE_SPECS: dict[str, dict[str, int]] = {
    "screen5k": {
        "steps": 5000,
        "full_pack_eval_every": 5000,
        "checkpoint_interval": 5000,
    },
    "shortlist10k": {
        "steps": 10000,
        "full_pack_eval_every": 10000,
        "checkpoint_interval": 10000,
    },
    "confirm20k": {
        "steps": 20000,
        "full_pack_eval_every": 20000,
        "checkpoint_interval": 20000,
    },
    "main100k": {
        "steps": 100000,
        "full_pack_eval_every": 10000,
        "checkpoint_interval": 10000,
    },
}


def _parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_items(raw)]


def _audio_tokens_per_update(global_batch_size: int, max_audio_seconds: int, quantizers: int) -> int:
    return int(global_batch_size * max_audio_seconds * 12.5 * quantizers)


def _experiment_id(stage: str, axis: str, variant: str, seed: int) -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_stage = stage.replace("_", "-")
    safe_axis = axis.replace("_", "-")
    safe_variant = variant.replace("_", "-")
    return f"exp-{ts}-{safe_stage}-{safe_axis}-{safe_variant}-s{seed}"


def _stage_overrides(
    *,
    stage: str,
    dataset_path: str,
    validation_split: str,
    max_samples: int,
    warmup_ratio: float,
) -> dict[str, object]:
    spec = STAGE_SPECS[stage]
    steps = int(spec["steps"])
    warmup_steps = max(1, int(round(steps * warmup_ratio)))
    return {
        "training.steps": steps,
        "training.dataset_path": dataset_path,
        "training.dataset": "tts_pretok",
        "validation.dataset": "tts_pretok",
        "validation.split": validation_split,
        "validation.seq_len": 4096,
        "validation.local_batch_size": 2,
        "training.local_batch_size": 1,
        "training.seq_len": 4096,
        "training.language_tokens": False,
        "training.languages": [],
        "training.sample_generate_every": 500,
        "training.sample_generate_num_samples": 2,
        "training.sample_generate_do_sample": False,
        "training.sample_generate_restrict_audio_vocab": True,
        "training.log_target_media": True,
        "training.log_unconstrained_named_media": False,
        "model.num_quantizers": 8,
        "tts_eval.enabled": True,
        "tts_eval.eval_every": int(spec["full_pack_eval_every"]),
        "tts_eval.full_pack_enabled": True,
        "tts_eval.full_pack_eval_every": int(spec["full_pack_eval_every"]),
        "tts_eval.full_pack_max_samples": max_samples,
        "checkpoint.enable_checkpoint": True,
        "checkpoint.interval": int(spec["checkpoint_interval"]),
        "lr_scheduler.warmup_steps": warmup_steps,
    }


def _merge_overrides(*payloads: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for payload in payloads:
        merged.update(payload)
    return merged


def _variant_rows(
    *,
    stage: str,
    variant_seed_pairs: list[tuple[SweepVariant, int]],
    validation_split: str,
    max_samples: int,
    dataset_path: str,
    config: str,
    modal_path: str,
    campaign_id: str,
    owner: str,
    mode: str,
    baseline_experiment_id: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, seed in variant_seed_pairs:
        overrides = _merge_overrides(
            ANCHOR.overrides,
            variant.overrides,
            _stage_overrides(
                stage=stage,
                dataset_path=dataset_path,
                validation_split=validation_split,
                max_samples=max_samples,
                warmup_ratio=variant.warmup_ratio,
            ),
            {
                "training.seed": seed,
                "training.deterministic": False,
            },
        )
        experiment_id = _experiment_id(stage, variant.axis, variant.name, seed)
        global_batch = int(overrides["training.global_batch_size"])
        max_audio_seconds = int(overrides["training.max_audio_seconds"])
        target_audio_tokens = _audio_tokens_per_update(global_batch, max_audio_seconds, 8)
        question = "Which English-only Q8/4096 settings improve held-out TTS quality on the frozen Emilia-40k subset?"
        cmd = [
            sys.executable,
            "scripts/exp/launch.py",
            "--mode",
            mode,
            "--config",
            config,
            "--modal-path",
            modal_path,
            "--experiment-id",
            experiment_id,
            "--campaign-id",
            campaign_id,
            "--phase",
            "mimi_q8_s4096_emilia40k_en_ablation",
            "--stage",
            stage,
            "--variant",
            variant.name,
            "--track",
            "english_only",
            "--axis",
            variant.axis,
            "--family",
            "mimi_q8_s4096",
            "--question",
            question,
            "--hypothesis",
            variant.description,
            "--owner",
            owner,
            "--codec",
            "mimi",
            "--tags",
            ",".join(
                [
                    "tinyaya",
                    "tts",
                    "mimi",
                    "q8",
                    "4096",
                    "emilia40k",
                    "english",
                    stage,
                    variant.axis,
                    variant.name,
                    f"seed{seed}",
                ]
            ),
        ]
        if baseline_experiment_id.strip():
            cmd.extend(["--baseline-experiment-id", baseline_experiment_id.strip()])
        for key, value in overrides.items():
            cmd.extend(["--override", f"{key}={json.dumps(value)}"])
        rows.append(
            {
                "stage": stage,
                "variant": variant.name,
                "axis": variant.axis,
                "seed": seed,
                "experiment_id": experiment_id,
                "steps": int(overrides["training.steps"]),
                "lr": overrides["optimizer.lr"],
                "global_batch_size": global_batch,
                "warmup_steps": int(overrides["lr_scheduler.warmup_steps"]),
                "warmup_ratio": variant.warmup_ratio,
                "max_audio_seconds": max_audio_seconds,
                "target_audio_tokens_per_update": target_audio_tokens,
                "validation_split": validation_split,
                "full_pack_max_samples": max_samples,
                "full_pack_eval_every": int(overrides["tts_eval.full_pack_eval_every"]),
                "checkpoint_interval": int(overrides["checkpoint.interval"]),
                "config": config,
                "modal_path": modal_path,
                "description": variant.description,
                "command": cmd,
                "command_str": " ".join(shlex.quote(part) for part in cmd),
            }
        )
    return rows


def _write_plan_files(stage: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    sweep_root = REPO_ROOT / "research" / "sweeps" / "mimi_q8_s4096_emilia40k_en"
    sweep_root.mkdir(parents=True, exist_ok=True)
    tsv_path = sweep_root / f"{stage}_plan.tsv"
    json_path = sweep_root / f"{stage}_plan.json"
    fieldnames = [
        "stage",
        "variant",
        "axis",
        "seed",
        "experiment_id",
        "steps",
        "lr",
        "global_batch_size",
        "warmup_steps",
        "warmup_ratio",
        "max_audio_seconds",
        "target_audio_tokens_per_update",
        "validation_split",
        "full_pack_max_samples",
        "full_pack_eval_every",
        "checkpoint_interval",
        "config",
        "modal_path",
        "description",
        "command_str",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return tsv_path, json_path


def _run_commands(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        subprocess.run(row["command"], cwd=str(REPO_ROOT), check=True)


def _pretokenize_command(args: argparse.Namespace) -> list[str]:
    return [
        "modal",
        "run",
        "modal/app.py",
        "--mode",
        "pretokenize_emilia",
        "--dataset-id",
        args.dataset_id,
        "--data-files",
        args.data_files,
        "--source-split",
        args.source_split,
        "--quantizers",
        "8",
        "--output-dir",
        args.dataset_path,
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-validation-samples",
        str(args.max_validation_samples),
        "--max-test-samples",
        str(args.max_test_samples),
        "--min-seconds",
        str(args.min_seconds),
        "--max-seconds",
        str(args.max_seconds),
        "--seed",
        str(args.dataset_seed),
        "--shard-size",
        str(args.shard_size),
        "--lang",
        "en",
        "--audio-codec-backend",
        "mimi",
        "--audio-codec-source",
        "hf_pretrained",
        "--audio-codec-model-id",
        "kyutai/mimi",
    ]


def _screen_pairs() -> list[tuple[SweepVariant, int]]:
    pairs: list[tuple[SweepVariant, int]] = [(ANCHOR, 0), (ANCHOR, 1)]
    for variant in SCREEN_VARIANTS:
        if variant.name == "anchor":
            continue
        pairs.append((variant, 0))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or launch the staged English-only Mimi Q8/4096 Emilia-40k ablation campaign."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("pretokenize", help="Plan or run the fixed Emilia-English subset build.")
    prep.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    prep.add_argument("--data-files", default=DEFAULT_DATA_FILES)
    prep.add_argument("--source-split", default="train")
    prep.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    prep.add_argument("--max-train-samples", type=int, default=38000)
    prep.add_argument("--max-validation-samples", type=int, default=1000)
    prep.add_argument("--max-test-samples", type=int, default=1000)
    prep.add_argument("--min-seconds", type=float, default=1.0)
    prep.add_argument("--max-seconds", type=float, default=20.0)
    prep.add_argument("--dataset-seed", type=int, default=42)
    prep.add_argument("--shard-size", type=int, default=500)
    prep.add_argument("--execute", action="store_true")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=["local", "modal"], default="modal")
    common.add_argument("--config", default=DEFAULT_CONFIG)
    common.add_argument("--modal-path", default=DEFAULT_MODAL_PATH)
    common.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    common.add_argument("--owner", default=DEFAULT_OWNER)
    common.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    common.add_argument("--validation-split", default="validation")
    common.add_argument("--full-pack-max-samples", type=int, default=1000)
    common.add_argument("--baseline-experiment-id", default="")
    common.add_argument("--execute", action="store_true")

    subparsers.add_parser(
        "screen5k",
        parents=[common],
        help="Plan or launch the 10-run 5k-step screen.",
    )

    shortlist = subparsers.add_parser(
        "shortlist10k",
        parents=[common],
        help="Plan or launch the 10k-step shortlist rung.",
    )
    shortlist.add_argument("--variants", required=True, help="Comma-separated shortlist variants.")
    shortlist.add_argument("--seeds", default="0,1")

    confirm = subparsers.add_parser(
        "confirm20k",
        parents=[common],
        help="Plan or launch the 20k-step confirmation rung.",
    )
    confirm.add_argument("--variants", required=True, help="Comma-separated confirmation variants.")
    confirm.add_argument("--seeds", default="0,1")

    main = subparsers.add_parser(
        "main100k",
        parents=[common],
        help="Plan or launch the 100k-step main run for the selected champion.",
    )
    main.add_argument("--variant", required=True, help="Winning variant to launch for the main run.")
    main.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "pretokenize":
        cmd = _pretokenize_command(args)
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        return

    if args.command == "screen5k":
        rows = _variant_rows(
            stage="screen5k",
            variant_seed_pairs=_screen_pairs(),
            validation_split=args.validation_split,
            max_samples=int(args.full_pack_max_samples),
            dataset_path=args.dataset_path,
            config=args.config,
            modal_path=args.modal_path,
            campaign_id=args.campaign_id,
            owner=args.owner,
            mode=args.mode,
            baseline_experiment_id=args.baseline_experiment_id,
        )
    elif args.command in {"shortlist10k", "confirm20k"}:
        stage = args.command
        seeds = _parse_int_list(args.seeds)
        variant_names = _parse_csv_items(args.variants)
        missing = [name for name in variant_names if name not in VARIANT_BY_NAME]
        if missing:
            raise SystemExit(
                f"Unknown variants: {', '.join(missing)}. Known variants: {', '.join(sorted(VARIANT_BY_NAME))}"
            )
        pairs = [(VARIANT_BY_NAME[name], seed) for name in variant_names for seed in seeds]
        rows = _variant_rows(
            stage=stage,
            variant_seed_pairs=pairs,
            validation_split=args.validation_split,
            max_samples=int(args.full_pack_max_samples),
            dataset_path=args.dataset_path,
            config=args.config,
            modal_path=args.modal_path,
            campaign_id=args.campaign_id,
            owner=args.owner,
            mode=args.mode,
            baseline_experiment_id=args.baseline_experiment_id,
        )
    else:
        variant_name = args.variant.strip()
        if variant_name not in VARIANT_BY_NAME:
            raise SystemExit(
                f"Unknown variant: {variant_name}. Known variants: {', '.join(sorted(VARIANT_BY_NAME))}"
            )
        rows = _variant_rows(
            stage="main100k",
            variant_seed_pairs=[(VARIANT_BY_NAME[variant_name], int(args.seed))],
            validation_split=args.validation_split,
            max_samples=int(args.full_pack_max_samples),
            dataset_path=args.dataset_path,
            config=args.config,
            modal_path=args.modal_path,
            campaign_id=args.campaign_id,
            owner=args.owner,
            mode=args.mode,
            baseline_experiment_id=args.baseline_experiment_id,
        )

    tsv_path, json_path = _write_plan_files(args.command, rows)
    print(f"wrote plan TSV: {tsv_path}")
    print(f"wrote plan JSON: {json_path}")
    if args.execute:
        _run_commands(rows)
    else:
        for row in rows:
            print(row["command_str"])


if __name__ == "__main__":
    main()
