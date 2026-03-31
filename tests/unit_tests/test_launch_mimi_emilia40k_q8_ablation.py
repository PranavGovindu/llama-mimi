import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "exp"
        / "launch_mimi_emilia40k_q8_ablation.py"
    )
    spec = importlib.util.spec_from_file_location("launch_mimi_ablation", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_screen5k_plan_has_expected_variants():
    module = _load_module()
    rows = module._variant_rows(
        stage="screen5k",
        variant_seed_pairs=module._screen_pairs(),
        validation_split="validation",
        max_samples=1000,
        dataset_path="/vol/data/emilia_en40k_mimi_q8",
        config="codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en.toml",
        modal_path="mimi/ablation_emilia40k_q8_s4096_en",
        campaign_id="campaign",
        owner="codex",
        mode="modal",
    )

    assert len(rows) == 10
    anchor_rows = [row for row in rows if row["variant"] == "anchor"]
    assert len(anchor_rows) == 2
    by_variant = {row["variant"]: row for row in rows if row["seed"] == 0}
    assert by_variant["anchor"]["target_audio_tokens_per_update"] == 25600
    assert by_variant["gbs_8"]["target_audio_tokens_per_update"] == 12800
    assert by_variant["gbs_32"]["target_audio_tokens_per_update"] == 51200
    assert by_variant["maxsec_12"]["max_audio_seconds"] == 12
    assert by_variant["maxsec_20"]["max_audio_seconds"] == 20
    assert by_variant["anchor"]["steps"] == 5000
    assert by_variant["anchor"]["full_pack_eval_every"] == 5000
    assert by_variant["warmup_2pct"]["warmup_steps"] == 100
    assert by_variant["warmup_8pct"]["warmup_steps"] == 400
