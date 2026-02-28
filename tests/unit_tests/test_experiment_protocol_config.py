from torchtitan.config_manager import ConfigManager


def test_protocol_sections_parse_from_config():
    cfg = ConfigManager().parse_args(
        ["--job.config_file", "config/tinyaya_q1_fleurs_overfit_1sample_viz5.toml"]
    )
    assert cfg.experiment.id == "overfit_q1_viz5"
    assert cfg.experiment.phase == "overfit_q1"
    assert cfg.tts_eval.enabled is True
    assert cfg.tts_eval.asr_model_id
    assert cfg.overfit_gate.require_unconstrained_audio is True
    assert cfg.overfit_gate.min_consecutive_passes >= 1
    assert cfg.artifact.save_git_snapshot is True


def test_protocol_defaults_remain_backward_compatible():
    cfg = ConfigManager().parse_args([])
    assert cfg.experiment.id == ""
    assert cfg.tts_eval.enabled is False
    assert cfg.overfit_gate.require_unconstrained_audio is False
    assert cfg.artifact.dump_root == ""
