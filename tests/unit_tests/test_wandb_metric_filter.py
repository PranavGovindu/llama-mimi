from torchtitan.components.metrics import WandBLogger


def test_wandb_metric_filter_keeps_core_training_curves():
    assert WandBLogger._should_log_metric("core/train_loss") is True
    assert WandBLogger._should_log_metric("core/train_loss_ema") is True
    assert WandBLogger._should_log_metric("grad_norm") is True
    assert WandBLogger._should_log_metric("lr") is True
    assert WandBLogger._should_log_metric("samples/generated_audio_0") is True
    assert (
        WandBLogger._should_log_metric("samples/generated_audio_spectrogram_0")
        is True
    )


def test_wandb_metric_filter_drops_gate_and_codebook_noise():
    assert WandBLogger._should_log_metric("core/gate_overall_pass") is False
    assert WandBLogger._should_log_metric("gates/unconstrained_audio_seen") is False
    assert WandBLogger._should_log_metric("core/codec_backend") is False
    assert (
        WandBLogger._should_log_metric(
            "samples/generated_unconstrained_codebook_frames_0"
        )
        is False
    )
    assert WandBLogger._should_log_metric("time_metrics/end_to_end(s)") is False
