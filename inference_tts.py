import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    StoppingCriteria,
)

from torchtitan.config_manager import JobConfig
from torchtitan.tools.audio_codec import load_audio_codec
from torchtitan.tools.audio_token_parser import (
    AllowTokenIdsLogitsProcessor,
    build_audio_code_id_map,
    build_spark_global_id_map,
    extract_audio_codes_bqt_from_token_ids,
    extract_spark_global_token_ids,
)


class StopOnAudioEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.target_ids = tokenizer("</audio>", add_special_tokens=False).input_ids

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) < len(self.target_ids):
            return False
        return input_ids[0][-len(self.target_ids) :].tolist() == self.target_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyAya codec-based text-to-speech.")
    parser.add_argument("--model-id", required=True, help="HF model id or local path.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--lang", default="en", help="Language code used as <lang_xx>.")
    parser.add_argument("--num-quantizers", type=int, default=0, help="0 = read from model config.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--restrict-audio-vocab",
        action="store_true",
        help="Constrain generation after <audio> to Mimi audio tokens plus </audio>.",
    )
    parser.add_argument("--output-file", default="output_tts.wav")
    parser.add_argument(
        "--audio-codec-backend",
        choices=["mimi", "s1_dac", "spark_bicodec"],
        default="mimi",
    )
    parser.add_argument(
        "--audio-codec-source",
        choices=["official_fish", "hf_pretrained"],
        default="official_fish",
    )
    parser.add_argument("--audio-codec-model-id", default="")
    parser.add_argument("--audio-codec-ckpt-path", default="")
    parser.add_argument("--audio-codec-trust-remote-code", action="store_true")
    parser.add_argument(
        "--spark-global-tokens",
        default="",
        help="Comma-separated Spark global token ids (e.g. 12,45,98).",
    )
    parser.add_argument(
        "--spark-global-tokens-file",
        default="",
        help="Optional text file with Spark global token ids separated by commas/spaces.",
    )
    parser.add_argument(
        "--spark-prompt-audio",
        default="",
        help="Optional prompt audio path to auto-extract Spark global tokens.",
    )
    return parser.parse_args()


def _parse_token_id_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    cleaned = raw.replace("\n", " ").replace("\t", " ").replace(",", " ")
    out: list[int] = []
    for piece in cleaned.split():
        try:
            out.append(int(piece))
        except ValueError:
            continue
    return out


def _resolve_spark_global_tokens(args: argparse.Namespace, audio_tokenizer) -> list[int]:
    direct = _parse_token_id_list(args.spark_global_tokens)
    if direct:
        return direct
    if args.spark_global_tokens_file.strip():
        raw = Path(args.spark_global_tokens_file).read_text(encoding="utf-8")
        from_file = _parse_token_id_list(raw)
        if from_file:
            return from_file
    if args.spark_prompt_audio.strip():
        wav, sr = torchaudio.load(args.spark_prompt_audio)
        if wav.ndim > 1:
            wav = torch.mean(wav, dim=0, keepdim=False)
        wav = wav.float().contiguous()
        if sr != int(audio_tokenizer.feature_extractor.sampling_rate):
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=int(sr),
                new_freq=int(audio_tokenizer.feature_extractor.sampling_rate),
            )
            sr = int(audio_tokenizer.feature_extractor.sampling_rate)
        inputs = audio_tokenizer.feature_extractor(
            raw_audio=wav.cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        ).to(audio_tokenizer.device)
        padding_mask = inputs.get("padding_mask")
        if padding_mask is None:
            padding_mask = inputs.get("attention_mask")
        if padding_mask is None:
            padding_mask = torch.ones_like(inputs["input_values"], dtype=torch.long)
        with torch.no_grad():
            enc = audio_tokenizer.encode(
                inputs["input_values"],
                padding_mask,
                num_quantizers=1,
            )
        raw_global = getattr(enc, "global_codes", None)
        if raw_global is not None:
            if not torch.is_tensor(raw_global):
                raw_global = torch.as_tensor(raw_global)
            if raw_global.ndim == 1:
                raw_global = raw_global.unsqueeze(0)
            if raw_global.ndim == 3:
                if raw_global.shape[1] == 1:
                    raw_global = raw_global[:, 0, :]
                elif raw_global.shape[2] == 1:
                    raw_global = raw_global[:, :, 0]
                else:
                    raw_global = raw_global.reshape(raw_global.shape[0], -1)
            if raw_global.ndim == 2 and raw_global.shape[0] >= 1:
                return raw_global[0].detach().cpu().to(torch.int64).tolist()
    return []


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    codec_cfg = JobConfig()
    codec_cfg.model.num_quantizers = max(int(args.num_quantizers), 1)
    codec_cfg.audio_codec.backend = args.audio_codec_backend
    codec_cfg.audio_codec.source = args.audio_codec_source
    if args.audio_codec_model_id.strip():
        codec_cfg.audio_codec.model_id = args.audio_codec_model_id.strip()
    if args.audio_codec_ckpt_path.strip():
        codec_cfg.audio_codec.codec_ckpt_path = args.audio_codec_ckpt_path.strip()
    codec_cfg.audio_codec.trust_remote_code = bool(args.audio_codec_trust_remote_code)
    audio_tokenizer, feature_extractor, codec_info = load_audio_codec(codec_cfg, device)
    stopping_criteria = StopOnAudioEnd(tokenizer)

    num_quantizers = args.num_quantizers or getattr(model.config, "num_quantizers", 1)
    if int(num_quantizers) > int(codec_info.max_codebooks):
        raise ValueError(
            f"num_quantizers={num_quantizers} exceeds codec max={codec_info.max_codebooks} "
            f"for backend={codec_info.backend}"
        )
    vocab = tokenizer.get_vocab()
    audio_start_id = vocab.get("<audio>")
    audio_end_id = vocab.get("</audio>")
    audio_code_id_map = build_audio_code_id_map(vocab)
    spark_global_id_map = build_spark_global_id_map(vocab)
    spark_global_start_id = vocab.get("<|start_global_token|>")
    spark_global_end_id = vocab.get("<|end_global_token|>")

    lang = args.lang.lower().replace("-", "_")
    backend = codec_info.backend.strip().lower()
    if backend == "spark_bicodec":
        spark_global_tokens = _resolve_spark_global_tokens(args, audio_tokenizer)
        if not spark_global_tokens:
            raise ValueError(
                "Spark inference requires global prompt tokens. Provide one of: "
                "--spark-global-tokens, --spark-global-tokens-file, --spark-prompt-audio"
            )
        global_text = "".join(
            f"<|bicodec_global_{int(tok)}|>" for tok in spark_global_tokens
        )
        prompt = (
            "<|task_tts|>"
            f"<|start_content|><lang_{lang}>{args.text}<|end_content|>"
            f"<|start_global_token|>{global_text}<|end_global_token|><audio>"
        )
        seed_global_codes = torch.tensor(
            spark_global_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
    else:
        prompt = f"<lang_{lang}>{args.text}<audio>"
        seed_global_codes = None

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = not args.greedy
    generate_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "stopping_criteria": [stopping_criteria],
        "eos_token_id": audio_end_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_k"] = args.top_k
    if args.restrict_audio_vocab:
        allowed_ids = sorted(audio_code_id_map.keys())
        if audio_end_id is not None:
            allowed_ids.append(audio_end_id)
        generate_kwargs["logits_processor"] = LogitsProcessorList(
            [AllowTokenIdsLogitsProcessor(allowed_ids)]
        )

    with torch.no_grad():
        generated = model.generate(**generate_kwargs)

    generated_ids = generated[0].detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    codes = extract_audio_codes_bqt_from_token_ids(
        token_ids=generated_ids,
        num_quantizers=num_quantizers,
        audio_code_id_map=audio_code_id_map,
        audio_start_id=audio_start_id,
        audio_end_id=audio_end_id,
    )
    if codes is None:
        raise ValueError("No valid audio token groups were generated.")
    codes = codes.to(device)
    decode_global_codes = None
    if backend == "spark_bicodec":
        generated_global_codes = extract_spark_global_token_ids(
            token_ids=generated_ids,
            spark_global_id_map=spark_global_id_map,
            start_global_id=spark_global_start_id,
            end_global_id=spark_global_end_id,
        )
        decode_global_codes = generated_global_codes
        if decode_global_codes is None:
            decode_global_codes = seed_global_codes
        if decode_global_codes is None:
            raise ValueError(
                "Spark decode requires global tokens; none found in prompt/generated ids."
            )
    audio_values = audio_tokenizer.decode(codes, global_codes=decode_global_codes)[0]
    torchaudio.save(
        args.output_file,
        audio_values[0].detach().cpu(),
        feature_extractor.sampling_rate,
    )
    print(
        "Codec: "
        f"backend={codec_info.backend} source={codec_info.source} "
        f"model_ref={codec_info.model_ref} sr={codec_info.sampling_rate}"
    )
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
