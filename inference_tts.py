import argparse

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
    extract_audio_codes_bqt_from_token_ids,
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
        choices=["mimi", "s1_dac"],
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
    return parser.parse_args()


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

    lang = args.lang.lower().replace("-", "_")
    prompt = f"<lang_{lang}>{args.text}<audio>"

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
    audio_values = audio_tokenizer.decode(codes)[0]
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
