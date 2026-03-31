import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from fnmatch import fnmatch
from pathlib import Path
from textwrap import dedent

import modal
import pyarrow as pa
import pyarrow.parquet as pq


APP_NAME = "tinyaya-mimi-tts"
DATA_VOL_NAME = "tinyaya-mimi-tts-data"
REPO_ROOT = Path(__file__).resolve().parents[1]
FISH_SPEECH_REPO_ROOT = REPO_ROOT.parent / "fish-speech"
SPARK_TTS_REPO_ROOT = REPO_ROOT.parent / "Spark-TTS"
QWEN3_TTS_REPO_ROOT = REPO_ROOT.parent / "Qwen3-TTS"
SGLANG_OMNI_REPO_ROOT = Path("/tmp/sglang-omni")
REMOTE_REPO_ROOT = "/root/repo"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
HF_SECRETS = [
    modal.Secret.from_name("datasynthgen-secrets"),
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("huggingface-secret-nullhawk"),
    modal.Secret.from_name("hf"),
    modal.Secret.from_name("hf-token"),
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libsndfile1", "sox")
    .pip_install(
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "torchcodec==0.9.1",
        "torchvision==0.24.1",
        "einops==0.8.1",
        "einx==0.3.0",
        "torchdata",
        "datasets",
        "blobfile",
        "tiktoken",
        "tabulate",
        "tyro",
        "soundfile",
        "librosa",
        "onnxruntime",
        "transformers",
        "torchmetrics",
        "huggingface_hub",
        "accelerate",
        "hydra-core",
        "omegaconf",
        "pyrootutils",
        "loguru",
        "descript-audio-codec",
        "descript-audiotools",
        "moshi",
        "dualcodec",
        "soxr==0.5.0.post1",
        "seaborn",
        "sox",
        "speechbrain",
        "wandb",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_REPO_ROOT,
        ignore=[".venv", ".git", "__pycache__", "outputs", "assets"],
    )
)
if FISH_SPEECH_REPO_ROOT.exists():
    image = image.add_local_dir(
        str(FISH_SPEECH_REPO_ROOT),
        remote_path="/root/fish-speech",
        ignore=[".git", ".venv", "__pycache__", "outputs", "checkpoints", "logs"],
    )
if SPARK_TTS_REPO_ROOT.exists():
    image = image.add_local_dir(
        str(SPARK_TTS_REPO_ROOT),
        remote_path="/root/spark-tts",
        ignore=[".git", ".venv", "__pycache__", "outputs", "checkpoints", "logs"],
    )
if QWEN3_TTS_REPO_ROOT.exists():
    image = image.add_local_dir(
        str(QWEN3_TTS_REPO_ROOT),
        remote_path="/root/qwen3-tts",
        ignore=[".git", ".venv", "__pycache__", "outputs", "checkpoints", "logs"],
    )

s2_pro_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libsndfile1", "sox")
    .pip_install(
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "torchcodec==0.8.0",
        "torchvision==0.24.1",
        "accelerate>=0.27.0",
        "transformers<5.0",
        "safetensors>=0.4.3",
        "pillow>=10.0.0",
        "huggingface_hub>=0.36.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.23.0",
        "sglang==0.5.8",
        "pyzmq>=25.0.0",
        "msgpack>=1.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "httpx",
        "xxhash>=3.0.0",
        "av>=16.1.0",
        "qwen-vl-utils==0.0.11",
        "numba==0.63.1",
        "librosa>=0.11.0",
        "flatten-dict",
        "argbind>=0.3.7",
        "julius",
        "markdown2",
        "randomname",
        "pyloudnorm",
        "pystoi",
        "tensorboard",
        "torch-stoi",
        "ffmpy",
        "pandas",
        "tabulate",
        "typer>=0.9.0",
        "openai==2.6.1",
        "openai-harmony==0.0.4",
        "soundfile>=0.12.0",
        "tiktoken",
        "hydra-core",
        "omegaconf",
        "pyrootutils",
        "loguru",
    )
    .run_commands(
        "python -m pip install --no-deps descript-audiotools==0.7.2 descript-audio-codec==1.0.0",
        "rm -rf /root/sglang-omni && git clone --depth 1 https://github.com/sgl-project/sglang-omni /root/sglang-omni",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_REPO_ROOT,
        ignore=[".venv", ".git", "__pycache__", "outputs", "assets"],
    )
)


def _safe_slug(value: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    if not slug:
        return "run"
    return slug[:max_len]


def _resolve_sglang_omni_repo() -> Path | None:
    for candidate in (
        Path("/root/sglang-omni"),
        Path("/tmp/sglang-omni"),
        Path("/root/repo/sglang-omni"),
    ):
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _repo_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "").strip()
    if py_path:
        env["PYTHONPATH"] = f"{REMOTE_REPO_ROOT}:{py_path}"
    else:
        env["PYTHONPATH"] = REMOTE_REPO_ROOT
    try:
        has_qwen_repo = Path("/root/qwen3-tts").exists()
    except OSError:
        has_qwen_repo = False
    sglang_omni_repo = _resolve_sglang_omni_repo()
    if sglang_omni_repo is not None:
        env["PYTHONPATH"] = f"{sglang_omni_repo}:{env['PYTHONPATH']}"
    if has_qwen_repo:
        env.setdefault("QWEN3_TTS_REPO", "/root/qwen3-tts")
    hf_token = _resolve_hf_token()
    if hf_token:
        env.setdefault("HF_TOKEN", hf_token)
        env.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    if extra:
        env.update(extra)
    return env


def _load_run_name_defaults(config_file: str) -> dict[str, object]:
    cfg_path = Path(REMOTE_REPO_ROOT) / config_file
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / config_file
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    model_id = str(raw.get("model", {}).get("name", "model"))
    model_name = model_id.split("/")[-1]
    dataset_name = str(raw.get("training", {}).get("dataset", "dataset"))
    seq_len = int(raw.get("training", {}).get("seq_len", 2048))
    pretrained = bool(raw.get("model", {}).get("pretrained", True))
    audio_codec_cfg = raw.get("audio_codec", {})
    codebook_size = int(audio_codec_cfg.get("codebook_size_override", 0) or 0)
    if codebook_size <= 0:
        codebook_size = 2048
    checkpoint_folder = str(raw.get("checkpoint", {}).get("folder", "checkpoint"))
    return {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "pretrained": pretrained,
        "codebook_size": codebook_size,
        "checkpoint_folder": checkpoint_folder,
    }


def _resolve_run_name(
    model_name: str,
    dataset_name: str,
    num_quantizers: int,
    seq_len: int,
    pretrained: bool,
    experiment_id: str,
) -> str:
    run_name = (
        f"{model_name}_{dataset_name}"
        f"-q{num_quantizers}"
        f"-s{seq_len}"
        f"{'-random' if not pretrained else ''}"
    )
    if experiment_id:
        run_name = f"{run_name}-{experiment_id}"
    return run_name


def _format_override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extend_cmd_with_overrides(cmd: list[str], overrides: dict[str, object]) -> None:
    for key, value in overrides.items():
        if value is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        cmd.extend([f"--{key_str}", _format_override_value(value)])


def _parse_overrides_json(overrides_json: str) -> dict[str, object]:
    raw = overrides_json.strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("overrides_json must decode to a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _ensure_model_in_hf_collection(
    hf_repo_id: str,
    hf_repo_private: bool,
    hf_collection_slug: str,
    hf_token: str | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "repo_id": hf_repo_id,
        "collection_slug": hf_collection_slug,
        "repo_ensured": False,
        "collection_item_added": False,
        "error": "",
    }
    if not hf_repo_id.strip():
        return result
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        result["error"] = f"huggingface_hub import failed: {exc}"
        return result

    api = HfApi(token=hf_token or None)
    try:
        api.create_repo(
            repo_id=hf_repo_id.strip(),
            repo_type="model",
            private=hf_repo_private,
            exist_ok=True,
        )
        result["repo_ensured"] = True
    except Exception as exc:
        result["error"] = f"create_repo failed: {exc}"
        return result

    if not hf_collection_slug.strip():
        return result

    add_fn = getattr(api, "add_collection_item", None)
    if add_fn is None:
        result["error"] = "huggingface_hub lacks add_collection_item API"
        return result

    try:
        try:
            add_fn(
                collection_slug=hf_collection_slug.strip(),
                item_id=hf_repo_id.strip(),
                item_type="model",
                exists_ok=True,
            )
        except TypeError:
            try:
                add_fn(
                    collection_slug=hf_collection_slug.strip(),
                    item_id=hf_repo_id.strip(),
                    item_type="model",
                )
            except TypeError:
                add_fn(hf_collection_slug.strip(), hf_repo_id.strip(), "model")
        result["collection_item_added"] = True
    except Exception as exc:
        result["error"] = f"add_collection_item failed: {exc}"
    return result


def _resolve_hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("TOKEN")
        or ""
    ).strip()


def _resolve_hf_repo_id(api, repo_name_or_id: str, namespace: str = "") -> str:
    repo = repo_name_or_id.strip()
    if not repo:
        raise ValueError("repo_name_or_id must be non-empty")
    if "/" in repo:
        return repo
    ns = namespace.strip()
    if not ns:
        whoami = api.whoami()
        if isinstance(whoami, dict):
            ns = str(whoami.get("name", "") or "").strip()
        else:
            ns = str(getattr(whoami, "name", "") or "").strip()
    if not ns:
        raise RuntimeError("Unable to infer Hugging Face namespace from token.")
    return f"{ns}/{repo}"


def _write_dataset_card(
    dataset_dir: Path,
    *,
    repo_id: str,
    manifest: dict[str, object],
) -> Path:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    selection = (
        manifest.get("selection", {})
        if isinstance(manifest.get("selection"), dict)
        else {}
    )
    audio_codec = (
        manifest.get("audio_codec", {})
        if isinstance(manifest.get("audio_codec"), dict)
        else {}
    )
    total_samples = sum(
        int(counts.get(split, 0) or 0) for split in ("train", "validation", "test")
    )
    if total_samples < 10_000:
        size_category = "n<10K"
    elif total_samples < 100_000:
        size_category = "10K<n<100K"
    elif total_samples < 1_000_000:
        size_category = "100K<n<1M"
    else:
        size_category = "n>1M"
    readme = dedent(
        f"""\
        ---
        pretty_name: {repo_id.split('/')[-1]}
        language:
        - en
        task_categories:
        - text-to-speech
        size_categories:
        - {size_category}
        ---

        # {repo_id.split('/')[-1]}

        Frozen pretokenized Emilia-English model-ready dataset for TinyAya + Mimi training.

        ## Layout

        - `train/lang=en/*.parquet`
        - optional `validation/lang=en/*.parquet`
        - optional `test/lang=en/*.parquet`
        - `dataset_manifest.json`

        ## Selection

        - source dataset: `{manifest.get('dataset_name', '')}`
        - data files: `{manifest.get('data_files', '')}`
        - source split: `{manifest.get('source_split', '')}`
        - quantizers: `{manifest.get('num_quantizers', '')}`
        - train samples: `{counts.get('train', 0)}`
        - validation samples: `{counts.get('validation', 0)}`
        - test samples: `{counts.get('test', 0)}`
        - min seconds: `{selection.get('min_seconds', '')}`
        - max seconds: `{selection.get('max_seconds', '')}`

        ## Audio Codec

        - backend: `{audio_codec.get('backend', '')}`
        - source: `{audio_codec.get('source', '')}`
        - model: `{audio_codec.get('model_ref', '')}`
        - sample rate: `{audio_codec.get('sample_rate', '')}`

        ## Notes

        This repo stores pretokenized training artifacts, not raw audio. Use `dataset_manifest.json`
        as the immutable split fingerprint for ablation reproducibility.
        """
    )
    readme_path = dataset_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def _write_dataset_manifest(output_root: Path, payload: dict[str, object]) -> Path:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    finalized = dict(payload)
    finalized["fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(finalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _assign_reference_pairs_impl(dataset_dir: str) -> dict[str, object]:
    dataset_root = Path(dataset_dir).resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("artifact_kind", "")).strip().lower() != "codec_pretok_tts":
        return {
            "status": "skipped",
            "reason": "artifact_not_codec_pretok_tts",
            "dataset_dir": str(dataset_root),
        }

    split_stats: dict[str, dict[str, int]] = {}
    reference_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    for split_name in ("train", "validation", "test"):
        split_dir = dataset_root / split_name / "lang=en"
        parquet_files = sorted(split_dir.glob("*.parquet")) if split_dir.exists() else []
        if not parquet_files:
            split_stats[split_name] = {
                "rows": 0,
                "speakers": 0,
                "rows_with_reference": 0,
            }
            continue

        speaker_counts: dict[str, int] = {}
        speaker_anchors: dict[str, list[dict[str, object]]] = {}
        total_rows = 0

        # Pass 1: collect up to two anchor utterances per speaker.
        for parquet_path in parquet_files:
            table = pq.ParquetFile(parquet_path).read(
                columns=["sample_id", "speaker_id", "audio_codes"]
            )
            sample_ids = table["sample_id"].to_pylist()
            speaker_ids = table["speaker_id"].to_pylist()
            audio_codes = table["audio_codes"].to_pylist()
            for sample_id, speaker_id, codes in zip(sample_ids, speaker_ids, audio_codes):
                total_rows += 1
                if speaker_id is None:
                    continue
                speaker_key = str(speaker_id).strip()
                if not speaker_key:
                    continue
                speaker_counts[speaker_key] = speaker_counts.get(speaker_key, 0) + 1
                anchors = speaker_anchors.setdefault(speaker_key, [])
                if len(anchors) < 2:
                    anchors.append(
                        {
                            "sample_id": str(sample_id),
                            "audio_codes": codes,
                        }
                    )

        # Pass 2: rewrite rows with deterministic same-speaker anchors.
        rows_with_reference = 0
        for parquet_path in parquet_files:
            table = pq.ParquetFile(parquet_path).read()
            rows = table.to_pylist()
            rewritten: list[dict[str, object]] = []
            for row in rows:
                speaker_id = row.get("speaker_id")
                if speaker_id is None:
                    row["ref_sample_id"] = None
                    row["ref_audio_codes"] = None
                    rewritten.append(row)
                    continue
                speaker_key = str(speaker_id).strip()
                anchors = speaker_anchors.get(speaker_key, [])
                if len(anchors) < 2:
                    row["ref_sample_id"] = None
                    row["ref_audio_codes"] = None
                    rewritten.append(row)
                    continue

                sample_id = str(row.get("sample_id", ""))
                primary = anchors[0]
                alternate = anchors[1]
                chosen = primary if sample_id != str(primary["sample_id"]) else alternate
                row["ref_sample_id"] = chosen["sample_id"]
                row["ref_audio_codes"] = chosen["audio_codes"]
                rows_with_reference += 1
                rewritten.append(row)

            rewritten_table = pa.Table.from_pylist(rewritten)
            tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
            pq.write_table(rewritten_table, tmp_path)
            tmp_path.replace(parquet_path)

        reference_counts[split_name] = rows_with_reference
        split_stats[split_name] = {
            "rows": total_rows,
            "speakers": len(speaker_counts),
            "rows_with_reference": rows_with_reference,
        }

    reference_cfg = dict(manifest.get("reference_conditioning", {}) or {})
    reference_cfg["enabled"] = True
    reference_cfg["pairing_rule"] = "deterministic_speaker_anchor_across_merged_split"
    reference_cfg["counts_with_reference"] = reference_counts
    manifest["reference_conditioning"] = reference_cfg
    manifest_path = _write_dataset_manifest(dataset_root, manifest)

    return {
        "status": "ok",
        "dataset_dir": str(dataset_root),
        "manifest_path": str(manifest_path),
        "split_stats": split_stats,
    }


def _canonical_data_file_patterns(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or [raw.strip()]


def _list_matching_dataset_files(dataset_id: str, data_files: str) -> list[str]:
    from huggingface_hub import HfApi

    token = _resolve_hf_token() or None
    api = HfApi(token=token)
    all_files = api.list_repo_files(dataset_id.strip(), repo_type="dataset")
    patterns = _canonical_data_file_patterns(data_files)
    matched = sorted(
        file_path
        for file_path in all_files
        if any(fnmatch(file_path, pattern) for pattern in patterns)
    )
    if not matched:
        raise RuntimeError(
            f"No dataset files in {dataset_id!r} matched patterns {patterns!r}."
        )
    return matched


def _partition_dataset_files(files: list[str], num_workers: int) -> list[list[str]]:
    worker_count = max(1, min(int(num_workers), len(files)))
    groups: list[list[str]] = [[] for _ in range(worker_count)]
    for index, file_path in enumerate(files):
        groups[index % worker_count].append(file_path)
    return [group for group in groups if group]


def _worker_root_for_output_dir(output_dir: str) -> str:
    target = Path(output_dir).expanduser()
    return str(target.parent / f"{target.name}__workers")


def _controller_state_path_for_output_dir(output_dir: str) -> Path:
    worker_root = Path(_worker_root_for_output_dir(output_dir)).expanduser()
    return worker_root / "_controller_state.json"


def _partition_dataset_files_into_chunks(
    files: list[str],
    files_per_chunk: int,
) -> list[list[str]]:
    if not files:
        return []
    chunk_size = max(1, int(files_per_chunk))
    return [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]


def _chunk_output_dir_for_index(output_dir: str, chunk_idx: int) -> str:
    worker_root = Path(_worker_root_for_output_dir(output_dir)).expanduser()
    return str(worker_root / f"chunk-{int(chunk_idx):05d}")


def _chunk_manifest_path(chunk_output_dir: str) -> Path:
    return Path(chunk_output_dir).expanduser().resolve() / "dataset_manifest.json"


def _load_json_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _chunk_state_status_for_manifest(chunk_output_dir: str) -> str:
    return "completed" if _chunk_manifest_path(chunk_output_dir).exists() else "pending"


def _state_preserves_existing_chunk_layout(existing: dict[str, object]) -> bool:
    if bool(existing.get("migrated_from_legacy_workers")):
        return True
    chunks = existing.get("chunks", []) or []
    for chunk in chunks:
        output_dir = str(chunk.get("output_dir", "") or "")
        name = Path(output_dir).name
        if name.startswith("worker-") or name.startswith("chunk-legacy-"):
            return True
    return False


def _load_or_create_parallel_emilia_state(
    *,
    output_dir: str,
    dataset_id: str,
    data_files: str,
    source_split: str,
    matched_files: list[str],
    files_per_chunk: int,
    parallel_workers: int,
    repo_name_or_id: str,
) -> dict[str, object]:
    state_path = _controller_state_path_for_output_dir(output_dir)
    existing = _load_json_file(state_path)
    worker_root = Path(_worker_root_for_output_dir(output_dir)).expanduser().resolve()

    chunk_groups = _partition_dataset_files_into_chunks(matched_files, files_per_chunk)
    chunks: list[dict[str, object]] = []
    for chunk_idx, chunk_files in enumerate(chunk_groups):
        chunk_output_dir = _chunk_output_dir_for_index(output_dir, chunk_idx)
        chunks.append(
            {
                "chunk_idx": chunk_idx,
                "output_dir": chunk_output_dir,
                "manifest_path": str(_chunk_manifest_path(chunk_output_dir)),
                "data_files": chunk_files,
                "call_id": "",
                "status": _chunk_state_status_for_manifest(chunk_output_dir),
                "attempts": 0,
                "last_call_status": "",
                "last_error": "",
            }
        )

    if existing is None:
        legacy_worker_dirs = sorted(worker_root.glob("worker-*")) if worker_root.exists() else []
        if legacy_worker_dirs:
            legacy_groups = _partition_dataset_files(matched_files, parallel_workers)
            legacy_chunks: list[dict[str, object]] = []
            for worker_idx, worker_files in enumerate(legacy_groups):
                worker_dir = (worker_root / f"worker-{worker_idx:04d}").resolve()
                worker_output_dir = str(worker_dir)
                if _chunk_manifest_path(worker_output_dir).exists():
                    legacy_chunks.append(
                        {
                            "chunk_idx": len(legacy_chunks),
                            "output_dir": worker_output_dir,
                            "manifest_path": str(_chunk_manifest_path(worker_output_dir)),
                            "data_files": worker_files,
                            "call_id": "",
                            "status": "completed",
                            "attempts": 0,
                            "last_call_status": "success",
                            "last_error": "",
                            "legacy_worker_idx": worker_idx,
                        }
                    )
                    continue

                subgroups = _partition_dataset_files_into_chunks(worker_files, files_per_chunk)
                for sub_idx, subgroup in enumerate(subgroups):
                    chunk_output_dir = str(
                        (worker_root / f"chunk-legacy-{worker_idx:04d}-{sub_idx:04d}").resolve()
                    )
                    legacy_chunks.append(
                        {
                            "chunk_idx": len(legacy_chunks),
                            "output_dir": chunk_output_dir,
                            "manifest_path": str(_chunk_manifest_path(chunk_output_dir)),
                            "data_files": subgroup,
                            "call_id": "",
                            "status": _chunk_state_status_for_manifest(chunk_output_dir),
                            "attempts": 0,
                            "last_call_status": "",
                            "last_error": "",
                            "legacy_worker_idx": worker_idx,
                        }
                    )
            state = {
                "version": 2,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "dataset_id": dataset_id,
                "data_files_pattern": data_files,
                "source_split": source_split,
                "output_dir": str(Path(output_dir).expanduser().resolve()),
                "worker_root": str(worker_root),
                "repo_name_or_id": repo_name_or_id,
                "parallel_workers": int(parallel_workers),
                "files_per_chunk": int(files_per_chunk),
                "matched_files": matched_files,
                "chunks": legacy_chunks,
                "merge": {"status": "pending", "manifest_path": ""},
                "upload": {"status": "pending", "repo_id": ""},
                "continuation_call_id": "",
                "migrated_from_legacy_workers": True,
            }
            _write_json_file(state_path, state)
            volume.commit()
            return state

    if existing is None:
        state = {
            "version": 2,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "dataset_id": dataset_id,
            "data_files_pattern": data_files,
            "source_split": source_split,
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "worker_root": str(worker_root),
            "repo_name_or_id": repo_name_or_id,
            "parallel_workers": int(parallel_workers),
            "files_per_chunk": int(files_per_chunk),
            "matched_files": matched_files,
            "chunks": chunks,
            "merge": {"status": "pending", "manifest_path": ""},
            "upload": {"status": "pending", "repo_id": ""},
            "continuation_call_id": "",
        }
        _write_json_file(state_path, state)
        volume.commit()
        return state

    if _state_preserves_existing_chunk_layout(existing):
        existing["updated_at"] = int(time.time())
        existing["dataset_id"] = dataset_id
        existing["data_files_pattern"] = data_files
        existing["source_split"] = source_split
        existing["output_dir"] = str(Path(output_dir).expanduser().resolve())
        existing["worker_root"] = str(worker_root)
        existing["repo_name_or_id"] = repo_name_or_id
        existing["parallel_workers"] = int(parallel_workers)
        existing["files_per_chunk"] = int(files_per_chunk)
        existing["matched_files"] = matched_files
        if "merge" not in existing:
            existing["merge"] = {"status": "pending", "manifest_path": ""}
        if "upload" not in existing:
            existing["upload"] = {"status": "pending", "repo_id": ""}
        if "continuation_call_id" not in existing:
            existing["continuation_call_id"] = ""
        _write_json_file(state_path, existing)
        volume.commit()
        return existing

    existing_chunks = {int(chunk.get("chunk_idx", -1)): dict(chunk) for chunk in existing.get("chunks", [])}
    merged_chunks: list[dict[str, object]] = []
    for chunk in chunks:
        chunk_idx = int(chunk["chunk_idx"])
        prior = existing_chunks.get(chunk_idx, {})
        merged_chunk = dict(chunk)
        if prior:
            merged_chunk["call_id"] = str(prior.get("call_id", "") or "")
            merged_chunk["attempts"] = int(prior.get("attempts", 0) or 0)
            merged_chunk["last_call_status"] = str(prior.get("last_call_status", "") or "")
            merged_chunk["last_error"] = str(prior.get("last_error", "") or "")
            if merged_chunk["status"] != "completed":
                merged_chunk["status"] = str(prior.get("status", "pending") or "pending")
        merged_chunks.append(merged_chunk)

    existing["updated_at"] = int(time.time())
    existing["dataset_id"] = dataset_id
    existing["data_files_pattern"] = data_files
    existing["source_split"] = source_split
    existing["output_dir"] = str(Path(output_dir).expanduser().resolve())
    existing["worker_root"] = str(Path(_worker_root_for_output_dir(output_dir)).expanduser().resolve())
    existing["repo_name_or_id"] = repo_name_or_id
    existing["parallel_workers"] = int(parallel_workers)
    existing["files_per_chunk"] = int(files_per_chunk)
    existing["matched_files"] = matched_files
    existing["chunks"] = merged_chunks
    if "merge" not in existing:
        existing["merge"] = {"status": "pending", "manifest_path": ""}
    if "upload" not in existing:
        existing["upload"] = {"status": "pending", "repo_id": ""}
    if "continuation_call_id" not in existing:
        existing["continuation_call_id"] = ""
    _write_json_file(state_path, existing)
    volume.commit()
    return existing


def _save_parallel_emilia_state(output_dir: str, state: dict[str, object]) -> None:
    state["updated_at"] = int(time.time())
    _write_json_file(_controller_state_path_for_output_dir(output_dir), state)
    volume.commit()


def _remove_chunk_output_dir(chunk_output_dir: str) -> None:
    path = Path(chunk_output_dir).expanduser().resolve()
    if path.exists():
        shutil.rmtree(path)


def _probe_function_call_status(function_call_id: str) -> str:
    if not function_call_id.strip():
        return "missing"
    call = modal.FunctionCall.from_id(function_call_id.strip())
    graph = call.get_call_graph()
    if not graph:
        return "pending"
    return str(graph[0].status.name).lower()


def _refresh_parallel_emilia_state(
    *,
    output_dir: str,
    state: dict[str, object],
) -> dict[str, object]:
    refreshed_chunks: list[dict[str, object]] = []
    for chunk in state.get("chunks", []):
        chunk_state = dict(chunk)
        chunk_output_dir = str(chunk_state.get("output_dir", "") or "")
        manifest_path = _chunk_manifest_path(chunk_output_dir)
        if manifest_path.exists():
            chunk_state["status"] = "completed"
            chunk_state["last_call_status"] = "success"
            refreshed_chunks.append(chunk_state)
            continue

        call_id = str(chunk_state.get("call_id", "") or "")
        if call_id:
            try:
                call_status = _probe_function_call_status(call_id)
            except Exception as exc:
                call_status = "probe_failed"
                chunk_state["last_error"] = f"{type(exc).__name__}: {exc}"
            if call_status == "success":
                try:
                    _wait_for_worker_manifest(Path(chunk_output_dir), max_wait_seconds=30.0)
                    chunk_state["status"] = "completed"
                    chunk_state["last_call_status"] = "success"
                    refreshed_chunks.append(chunk_state)
                    continue
                except FileNotFoundError:
                    call_status = "success_without_manifest"
            chunk_state["last_call_status"] = call_status
            if call_status in {"pending"}:
                chunk_state["status"] = "running"
            else:
                chunk_state["status"] = "pending"
                chunk_state["call_id"] = ""
        else:
            chunk_state["status"] = "pending"

        refreshed_chunks.append(chunk_state)

    state["chunks"] = refreshed_chunks
    _save_parallel_emilia_state(output_dir, state)
    return state


def _all_parallel_chunks_completed(state: dict[str, object]) -> bool:
    chunks = state.get("chunks", []) or []
    return bool(chunks) and all(str(chunk.get("status", "")) == "completed" for chunk in chunks)


def _build_pretokenize_emilia_cmd(
    *,
    quantizers: int,
    output_dir: str,
    dataset_id: str,
    data_files: str,
    source_split: str,
    max_train_samples: int,
    max_validation_samples: int,
    max_test_samples: int,
    min_seconds: float,
    max_seconds: float,
    seq_len: int,
    reference_seq_len: int,
    export_format: str,
    mask_text_loss: bool,
    language_tokens: bool,
    keep_audio_codes: bool,
    emit_static_references: bool,
    validation_count: int,
    test_count: int,
    split_strategy: str,
    max_samples: int,
    batch_max_clips: int,
    batch_max_audio_seconds: float,
    seed: int,
    shard_size: int,
    audio_codec_backend: str,
    audio_codec_source: str,
    audio_codec_model_id: str,
    audio_codec_ckpt_path: str,
    audio_codec_trust_remote_code: bool,
    log_prefix: str,
) -> list[str]:
    cmd = [
        "python",
        "scripts/pretokenize_emilia.py",
        "--output-dir",
        output_dir.strip(),
        "--dataset-id",
        dataset_id.strip(),
        "--data-files",
        data_files.strip(),
        "--source-split",
        source_split.strip(),
        "--num-quantizers",
        str(quantizers),
        "--max-train-samples",
        str(max_train_samples),
        "--max-validation-samples",
        str(max_validation_samples),
        "--max-test-samples",
        str(max_test_samples),
        "--min-seconds",
        str(min_seconds),
        "--max-seconds",
        str(max_seconds),
        "--seq-len",
        str(seq_len),
        "--reference-seq-len",
        str(reference_seq_len),
        "--export-format",
        export_format.strip(),
        "--validation-count",
        str(validation_count),
        "--test-count",
        str(test_count),
        "--split-strategy",
        split_strategy.strip(),
        "--max-samples",
        str(max_samples),
        "--batch-max-clips",
        str(batch_max_clips),
        "--batch-max-audio-seconds",
        str(batch_max_audio_seconds),
        "--seed",
        str(seed),
        "--shard-size",
        str(shard_size),
        "--audio-codec-backend",
        audio_codec_backend.strip(),
        "--audio-codec-source",
        audio_codec_source.strip(),
    ]
    if log_prefix.strip():
        cmd.extend(["--log-prefix", log_prefix.strip()])
    cmd.append("--mask-text-loss" if mask_text_loss else "--no-mask-text-loss")
    if language_tokens:
        cmd.append("--language-tokens")
    if keep_audio_codes:
        cmd.append("--keep-audio-codes")
    if emit_static_references:
        cmd.append("--emit-static-references")
    if audio_codec_model_id.strip():
        cmd.extend(["--audio-codec-model-id", audio_codec_model_id.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    return cmd


def _merge_split_hash(
    split_hashers: dict[str, "hashlib._Hash"],
    split_name: str,
    worker_hash: str,
) -> None:
    if not worker_hash:
        return
    split_hashers[split_name].update(worker_hash.encode("utf-8"))
    split_hashers[split_name].update(b"\n")


def _wait_for_worker_manifest(
    worker_dir: Path,
    *,
    max_wait_seconds: float = 60.0,
    poll_interval_seconds: float = 1.0,
) -> Path:
    manifest_path = worker_dir / "dataset_manifest.json"
    deadline = time.monotonic() + max_wait_seconds
    while True:
        volume.reload()
        if manifest_path.exists():
            return manifest_path
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"Worker manifest missing: {manifest_path}")
        time.sleep(poll_interval_seconds)


def _upload_dataset_folder_to_hf_impl(
    *,
    dataset_dir: str,
    repo_name_or_id: str,
    namespace: str,
    private: bool,
    commit_message: str,
) -> dict[str, object]:
    hf_token = _resolve_hf_token()
    if not hf_token:
        raise RuntimeError("No Hugging Face token found in Modal secrets/environment.")

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    resolved_repo_id = _resolve_hf_repo_id(api, repo_name_or_id, namespace=namespace)
    api.create_repo(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    manifest_path = dataset_path / "dataset_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    card_path = _write_dataset_card(
        dataset_path,
        repo_id=resolved_repo_id,
        manifest=manifest,
    )
    commit_message = commit_message.strip() or (
        f"Upload frozen pretokenized dataset from {dataset_path.name}"
    )

    upload_large = getattr(api, "upload_large_folder", None)
    if callable(upload_large):
        upload_large(
            folder_path=str(dataset_path),
            repo_id=resolved_repo_id,
            repo_type="dataset",
        )
    else:
        api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=resolved_repo_id,
            repo_type="dataset",
            path_in_repo="",
            commit_message=commit_message,
        )

    return {
        "status": "ok",
        "repo_id": resolved_repo_id,
        "dataset_dir": str(dataset_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(card_path),
        "private": bool(private),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={"/vol": volume},
    secrets=[*HF_SECRETS, modal.Secret.from_name("wandb")],
)
def pretokenize_fleurs(
    split: str = "train",
    languages: str = "en hi te es fr de ar sw ta bn zh",
    quantizers: int = 1,
    max_samples_per_language: int = 0,
    output_dir: str = "",
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "official_fish",
    audio_codec_model_id: str = "",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
):
    out_dir = output_dir.strip() or f"/vol/data/fleurs_pretok_q{quantizers}"
    cmd = [
        "python",
        "scripts/pretokenize_fleurs.py",
        "--languages",
        *languages.split(),
        "--split",
        split,
        "--num-quantizers",
        str(quantizers),
        "--output-dir",
        out_dir,
        "--max-samples-per-language",
        str(max_samples_per_language),
        "--audio-codec-backend",
        audio_codec_backend.strip(),
    ]
    if audio_codec_source.strip():
        cmd.extend(["--audio-codec-source", audio_codec_source.strip()])
    if audio_codec_model_id.strip():
        cmd.extend(["--audio-codec-model-id", audio_codec_model_id.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    return {
        "output_dir": out_dir,
        "split": split,
        "quantizers": quantizers,
        "audio_codec_backend": audio_codec_backend,
        "audio_codec_source": audio_codec_source,
        "audio_codec_model_id": audio_codec_model_id,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0, max_delay=60.0),
)
def pretokenize_emilia_subset(
    quantizers: int = 8,
    output_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
    dataset_id: str = "amphion/Emilia-Dataset",
    data_files: str = "Emilia/EN/*.tar",
    source_split: str = "train",
    max_train_samples: int = 0,
    max_validation_samples: int = 0,
    max_test_samples: int = 0,
    min_seconds: float = 1.0,
    max_seconds: float = 30.0,
    seq_len: int = 4096,
    reference_seq_len: int = 1024,
    export_format: str = "codec_only",
    mask_text_loss: bool = True,
    language_tokens: bool = False,
    keep_audio_codes: bool = False,
    emit_static_references: bool = False,
    validation_count: int = 0,
    test_count: int = 0,
    split_strategy: str = "train_only",
    max_samples: int = 0,
    batch_max_clips: int = 16,
    batch_max_audio_seconds: float = 180.0,
    seed: int = 42,
    shard_size: int = 10_000,
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "hf_pretrained",
    audio_codec_model_id: str = "kyutai/mimi",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
    log_prefix: str = "",
):
    print(
        "[pretokenize_emilia_subset] starting "
        f"log_prefix={log_prefix or '-'} output_dir={output_dir} data_files={data_files} "
        f"max_samples={max_samples} split_strategy={split_strategy}",
        flush=True,
    )
    cmd = _build_pretokenize_emilia_cmd(
        quantizers=quantizers,
        output_dir=output_dir,
        dataset_id=dataset_id,
        data_files=data_files,
        source_split=source_split,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        seq_len=seq_len,
        reference_seq_len=reference_seq_len,
        export_format=export_format,
        mask_text_loss=mask_text_loss,
        language_tokens=language_tokens,
        keep_audio_codes=keep_audio_codes,
        emit_static_references=emit_static_references,
        validation_count=validation_count,
        test_count=test_count,
        split_strategy=split_strategy,
        max_samples=max_samples,
        batch_max_clips=batch_max_clips,
        batch_max_audio_seconds=batch_max_audio_seconds,
        seed=seed,
        shard_size=shard_size,
        audio_codec_backend=audio_codec_backend,
        audio_codec_source=audio_codec_source,
        audio_codec_model_id=audio_codec_model_id,
        audio_codec_ckpt_path=audio_codec_ckpt_path,
        audio_codec_trust_remote_code=audio_codec_trust_remote_code,
        log_prefix=log_prefix,
    )
    print(f"[pretokenize_emilia_subset] command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    manifest_path = Path(output_dir) / "dataset_manifest.json"
    if manifest_path.exists():
        print(
            f"[pretokenize_emilia_subset] manifest_ready={manifest_path}",
            flush=True,
        )
    return {
        "status": "ok",
        "output_dir": output_dir,
        "dataset_id": dataset_id,
        "data_files": data_files,
        "quantizers": quantizers,
        "audio_codec_backend": audio_codec_backend,
        "audio_codec_source": audio_codec_source,
        "audio_codec_model_id": audio_codec_model_id,
    }


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def upload_dataset_folder_to_hf(
    dataset_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
    repo_name_or_id: str = "emilia-en-mimi-q8-s4096",
    namespace: str = "",
    private: bool = True,
    commit_message: str = "",
):
    print(
        "[upload_dataset_folder_to_hf] starting "
        f"dataset_dir={dataset_dir} repo_name_or_id={repo_name_or_id} namespace={namespace or '-'}",
        flush=True,
    )
    result = _upload_dataset_folder_to_hf_impl(
        dataset_dir=dataset_dir,
        repo_name_or_id=repo_name_or_id,
        namespace=namespace,
        private=private,
        commit_message=commit_message,
    )
    print(
        f"[upload_dataset_folder_to_hf] complete repo_id={result.get('repo_id')} "
        f"dataset_dir={result.get('dataset_dir')}",
        flush=True,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={"/vol": volume},
)
def assign_reference_pairs_for_dataset(
    dataset_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
):
    print(
        f"[assign_reference_pairs_for_dataset] starting dataset_dir={dataset_dir}",
        flush=True,
    )
    result = _assign_reference_pairs_impl(dataset_dir=dataset_dir)
    print(
        f"[assign_reference_pairs_for_dataset] complete status={result.get('status')} "
        f"dataset_dir={dataset_dir}",
        flush=True,
    )
    volume.commit()
    return result


def _merge_pretokenized_emilia_workers_impl(
    *,
    worker_output_dirs: list[str],
    output_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
    dataset_id: str = "amphion/Emilia-Dataset",
    data_files: str = "Emilia/EN/*.tar",
    source_split: str = "train",
) -> dict[str, object]:
    print(
        "[merge_pretokenized_emilia_workers] starting "
        f"workers={len(worker_output_dirs)} output_dir={output_dir}",
        flush=True,
    )
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        existing = list(output_path.iterdir())
        if existing:
            raise RuntimeError(
                f"Refusing to merge into non-empty output dir: {output_path}"
            )
    output_path.mkdir(parents=True, exist_ok=True)

    merged_counts = {"train": 0, "validation": 0, "test": 0}
    merged_reference_counts = {"train": 0, "validation": 0, "test": 0}
    merged_dropped = {
        "missing_fields": 0,
        "duration": 0,
        "token_overflow": 0,
        "decode_error": 0,
        "assignment_skip": 0,
    }
    merged_written_files: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    split_hashers = {
        "train": hashlib.sha256(),
        "validation": hashlib.sha256(),
        "test": hashlib.sha256(),
    }
    shard_index = {"train": 0, "validation": 0, "test": 0}
    template_manifest: dict[str, object] | None = None

    for worker_dir_raw in sorted(worker_output_dirs):
        worker_dir = Path(worker_dir_raw).resolve()
        print(
            f"[merge_pretokenized_emilia_workers] reading worker_dir={worker_dir}",
            flush=True,
        )
        manifest_path = _wait_for_worker_manifest(worker_dir)
        worker_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if template_manifest is None:
            template_manifest = worker_manifest

        worker_counts = worker_manifest.get("counts", {}) or {}
        worker_reference = (
            (worker_manifest.get("reference_conditioning", {}) or {}).get(
                "counts_with_reference", {}
            )
            or {}
        )
        worker_dropped = worker_manifest.get("dropped", {}) or {}
        worker_hashes = worker_manifest.get("sample_id_sha256_by_split", {}) or {}
        merged_dropped["missing_fields"] += int(worker_dropped.get("missing_fields", 0) or 0)
        merged_dropped["duration"] += int(worker_dropped.get("duration", 0) or 0)
        merged_dropped["token_overflow"] += int(
            worker_dropped.get("token_overflow", 0) or 0
        )
        merged_dropped["decode_error"] += int(worker_dropped.get("decode_error", 0) or 0)
        merged_dropped["assignment_skip"] += int(
            worker_dropped.get("assignment_skip", 0) or 0
        )
        for split_name in ("train", "validation", "test"):
            merged_counts[split_name] += int(worker_counts.get(split_name, 0) or 0)
            merged_reference_counts[split_name] += int(
                worker_reference.get(split_name, 0) or 0
            )
            _merge_split_hash(
                split_hashers,
                split_name,
                str(worker_hashes.get(split_name, "") or ""),
            )

            src_split_dir = worker_dir / split_name / "lang=en"
            if not src_split_dir.exists():
                continue
            for src_file in sorted(src_split_dir.glob("*.parquet")):
                dst_file = (
                    output_path
                    / split_name
                    / "lang=en"
                    / f"part-{shard_index[split_name]:05d}.parquet"
                )
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                merged_written_files[split_name].append(str(dst_file.relative_to(output_path)))
                shard_index[split_name] += 1

    base_manifest = template_manifest or {}
    reference_cfg = dict(base_manifest.get("reference_conditioning", {}) or {})
    reference_cfg["counts_with_reference"] = merged_reference_counts
    merged_manifest = {
        "dataset_name": dataset_id,
        "data_files": data_files,
        "source_split": source_split,
        "artifact_kind": base_manifest.get("artifact_kind", "model_ready_tts"),
        "objective": base_manifest.get("objective", "text_audio_pair"),
        "reference_conditioning": reference_cfg,
        "num_quantizers": int(base_manifest.get("num_quantizers", 8) or 8),
        "tokenizer_name": str(base_manifest.get("tokenizer_name", "")),
        "seq_len": int(base_manifest.get("seq_len", 4096) or 4096),
        "mask_text_loss": bool(base_manifest.get("mask_text_loss", True)),
        "language_tokens": bool(base_manifest.get("language_tokens", False)),
        "split_strategy": str(base_manifest.get("split_strategy", "train_only")),
        "selection": dict(base_manifest.get("selection", {}) or {}),
        "audio_codec": dict(base_manifest.get("audio_codec", {}) or {}),
        "counts": merged_counts,
        "dropped": merged_dropped,
        "shard_size": int(base_manifest.get("shard_size", 10_000) or 10_000),
        "written_files": merged_written_files,
        "source_scan": {
            "workers": len(worker_output_dirs),
            "worker_dirs": [str(Path(path).resolve()) for path in sorted(worker_output_dirs)],
        },
        "sample_id_sha256_by_split": {
            split_name: hasher.hexdigest() for split_name, hasher in split_hashers.items()
        },
    }
    manifest_path = _write_dataset_manifest(output_path, merged_manifest)
    print(
        "[merge_pretokenized_emilia_workers] complete "
        f"manifest_path={manifest_path} train={merged_counts['train']} "
        f"validation={merged_counts['validation']} test={merged_counts['test']}",
        flush=True,
    )
    return {
        "status": "ok",
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "counts": merged_counts,
    }


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={"/vol": volume},
)
def merge_pretokenized_emilia_workers(
    worker_output_dirs: list[str],
    output_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
    dataset_id: str = "amphion/Emilia-Dataset",
    data_files: str = "Emilia/EN/*.tar",
    source_split: str = "train",
):
    result = _merge_pretokenized_emilia_workers_impl(
        worker_output_dirs=worker_output_dirs,
        output_dir=output_dir,
        dataset_id=dataset_id,
        data_files=data_files,
        source_split=source_split,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def pretokenize_emilia_subset_to_hf(
    quantizers: int = 8,
    output_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
    dataset_id: str = "amphion/Emilia-Dataset",
    data_files: str = "Emilia/EN/*.tar",
    source_split: str = "train",
    max_train_samples: int = 0,
    max_validation_samples: int = 0,
    max_test_samples: int = 0,
    min_seconds: float = 1.0,
    max_seconds: float = 30.0,
    seq_len: int = 4096,
    reference_seq_len: int = 1024,
    export_format: str = "codec_only",
    mask_text_loss: bool = True,
    language_tokens: bool = False,
    keep_audio_codes: bool = False,
    emit_static_references: bool = False,
    validation_count: int = 0,
    test_count: int = 0,
    split_strategy: str = "train_only",
    max_samples: int = 0,
    batch_max_clips: int = 16,
    batch_max_audio_seconds: float = 180.0,
    seed: int = 42,
    shard_size: int = 10_000,
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "hf_pretrained",
    audio_codec_model_id: str = "kyutai/mimi",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
    log_prefix: str = "",
    repo_name_or_id: str = "emilia-en-mimi-q8-s4096",
    namespace: str = "",
    private: bool = True,
    commit_message: str = "",
):
    print(
        "[pretokenize_emilia_subset_to_hf] starting "
        f"log_prefix={log_prefix or '-'} output_dir={output_dir} repo_name_or_id={repo_name_or_id}",
        flush=True,
    )
    cmd = _build_pretokenize_emilia_cmd(
        quantizers=quantizers,
        output_dir=output_dir,
        dataset_id=dataset_id,
        data_files=data_files,
        source_split=source_split,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        seq_len=seq_len,
        reference_seq_len=reference_seq_len,
        export_format=export_format,
        mask_text_loss=mask_text_loss,
        language_tokens=language_tokens,
        keep_audio_codes=keep_audio_codes,
        emit_static_references=emit_static_references,
        validation_count=validation_count,
        test_count=test_count,
        split_strategy=split_strategy,
        max_samples=max_samples,
        batch_max_clips=batch_max_clips,
        batch_max_audio_seconds=batch_max_audio_seconds,
        seed=seed,
        shard_size=shard_size,
        audio_codec_backend=audio_codec_backend,
        audio_codec_source=audio_codec_source,
        audio_codec_model_id=audio_codec_model_id,
        audio_codec_ckpt_path=audio_codec_ckpt_path,
        audio_codec_trust_remote_code=audio_codec_trust_remote_code,
        log_prefix=log_prefix,
    )
    print(f"[pretokenize_emilia_subset_to_hf] command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    manifest_path = Path(output_dir) / "dataset_manifest.json"
    if manifest_path.exists():
        print(
            f"[pretokenize_emilia_subset_to_hf] manifest_ready={manifest_path}",
            flush=True,
        )
    print(
        "[pretokenize_emilia_subset_to_hf] uploading dataset folder to hf "
        f"repo_name_or_id={repo_name_or_id}",
        flush=True,
    )
    upload_result = _upload_dataset_folder_to_hf_impl(
        dataset_dir=output_dir,
        repo_name_or_id=repo_name_or_id,
        namespace=namespace,
        private=private,
        commit_message=commit_message,
    )
    print(
        f"[pretokenize_emilia_subset_to_hf] upload_complete repo_id={upload_result.get('repo_id')}",
        flush=True,
    )
    return {
        "status": "ok",
        "output_dir": output_dir,
        "dataset_id": dataset_id,
        "data_files": data_files,
        "quantizers": quantizers,
        "upload": upload_result,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def pretokenize_fleurs_s1(
    split: str = "train",
    languages: str = "en",
    quantizers: int = 9,
    max_samples_per_language: int = 0,
    output_dir: str = "/vol/data/fleurs_pretok_s1_q9",
    audio_codec_source: str = "official_fish",
    audio_codec_model_id: str = "jordand/fish-s1-dac-min",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
):
    cmd = [
        "python",
        "codecs/s1_dac/scripts/pretokenize_fleurs.py",
        "--languages",
        *languages.split(),
        "--split",
        split,
        "--num-quantizers",
        str(quantizers),
        "--output-dir",
        output_dir,
        "--max-samples-per-language",
        str(max_samples_per_language),
        "--audio-codec-source",
        audio_codec_source.strip(),
        "--audio-codec-model-id",
        audio_codec_model_id.strip(),
    ]
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    return {
        "output_dir": output_dir,
        "split": split,
        "quantizers": quantizers,
        "audio_codec_source": audio_codec_source,
        "audio_codec_model_id": audio_codec_model_id,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def pretokenize_single_wav(
    input_wav_path: str = "/vol/data/raw/download.wav",
    text: str = "",
    lang: str = "en",
    sample_id: str = "download_001",
    quantizers: int = 8,
    max_seconds: float = 20.0,
    output_dir: str = "/vol/data/custom_download_q8",
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "official_fish",
    audio_codec_model_id: str = "",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
):
    cmd = [
        "python",
        "scripts/pretokenize_single_wav.py",
        "--input-wav",
        input_wav_path,
        "--output-dir",
        output_dir,
        "--split",
        "train",
        "--lang",
        lang,
        "--sample-id",
        sample_id,
        "--num-quantizers",
        str(quantizers),
        "--max-seconds",
        str(max_seconds),
        "--audio-codec-backend",
        audio_codec_backend.strip(),
    ]
    if text.strip():
        cmd.extend(["--text", text.strip()])
    if audio_codec_source.strip():
        cmd.extend(["--audio-codec-source", audio_codec_source.strip()])
    if audio_codec_model_id.strip():
        cmd.extend(["--audio-codec-model-id", audio_codec_model_id.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    return {
        "status": "ok",
        "output_dir": output_dir,
        "input_wav_path": input_wav_path,
        "quantizers": quantizers,
        "max_seconds": max_seconds,
        "sample_id": sample_id,
        "audio_codec_backend": audio_codec_backend,
        "audio_codec_source": audio_codec_source,
        "audio_codec_model_id": audio_codec_model_id,
    }


@app.function(
    image=s2_pro_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def fish_speech_reference_tts(
    input_wav_path: str = "/vol/data/raw/reference.wav",
    devanagari_text: str = "",
    latin_text: str = "",
    prompt_text: str = "",
    checkpoint_repo_id: str = "fishaudio/openaudio-s1-mini",
    checkpoint_dir: str = "/vol/checkpoints/openaudio-s1-mini",
    output_dir: str = "/vol/outputs/fish_speech/openaudio_s1_reference_demo",
    device: str = "cuda",
):
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    input_path = Path(input_wav_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference WAV not found: {input_wav_path}")
    if not devanagari_text.strip():
        raise ValueError("devanagari_text must be non-empty")
    if not latin_text.strip():
        raise ValueError("latin_text must be non-empty")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_token = _resolve_hf_token() or None
    snapshot_download(
        repo_id=checkpoint_repo_id,
        local_dir=str(checkpoint_path),
        token=hf_token,
    )
    volume.commit()

    env = _repo_env()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    prompt_text_value = prompt_text.strip()
    if not prompt_text_value:
        asr = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if device.startswith("cuda") else -1,
        )
        asr_result = asr(str(input_path))
        if isinstance(asr_result, dict):
            prompt_text_value = str(asr_result.get("text", "")).strip()
        else:
            prompt_text_value = str(asr_result).strip()
        if not prompt_text_value:
            raise RuntimeError("Failed to auto-transcribe reference audio for Fish Speech.")

    codec_ckpt = checkpoint_path / "codec.pth"
    ref_recon_wav = output_path / "reference_reconstructed.wav"
    subprocess.run(
        [
            "python",
            "fish_speech/models/dac/inference.py",
            "-i",
            str(input_path),
            "-o",
            str(ref_recon_wav),
            "--checkpoint-path",
            str(codec_ckpt),
            "--device",
            device,
        ],
        check=True,
        cwd="/root/fish-speech",
        env=env,
    )
    prompt_tokens_path = ref_recon_wav.with_suffix(".npy")
    if not prompt_tokens_path.exists():
        raise FileNotFoundError(
            f"Prompt token file was not created: {prompt_tokens_path}"
        )

    generations = [
        ("devanagari_hindi", devanagari_text.strip()),
        ("latin_hindi", latin_text.strip()),
    ]
    outputs: dict[str, str] = {}
    for name, text in generations:
        gen_dir = output_path / name
        gen_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "python",
                "fish_speech/models/text2semantic/inference.py",
                "--text",
                text,
                "--prompt-text",
                prompt_text_value,
                "--prompt-tokens",
                str(prompt_tokens_path),
                "--checkpoint-path",
                str(checkpoint_path),
                "--device",
                device,
                "--output-dir",
                str(gen_dir),
            ],
            check=True,
            cwd="/root/fish-speech",
            env=env,
        )
        codes_path = gen_dir / "codes_0.npy"
        if not codes_path.exists():
            raise FileNotFoundError(f"Fish Speech codes not found: {codes_path}")
        wav_path = output_path / f"{name}.wav"
        subprocess.run(
            [
                "python",
                "fish_speech/models/dac/inference.py",
                "-i",
                str(codes_path),
                "-o",
                str(wav_path),
                "--checkpoint-path",
                str(codec_ckpt),
                "--device",
                device,
            ],
            check=True,
            cwd="/root/fish-speech",
            env=env,
        )
        outputs[name] = str(wav_path)

    metadata = {
        "status": "ok",
        "model_repo_id": checkpoint_repo_id,
        "checkpoint_dir": str(checkpoint_path),
        "input_wav_path": str(input_path),
        "prompt_text": prompt_text_value,
        "prompt_tokens_path": str(prompt_tokens_path),
        "reference_reconstructed_wav": str(ref_recon_wav),
        "outputs": outputs,
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    volume.commit()
    return metadata


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def fish_audio_s2_pro_reference_tts(
    input_wav_path: str = "/vol/data/raw/reference.wav",
    devanagari_text: str = "",
    latin_text: str = "",
    prompt_text: str = "",
    checkpoint_repo_id: str = "fishaudio/s2-pro",
    checkpoint_dir: str = "/vol/checkpoints/fishaudio_s2_pro",
    output_dir: str = "/vol/outputs/fish_audio_s2_pro/reference_demo",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    import httpx
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    input_path = Path(input_wav_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference WAV not found: {input_wav_path}")
    if not devanagari_text.strip():
        raise ValueError("devanagari_text must be non-empty")
    if not latin_text.strip():
        raise ValueError("latin_text must be non-empty")
    sglang_omni_repo = _resolve_sglang_omni_repo()
    if sglang_omni_repo is None:
        runtime_clone_path = Path("/root/sglang-omni")
        runtime_clone_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/sgl-project/sglang-omni",
                str(runtime_clone_path),
            ],
            check=True,
        )
        sglang_omni_repo = _resolve_sglang_omni_repo()
    if sglang_omni_repo is None:
        raise RuntimeError("sglang-omni repo is still unavailable after runtime clone.")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_token = _resolve_hf_token() or None
    snapshot_download(
        repo_id=checkpoint_repo_id,
        local_dir=str(checkpoint_path),
        token=hf_token,
    )
    volume.commit()

    env = _repo_env()
    env["PYTHONUNBUFFERED"] = "1"
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    prompt_text_value = prompt_text.strip()
    if not prompt_text_value:
        asr = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-small",
            device=0,
            chunk_length_s=30,
        )
        asr_result = asr(str(input_path), return_timestamps=True)
        if isinstance(asr_result, dict):
            prompt_text_value = str(asr_result.get("text", "")).strip()
        else:
            prompt_text_value = str(asr_result).strip()
        if not prompt_text_value:
            raise RuntimeError("Failed to auto-transcribe reference audio for S2 Pro.")

    server_cmd = [
        "python",
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        str(checkpoint_path),
        "--config",
        str(sglang_omni_repo / "examples" / "configs" / "s2pro_tts.yaml"),
        "--host",
        host,
        "--port",
        str(port),
        "--model-name",
        "fishaudio-s2-pro",
        "--log-level",
        "info",
    ]
    server_log_path = output_path / "server.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    server_proc = subprocess.Popen(
        server_cmd,
        cwd="/root/sglang-omni",
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        health_url = f"http://{host}:{port}/health"
        started = False
        for _ in range(600):
            if server_proc.poll() is not None:
                server_log.flush()
                startup_logs = server_log_path.read_text(encoding="utf-8") if server_log_path.exists() else ""
                raise RuntimeError(
                    "S2 Pro server exited before becoming healthy.\n"
                    + startup_logs[-12000:]
                )
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(health_url)
                if resp.status_code == 200:
                    payload = resp.json()
                    if bool(payload.get("running")):
                        started = True
                        break
            except Exception:
                pass
            time.sleep(2.0)
        if not started:
            server_log.flush()
            startup_logs = server_log_path.read_text(encoding="utf-8") if server_log_path.exists() else ""
            raise RuntimeError(
                "Timed out waiting for S2 Pro server health.\n"
                + startup_logs[-12000:]
            )

        requests_to_run = [
            ("devanagari_hindi", devanagari_text.strip()),
            ("latin_hindi", latin_text.strip()),
        ]
        outputs: dict[str, str] = {}
        request_logs: list[dict[str, object]] = []

        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            for name, text in requests_to_run:
                response = client.post(
                    f"http://{host}:{port}/v1/audio/speech",
                    json={
                        "model": "fishaudio-s2-pro",
                        "input": text,
                        "response_format": "wav",
                        "references": [
                            {
                                "audio_path": str(input_path),
                                "text": prompt_text_value,
                            }
                        ],
                    },
                )
                response.raise_for_status()
                wav_path = output_path / f"{name}.wav"
                wav_path.write_bytes(response.content)
                outputs[name] = str(wav_path)
                request_logs.append(
                    {
                        "name": name,
                        "text": text,
                        "wav_path": str(wav_path),
                        "bytes": len(response.content),
                    }
                )

        metadata = {
            "status": "ok",
            "model_repo_id": checkpoint_repo_id,
            "checkpoint_dir": str(checkpoint_path),
            "input_wav_path": str(input_path),
            "prompt_text": prompt_text_value,
            "outputs": outputs,
            "requests": request_logs,
            "server_log_path": str(server_log_path),
        }
        metadata_path = output_path / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        volume.commit()
        return metadata
    finally:
        server_log.flush()
        server_log.close()
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=20)


@app.function(
    image=s2_pro_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def fish_audio_s2_pro_sp_sp010_hindi_demo():
    return fish_audio_s2_pro_reference_tts.local(
        input_wav_path="/vol/data/raw/SP_SP010_1.wav",
        prompt_text=(
            "Oh god, I'm just so happy. Oh, and it's all your fault. Oh honestly, "
            "probably still your house. But still I mean running the dishes through "
            "the dishwasher, putting them up. Yeah yeah, alright, alright yeah, okay. "
            "I guess I do have a lot of explaining to do don't I? Huh, but feeding off "
            "of other people just feels weird. Like, like I'm cheating you know. "
            "You're sorry? Oh baby I'm sorry too. I'm sorry this whole thing happened."
        ),
        devanagari_text=(
            "आज शाम मौसम बहुत सुहावना है, हल्की ठंडी हवा चल रही है और दूर कहीं से "
            "चाय की खुशबू आ रही है। मैं बस थोड़ा टहलने निकला हूँ और सोच रहा हूँ कि "
            "ज़िंदगी में कभी-कभी धीरे चलना भी ज़रूरी होता है।"
        ),
        latin_text=(
            "Aaj shaam mausam bahut suhaavna hai, halki thandi hawa chal rahi hai aur "
            "door kahin se chai ki khushboo aa rahi hai. Main bas thoda tahalne nikla "
            "hoon aur soch raha hoon ki zindagi mein kabhi-kabhi dheere chalna bhi "
            "zaroori hota hai."
        ),
        output_dir="/vol/outputs/fish_audio_s2_pro/sp_sp010_hindi_demo",
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def codec_reconstruction_report(
    input_wav_bytes: bytes,
    input_filename: str = "input.wav",
    text: str = "",
    lang: str = "en",
    sample_id: str = "sample_0001",
    quantizers: int = 8,
    max_seconds: float = 20.0,
    output_dir: str = "/vol/outputs/codec_recon/mimi_q8",
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "hf_pretrained",
    audio_codec_model_id: str = "",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
):
    tmp_dir = Path("/vol/data/raw/codec_reconstruction")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_dir / _safe_slug(input_filename)
    input_path.write_bytes(input_wav_bytes)

    cmd = [
        "python",
        "scripts/research/codec_reconstruction_report.py",
        "--input-wav",
        str(input_path),
        "--output-dir",
        output_dir,
        "--lang",
        lang,
        "--sample-id",
        sample_id,
        "--num-quantizers",
        str(quantizers),
        "--max-seconds",
        str(max_seconds),
        "--audio-codec-backend",
        audio_codec_backend.strip(),
    ]
    if text.strip():
        cmd.extend(["--text", text.strip()])
    if audio_codec_source.strip():
        cmd.extend(["--audio-codec-source", audio_codec_source.strip()])
    if audio_codec_model_id.strip():
        cmd.extend(["--audio-codec-model-id", audio_codec_model_id.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")

    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    report_path = Path(output_dir) / "report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "output_dir": output_dir,
        "report_path": str(report_path),
        "report": report,
        "audio_codec_backend": audio_codec_backend,
        "audio_codec_source": audio_codec_source,
        "audio_codec_model_id": audio_codec_model_id,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def pretokenize_single_wav_s1(
    input_wav_path: str = "/vol/data/raw/download.wav",
    text: str = "",
    lang: str = "en",
    sample_id: str = "download_001",
    quantizers: int = 9,
    max_seconds: float = 20.0,
    output_dir: str = "/vol/data/custom_download_s1_q9",
    audio_codec_source: str = "official_fish",
    audio_codec_model_id: str = "jordand/fish-s1-dac-min",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
):
    cmd = [
        "python",
        "codecs/s1_dac/scripts/pretokenize_single_wav.py",
        "--input-wav",
        input_wav_path,
        "--output-dir",
        output_dir,
        "--split",
        "train",
        "--lang",
        lang,
        "--sample-id",
        sample_id,
        "--num-quantizers",
        str(quantizers),
        "--max-seconds",
        str(max_seconds),
        "--audio-codec-source",
        audio_codec_source.strip(),
        "--audio-codec-model-id",
        audio_codec_model_id.strip(),
    ]
    if text.strip():
        cmd.extend(["--text", text.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    return {
        "status": "ok",
        "output_dir": output_dir,
        "input_wav_path": input_wav_path,
        "quantizers": quantizers,
        "max_seconds": max_seconds,
        "sample_id": sample_id,
        "audio_codec_source": audio_codec_source,
        "audio_codec_model_id": audio_codec_model_id,
    }


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=[*HF_SECRETS, modal.Secret.from_name("wandb")],
)
def train(
    path: str = "q1",
    experiment_id: str = "",
    campaign_id: str = "",
    phase: str = "",
    stage: str = "",
    variant: str = "",
    steps: int = 0,
    track: str = "",
    axis: str = "",
    family: str = "",
    question: str = "",
    hypothesis: str = "",
    brief_path: str = "",
    baseline_experiment_id: str = "",
    owner: str = "",
    tags: str = "",
    num_quantizers: int = 0,
    seed: int = -1,
    deterministic: bool = False,
    overfit_num_samples: int = 0,
    dataset_path: str = "",
    checkpoint_interval: int = 0,
    checkpoint_keep_latest_k: int = 0,
    checkpoint_folder: str = "",
    checkpoint_async_mode: str = "async",
    audio_codec_backend: str = "",
    audio_codec_source: str = "",
    audio_codec_model_id: str = "",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
    hf_repo_id: str = "",
    hf_repo_private: bool = False,
    hf_collection_slug: str = "",
    hf_upload_every: int = 200,
    wandb_group: str = "",
    wandb_tags: str = "",
    overrides_json: str = "",
):
    config_map = {
        "q1": "config/tinyaya_q1_fleurs.toml",
        "q8": "config/tinyaya_q8_fleurs.toml",
        "overfit1": "config/tinyaya_q1_fleurs_overfit_1sample.toml",
        "overfit_smoke": "config/tinyaya_q1_fleurs_overfit_1sample_smoke.toml",
        "overfit_strict": "config/tinyaya_q1_fleurs_overfit_1sample_strict.toml",
        "overfit_viz5": "config/tinyaya_q1_fleurs_overfit_1sample_viz5.toml",
        # Canonical codec-aware profile keys.
        "mimi/overfit_download_q8": "codecs/mimi/configs/tinyaya_mimi_q8_download_overfit_1sample.toml",
        "mimi/ablation_emilia40k_q8_s4096_en": "codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en.toml",
        "mimi/clone_flat_emilia40k_q8_s4096_en": "codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en_clone_flat.toml",
        "mimi/clone_grouped_emilia40k_q8_s4096_en": "codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en_clone_grouped.toml",
        "s1_dac/overfit_download_q9": "codecs/s1_dac/configs/tinyaya_s1_q9_download_overfit_1sample.toml",
        "spark_bicodec/overfit_download_q1": "codecs/spark_bicodec/configs/tinyaya_spark_q1_download_overfit_1sample.toml",
        "spark_bicodec/baseline_fleurs_q1_ehite": "codecs/spark_bicodec/configs/tinyaya_spark_q1_fleurs_ehite_baseline.toml",
        "dualcodec/overfit_download_12hz_q1": "codecs/dualcodec/configs/tinyaya_dualcodec_12hz_q1_download_overfit_1sample.toml",
        "dualcodec/overfit_download_12hz_q8": "codecs/dualcodec/configs/tinyaya_dualcodec_12hz_q8_download_overfit_1sample.toml",
        "dualcodec/overfit_download_25hz_q12": "codecs/dualcodec/configs/tinyaya_dualcodec_25hz_q12_download_overfit_1sample.toml",
        "dualcodec/smoke_download_12hz_q1": "codecs/dualcodec/configs/tinyaya_dualcodec_12hz_q1_download_smoke_5step.toml",
        "dualcodec/overfit_download_q8": "codecs/dualcodec/configs/tinyaya_dualcodec_12hz_q8_download_overfit_1sample.toml",
        "dualcodec/overfit_download_q12": "codecs/dualcodec/configs/tinyaya_dualcodec_25hz_q12_download_overfit_1sample.toml",
        "qwen_codec/smoke_download_12hz_q16": "codecs/qwen_codec/configs/tinyaya_qwen12hz_q16_download_smoke_5step.toml",
        "qwen_codec/overfit_download_12hz_q16": "codecs/qwen_codec/configs/tinyaya_qwen12hz_q16_download_overfit_1sample.toml",
        # Legacy aliases retained for compatibility.
        "overfit_download_q8": "codecs/mimi/configs/tinyaya_mimi_q8_download_overfit_1sample.toml",
        "overfit_download_s1_q10": "codecs/s1_dac/configs/tinyaya_s1_q9_download_overfit_1sample.toml",
    }
    deprecated_path_aliases = {"overfit_download_q8", "overfit_download_s1_q10"}
    if path not in config_map:
        raise ValueError(
            "path must be one of: q1, q8, overfit1, overfit_smoke, overfit_strict, "
            "overfit_viz5, mimi/overfit_download_q8, mimi/ablation_emilia40k_q8_s4096_en, "
            "mimi/clone_flat_emilia40k_q8_s4096_en, mimi/clone_grouped_emilia40k_q8_s4096_en, "
            "s1_dac/overfit_download_q9, "
            "spark_bicodec/overfit_download_q1, "
            "spark_bicodec/baseline_fleurs_q1_ehite, "
            "dualcodec/overfit_download_12hz_q1, "
            "dualcodec/overfit_download_12hz_q8, "
            "dualcodec/overfit_download_25hz_q12, "
            "dualcodec/smoke_download_12hz_q1, "
            "qwen_codec/smoke_download_12hz_q16, "
            "qwen_codec/overfit_download_12hz_q16, "
            "dualcodec/overfit_download_q8, dualcodec/overfit_download_q12, "
            "overfit_download_q8, overfit_download_s1_q10"
        )
    if path in deprecated_path_aliases:
        print(
            f"[DEPRECATED] modal train path '{path}' is an alias. "
            "Use codec-aware path IDs under '<codec>/<profile>'.",
            flush=True,
        )
    config_file = config_map[path]
    run_defaults = _load_run_name_defaults(config_file)
    if int(num_quantizers) > 0:
        resolved_q = int(num_quantizers)
    else:
        cfg_path = Path(REMOTE_REPO_ROOT) / config_file
        if not cfg_path.exists():
            cfg_path = REPO_ROOT / config_file
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        resolved_q = int(raw.get("model", {}).get("num_quantizers", 1))

    run_name = _resolve_run_name(
        model_name=str(run_defaults["model_name"]),
        dataset_name=str(run_defaults["dataset_name"]),
        num_quantizers=resolved_q,
        seq_len=int(run_defaults["seq_len"]),
        pretrained=bool(run_defaults["pretrained"]),
        experiment_id=experiment_id.strip(),
    )

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "-m",
        "torchtitan.train",
        "--job.config_file",
        config_file,
        "--job.dump_folder",
        "/vol/outputs",
    ]
    if experiment_id:
        cmd.extend(["--experiment.id", experiment_id])
    if campaign_id:
        cmd.extend(["--experiment.campaign_id", campaign_id])
    if phase:
        cmd.extend(["--experiment.phase", phase])
    if stage:
        cmd.extend(["--experiment.stage", stage])
    if variant:
        cmd.extend(["--experiment.variant", variant])
    if track:
        cmd.extend(["--experiment.track", track])
    if axis:
        cmd.extend(["--experiment.axis", axis])
    if family:
        cmd.extend(["--experiment.family", family])
    if question:
        cmd.extend(["--experiment.question", question])
    if hypothesis:
        cmd.extend(["--experiment.hypothesis", hypothesis])
    if brief_path:
        cmd.extend(["--experiment.brief_path", brief_path])
    if baseline_experiment_id:
        cmd.extend(["--experiment.baseline_experiment_id", baseline_experiment_id])
    if owner:
        cmd.extend(["--experiment.owner", owner])
    if tags:
        cmd.extend(["--experiment.tags", tags])
    if steps > 0:
        cmd.extend(["--training.steps", str(steps)])
    if num_quantizers > 0:
        cmd.extend(["--model.num_quantizers", str(num_quantizers)])
    if seed >= 0:
        cmd.extend(["--training.seed", str(seed)])
    if deterministic:
        cmd.extend(["--training.deterministic", "true"])
    if overfit_num_samples > 0:
        cmd.extend(["--training.overfit_num_samples", str(overfit_num_samples)])
    if dataset_path.strip():
        cmd.extend(["--training.dataset_path", dataset_path.strip()])
    if audio_codec_backend.strip():
        cmd.extend(["--audio_codec.backend", audio_codec_backend.strip()])
    if audio_codec_source.strip():
        cmd.extend(["--audio_codec.source", audio_codec_source.strip()])
    if audio_codec_model_id.strip():
        cmd.extend(["--audio_codec.model_id", audio_codec_model_id.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio_codec.codec_ckpt_path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.extend(["--audio_codec.trust_remote_code", "true"])
    if checkpoint_interval > 0:
        cmd.extend(["--checkpoint.enable_checkpoint", "true"])
        cmd.extend(["--checkpoint.interval", str(checkpoint_interval)])
        mode = checkpoint_async_mode.strip().lower()
        if mode in {"disabled", "async", "async_with_pinned_mem"}:
            cmd.extend(["--checkpoint.async_mode", mode])
    if checkpoint_keep_latest_k > 0:
        cmd.extend(["--checkpoint.keep_latest_k", str(checkpoint_keep_latest_k)])
    if checkpoint_folder.strip():
        cmd.extend(["--checkpoint.folder", checkpoint_folder.strip()])
    elif checkpoint_interval > 0:
        auto_ckpt = f"checkpoint_{_safe_slug(experiment_id or run_name)}"
        cmd.extend(["--checkpoint.folder", auto_ckpt])
        checkpoint_folder = auto_ckpt
    extra_overrides = _parse_overrides_json(overrides_json)
    if extra_overrides:
        _extend_cmd_with_overrides(cmd, extra_overrides)

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("TOKEN")
    )
    wandb_key = (
        os.environ.get("WANDB_API_KEY")
        or os.environ.get("WANDB_KEY")
        or os.environ.get("WANDB_TOKEN")
        or os.environ.get("WANDB")
    )

    env = {"WANDB_PROJECT": "tinyaya-mimi-tts"}
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
    if wandb_group.strip():
        env["WANDB_RUN_GROUP"] = wandb_group.strip()
    if wandb_tags.strip():
        env["WANDB_TAGS"] = wandb_tags.strip()

    hf_collection_sync: dict[str, object] = {}
    if hf_repo_id.strip():
        hf_collection_sync = _ensure_model_in_hf_collection(
            hf_repo_id=hf_repo_id.strip(),
            hf_repo_private=hf_repo_private,
            hf_collection_slug=hf_collection_slug.strip(),
            hf_token=hf_token,
        )

    uploader_proc = None
    if hf_repo_id.strip():
        resolved_checkpoint_folder = checkpoint_folder.strip() or str(
            run_defaults["checkpoint_folder"]
        )
        ckpt_dir = Path("/vol/outputs") / run_name / resolved_checkpoint_folder
        upload_cmd = [
            "python",
            "scripts/exp/upload_checkpoints_hf.py",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--repo-id",
            hf_repo_id.strip(),
            "--upload-format",
            "hf_pretrained",
            "--model-name",
            str(run_defaults.get("model_id", "CohereLabs/tiny-aya-fire")),
            "--num-quantizers",
            str(resolved_q),
            "--codebook-size",
            str(int(run_defaults.get("codebook_size", 2048))),
            "--export-dtype",
            "float16",
            "--upload-every",
            str(int(hf_upload_every)),
            "--poll-seconds",
            "30",
            "--idle-exit-seconds",
            "900",
        ]
        if hf_repo_private:
            upload_cmd.append("--private")
        uploader_proc = subprocess.Popen(
            upload_cmd,
            cwd=REMOTE_REPO_ROOT,
            env={**os.environ, **env},
        )

    try:
        subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env(env))

        output_run_dir = Path("/vol/outputs") / run_name
        full_eval_root = output_run_dir / "full_eval"
        resolved_checkpoint_folder = checkpoint_folder.strip() or str(
            run_defaults["checkpoint_folder"]
        )
        checkpoint_root = output_run_dir / resolved_checkpoint_folder

        def _latest_step_dir(root: Path, prefix: str) -> Path | None:
            candidates = [path for path in root.glob(f"{prefix}*") if path.is_dir()]
            if not candidates:
                return None
            def _step_num(path: Path) -> int:
                raw = path.name.split(prefix, 1)[-1].lstrip("_-")
                try:
                    return int(raw)
                except ValueError:
                    return -1
            candidates.sort(key=_step_num)
            return candidates[-1]

        latest_full_eval_step = _latest_step_dir(full_eval_root, "step")
        latest_checkpoint_step = _latest_step_dir(checkpoint_root, "step")

        if latest_full_eval_step is not None and latest_checkpoint_step is not None:
            salmon_output = latest_full_eval_step / "salmon.json"
            salmon_cmd = [
                "python",
                "scripts/research/run_salmon_checkpoint.py",
                "--checkpoint-dir",
                str(latest_checkpoint_step),
                "--output-json",
                str(salmon_output),
                "--model-name",
                str(run_defaults.get("model_id", "CohereLabs/tiny-aya-fire")),
                "--num-quantizers",
                str(resolved_q),
                "--codebook-size",
                str(int(run_defaults.get("codebook_size", 2048))),
            ]
            subprocess.run(
                salmon_cmd,
                check=False,
                cwd=REMOTE_REPO_ROOT,
                env=_repo_env(env),
            )

        for post_cmd in (
            [
                "python",
                "scripts/research/score_tts_eval.py",
                "--run-dir",
                str(output_run_dir),
                "--enable-utmos",
                "--progress-every",
                "50",
            ],
            [
                "python",
                "scripts/research/build_scorecard.py",
                "--run-dir",
                str(output_run_dir),
                "--dump-root",
                str(output_run_dir),
            ],
        ):
            subprocess.run(
                post_cmd,
                check=False,
                cwd=REMOTE_REPO_ROOT,
                env=_repo_env(env),
            )
    finally:
        if uploader_proc is not None:
            try:
                uploader_proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                uploader_proc.terminate()
                try:
                    uploader_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    uploader_proc.kill()

    volume.commit()
    return {
        "status": "ok",
        "config": config_file,
        "run_name": run_name,
        "num_quantizers": resolved_q,
        "checkpoint_folder": checkpoint_folder,
        "hf_repo_id": hf_repo_id,
        "hf_collection_slug": hf_collection_slug,
        "hf_collection_sync": hf_collection_sync,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 30,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def infer(
    checkpoint_ref: str,
    text: str,
    lang: str = "en",
    num_quantizers: int = 1,
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "official_fish",
    audio_codec_model_id: str = "",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
    spark_global_tokens: str = "",
    spark_global_tokens_file: str = "",
    spark_prompt_audio: str = "",
):
    logs_dir = Path("/vol/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(logs_dir / "infer_output.wav")
    cmd = [
        "python",
        "inference_tts.py",
        "--model-id",
        checkpoint_ref,
        "--text",
        text,
        "--lang",
        lang,
        "--num-quantizers",
        str(num_quantizers),
        "--output-file",
        output_file,
    ]
    if audio_codec_backend.strip():
        cmd.extend(["--audio-codec-backend", audio_codec_backend.strip()])
    if audio_codec_source.strip():
        cmd.extend(["--audio-codec-source", audio_codec_source.strip()])
    if audio_codec_model_id.strip():
        cmd.extend(["--audio-codec-model-id", audio_codec_model_id.strip()])
    if audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", audio_codec_ckpt_path.strip()])
    if audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    if spark_global_tokens.strip():
        cmd.extend(["--spark-global-tokens", spark_global_tokens.strip()])
    if spark_global_tokens_file.strip():
        cmd.extend(["--spark-global-tokens-file", spark_global_tokens_file.strip()])
    if spark_prompt_audio.strip():
        cmd.extend(["--spark-prompt-audio", spark_prompt_audio.strip()])
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env=_repo_env())
    volume.commit()
    return {"output_file": output_file}


@app.function(
    image=image,
    timeout=60 * 10,
    secrets=HF_SECRETS,
)
def create_hf_collection(
    title: str,
    namespace: str = "rumik-ai",
    description: str = "",
    private: bool = False,
):
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("TOKEN")
    )
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token or None)
    create_fn = getattr(api, "create_collection", None)
    if create_fn is None:
        raise RuntimeError("huggingface_hub does not expose create_collection")

    try:
        coll = create_fn(
            title=title.strip(),
            namespace=namespace.strip(),
            description=description.strip(),
            private=private,
        )
    except TypeError:
        coll = create_fn(title.strip(), namespace.strip(), description.strip(), private)

    slug = ""
    if isinstance(coll, dict):
        slug = str(coll.get("slug") or coll.get("id") or "").strip()
    else:
        slug = str(getattr(coll, "slug", "") or getattr(coll, "id", "")).strip()
    if not slug:
        raise RuntimeError(f"Unable to resolve collection slug from response: {coll!r}")

    print(f"HF_COLLECTION_SLUG={slug}", flush=True)
    return {"slug": slug}


def _launch_parallel_emilia_to_hf(
    *,
    quantizers: int,
    output_dir: str,
    dataset_id: str,
    data_files: str,
    source_split: str,
    max_train_samples: int,
    max_validation_samples: int,
    max_test_samples: int,
    min_seconds: float,
    max_seconds: float,
    seq_len: int,
    reference_seq_len: int,
    export_format: str,
    mask_text_loss: bool,
    language_tokens: bool,
    keep_audio_codes: bool,
    emit_static_references: bool,
    validation_count: int,
    test_count: int,
    split_strategy: str,
    max_samples: int,
    batch_max_clips: int,
    batch_max_audio_seconds: float,
    seed: int,
    shard_size: int,
    audio_codec_backend: str,
    audio_codec_source: str,
    audio_codec_model_id: str,
    audio_codec_ckpt_path: str,
    audio_codec_trust_remote_code: bool,
    repo_name_or_id: str,
    namespace: str,
    private: bool,
    commit_message: str,
    parallel_workers: int,
    max_data_files: int,
    files_per_chunk: int,
    controller_runtime_seconds: int,
    controller_poll_seconds: float,
    controller_handoff_buffer_seconds: int,
    auto_continue: bool,
) -> dict[str, object]:
    output_path = Path(output_dir).expanduser().resolve()
    worker_root = Path(_worker_root_for_output_dir(output_dir)).expanduser().resolve()
    matched_files = _list_matching_dataset_files(dataset_id, data_files)
    if max_data_files > 0:
        matched_files = matched_files[: int(max_data_files)]
    if not matched_files:
        raise RuntimeError(
            f"No dataset files in {dataset_id!r} matched patterns {data_files!r} after filtering."
        )
    state = _load_or_create_parallel_emilia_state(
        output_dir=output_dir,
        dataset_id=dataset_id,
        data_files=data_files,
        source_split=source_split,
        matched_files=matched_files,
        files_per_chunk=files_per_chunk,
        parallel_workers=parallel_workers,
        repo_name_or_id=repo_name_or_id,
    )
    total_chunks = len(state.get("chunks", []) or [])
    print(
        f"[parallel-emilia] dataset_id={dataset_id} matched_files={len(matched_files)} "
        f"parallel_workers={parallel_workers} chunks={total_chunks} "
        f"split_strategy={split_strategy}",
        flush=True,
    )
    controller_started_at = time.monotonic()
    runtime_budget = max(300, int(controller_runtime_seconds))
    handoff_buffer = max(60, int(controller_handoff_buffer_seconds))
    poll_seconds = max(5.0, float(controller_poll_seconds))

    while True:
        volume.reload()
        state = _refresh_parallel_emilia_state(output_dir=output_dir, state=state)
        chunks = [dict(chunk) for chunk in state.get("chunks", [])]
        running_chunks = [chunk for chunk in chunks if str(chunk.get("status", "")) == "running"]
        pending_chunks = [chunk for chunk in chunks if str(chunk.get("status", "")) != "completed"]
        completed_chunks = [chunk for chunk in chunks if str(chunk.get("status", "")) == "completed"]
        print(
            "[parallel-emilia] progress "
            f"completed={len(completed_chunks)}/{len(chunks)} "
            f"running={len(running_chunks)} pending={len(pending_chunks) - len(running_chunks)}",
            flush=True,
        )

        if _all_parallel_chunks_completed(state):
            merge_status = str(((state.get("merge", {}) or {}).get("status", "")) or "pending")
            upload_status = str(((state.get("upload", {}) or {}).get("status", "")) or "pending")
            if merge_status != "success":
                worker_output_dirs = [
                    str(chunk.get("output_dir", "") or "")
                    for chunk in chunks
                    if str(chunk.get("status", "")) == "completed"
                ]
                merge_result = _merge_pretokenized_emilia_workers_impl(
                    worker_output_dirs=worker_output_dirs,
                    output_dir=str(output_path),
                    dataset_id=dataset_id,
                    data_files=data_files,
                    source_split=source_split,
                )
                state["merge"] = {
                    "status": "success",
                    "manifest_path": str(Path(merge_result["manifest_path"]).resolve()),
                }
                _save_parallel_emilia_state(output_dir, state)
                print("[parallel-emilia] merge complete", flush=True)
                print(json.dumps(merge_result, indent=2, ensure_ascii=False), flush=True)
            if upload_status != "success":
                upload_result = _upload_dataset_folder_to_hf_impl(
                    dataset_dir=str(output_path),
                    repo_name_or_id=repo_name_or_id,
                    namespace=namespace,
                    private=private,
                    commit_message=commit_message,
                )
                state["upload"] = {
                    "status": "success",
                    "repo_id": str(upload_result.get("repo_id", "") or ""),
                    "repo_url": str(upload_result.get("repo_url", "") or ""),
                }
                _save_parallel_emilia_state(output_dir, state)
                print("[parallel-emilia] upload complete", flush=True)
                print(json.dumps(upload_result, indent=2, ensure_ascii=False), flush=True)
                return {
                    "status": "ok",
                    "matched_files": len(matched_files),
                    "chunks": len(chunks),
                    "worker_output_root": str(worker_root),
                    "output_dir": str(output_path),
                    "merge": state["merge"],
                    "upload": state["upload"],
                }

        available_slots = max(0, int(parallel_workers) - len(running_chunks))
        if available_slots > 0:
            spawnable_chunks = [
                chunk
                for chunk in chunks
                if str(chunk.get("status", "")) == "pending"
                and not _chunk_manifest_path(str(chunk.get("output_dir", "") or "")).exists()
            ][:available_slots]
            for chunk in spawnable_chunks:
                chunk_idx = int(chunk.get("chunk_idx", -1))
                chunk_output_dir = str(chunk.get("output_dir", "") or "")
                chunk_files = list(chunk.get("data_files", []) or [])
                _remove_chunk_output_dir(chunk_output_dir)
                volume.commit()
                print(
                    f"[parallel-emilia] spawn chunk={chunk_idx} shards={len(chunk_files)} "
                    f"first={chunk_files[0]} last={chunk_files[-1]}",
                    flush=True,
                )
                call = pretokenize_emilia_subset.spawn(
                    quantizers=quantizers,
                    output_dir=chunk_output_dir,
                    dataset_id=dataset_id,
                    data_files=",".join(chunk_files),
                    source_split=source_split,
                    max_train_samples=max_train_samples,
                    max_validation_samples=max_validation_samples,
                    max_test_samples=max_test_samples,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                    seq_len=seq_len,
                    reference_seq_len=reference_seq_len,
                    export_format=export_format,
                    mask_text_loss=mask_text_loss,
                    language_tokens=language_tokens,
                    keep_audio_codes=keep_audio_codes,
                    emit_static_references=emit_static_references,
                    validation_count=validation_count,
                    test_count=test_count,
                    split_strategy=split_strategy,
                    max_samples=max_samples,
                    batch_max_clips=batch_max_clips,
                    batch_max_audio_seconds=batch_max_audio_seconds,
                    seed=seed,
                    shard_size=shard_size,
                    audio_codec_backend=audio_codec_backend,
                    audio_codec_source=audio_codec_source,
                    audio_codec_model_id=audio_codec_model_id,
                    audio_codec_ckpt_path=audio_codec_ckpt_path,
                    audio_codec_trust_remote_code=audio_codec_trust_remote_code,
                    log_prefix=f"chunk-{chunk_idx:05d}",
                )
                for state_chunk in state.get("chunks", []):
                    if int(state_chunk.get("chunk_idx", -1)) == chunk_idx:
                        state_chunk["call_id"] = str(call.object_id)
                        state_chunk["status"] = "running"
                        state_chunk["attempts"] = int(state_chunk.get("attempts", 0) or 0) + 1
                        state_chunk["last_call_status"] = "pending"
                        state_chunk["last_error"] = ""
                        break
            _save_parallel_emilia_state(output_dir, state)

        elapsed = time.monotonic() - controller_started_at
        remaining_budget = runtime_budget - elapsed
        if remaining_budget <= handoff_buffer:
            incomplete_chunks = [
                chunk for chunk in state.get("chunks", []) if str(chunk.get("status", "")) != "completed"
            ]
            if incomplete_chunks and auto_continue:
                continuation_call = pretokenize_emilia_parallel_to_hf.spawn(
                    quantizers=quantizers,
                    output_dir=output_dir,
                    dataset_id=dataset_id,
                    data_files=data_files,
                    source_split=source_split,
                    max_train_samples=max_train_samples,
                    max_validation_samples=max_validation_samples,
                    max_test_samples=max_test_samples,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                    seq_len=seq_len,
                    reference_seq_len=reference_seq_len,
                    export_format=export_format,
                    mask_text_loss=mask_text_loss,
                    language_tokens=language_tokens,
                    keep_audio_codes=keep_audio_codes,
                    emit_static_references=emit_static_references,
                    validation_count=validation_count,
                    test_count=test_count,
                    split_strategy=split_strategy,
                    max_samples=max_samples,
                    batch_max_clips=batch_max_clips,
                    batch_max_audio_seconds=batch_max_audio_seconds,
                    seed=seed,
                    shard_size=shard_size,
                    audio_codec_backend=audio_codec_backend,
                    audio_codec_source=audio_codec_source,
                    audio_codec_model_id=audio_codec_model_id,
                    audio_codec_ckpt_path=audio_codec_ckpt_path,
                    audio_codec_trust_remote_code=audio_codec_trust_remote_code,
                    repo_name_or_id=repo_name_or_id,
                    namespace=namespace,
                    private=private,
                    commit_message=commit_message,
                    parallel_workers=parallel_workers,
                    max_data_files=max_data_files,
                    files_per_chunk=files_per_chunk,
                    controller_runtime_seconds=controller_runtime_seconds,
                    controller_poll_seconds=controller_poll_seconds,
                    controller_handoff_buffer_seconds=controller_handoff_buffer_seconds,
                    auto_continue=auto_continue,
                )
                state["continuation_call_id"] = str(continuation_call.object_id)
                _save_parallel_emilia_state(output_dir, state)
                return {
                    "status": "continued",
                    "matched_files": len(matched_files),
                    "chunks": len(chunks),
                    "worker_output_root": str(worker_root),
                    "output_dir": str(output_path),
                    "continuation_call_id": str(continuation_call.object_id),
                }
            if incomplete_chunks:
                return {
                    "status": "incomplete",
                    "matched_files": len(matched_files),
                    "chunks": len(chunks),
                    "worker_output_root": str(worker_root),
                    "output_dir": str(output_path),
                }

        time.sleep(poll_seconds)


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0, max_delay=60.0),
)
def pretokenize_emilia_parallel_to_hf(
    quantizers: int = 8,
    output_dir: str = "/vol/data/emilia_en_full_mimi_q8_s4096",
    dataset_id: str = "amphion/Emilia-Dataset",
    data_files: str = "Emilia/EN/*.tar",
    source_split: str = "train",
    max_train_samples: int = 0,
    max_validation_samples: int = 0,
    max_test_samples: int = 0,
    min_seconds: float = 1.0,
    max_seconds: float = 30.0,
    seq_len: int = 4096,
    reference_seq_len: int = 1024,
    export_format: str = "codec_only",
    mask_text_loss: bool = True,
    language_tokens: bool = False,
    keep_audio_codes: bool = False,
    emit_static_references: bool = False,
    validation_count: int = 0,
    test_count: int = 0,
    split_strategy: str = "train_only",
    max_samples: int = 0,
    batch_max_clips: int = 16,
    batch_max_audio_seconds: float = 180.0,
    seed: int = 42,
    shard_size: int = 10_000,
    audio_codec_backend: str = "mimi",
    audio_codec_source: str = "hf_pretrained",
    audio_codec_model_id: str = "kyutai/mimi",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
    repo_name_or_id: str = "emilia-en-mimi-q8-s4096",
    namespace: str = "",
    private: bool = True,
    commit_message: str = "",
    parallel_workers: int = 8,
    max_data_files: int = 0,
    files_per_chunk: int = 16,
    controller_runtime_seconds: int = 60 * 60 * 6,
    controller_poll_seconds: float = 60.0,
    controller_handoff_buffer_seconds: int = 60 * 15,
    auto_continue: bool = True,
):
    print(
        "[pretokenize_emilia_parallel_to_hf] starting "
        f"output_dir={output_dir} repo_name_or_id={repo_name_or_id} "
        f"parallel_workers={parallel_workers} files_per_chunk={files_per_chunk} "
        f"data_files={data_files}",
        flush=True,
    )
    result = _launch_parallel_emilia_to_hf(
        quantizers=quantizers,
        output_dir=output_dir,
        dataset_id=dataset_id,
        data_files=data_files,
        source_split=source_split,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        seq_len=seq_len,
        reference_seq_len=reference_seq_len,
        export_format=export_format,
        mask_text_loss=mask_text_loss,
        language_tokens=language_tokens,
        keep_audio_codes=keep_audio_codes,
        emit_static_references=emit_static_references,
        validation_count=validation_count,
        test_count=test_count,
        split_strategy=split_strategy,
        max_samples=max_samples,
        batch_max_clips=batch_max_clips,
        batch_max_audio_seconds=batch_max_audio_seconds,
        seed=seed,
        shard_size=shard_size,
        audio_codec_backend=audio_codec_backend,
        audio_codec_source=audio_codec_source,
        audio_codec_model_id=audio_codec_model_id,
        audio_codec_ckpt_path=audio_codec_ckpt_path,
        audio_codec_trust_remote_code=audio_codec_trust_remote_code,
        repo_name_or_id=repo_name_or_id,
        namespace=namespace,
        private=private,
        commit_message=commit_message,
        parallel_workers=parallel_workers,
        max_data_files=max_data_files,
        files_per_chunk=files_per_chunk,
        controller_runtime_seconds=controller_runtime_seconds,
        controller_poll_seconds=controller_poll_seconds,
        controller_handoff_buffer_seconds=controller_handoff_buffer_seconds,
        auto_continue=auto_continue,
    )
    print(
        "[pretokenize_emilia_parallel_to_hf] complete "
        f"output_dir={result.get('output_dir')} repo_id={((result.get('upload') or {}) if isinstance(result, dict) else {}).get('repo_id', '')}",
        flush=True,
    )
    return result


@app.local_entrypoint()
def main(
    mode: str = "train",
    path: str = "overfit1",
    split: str = "train",
    languages: str = "en hi te es fr de ar sw ta bn zh",
    quantizers: int = 1,
    experiment_id: str = "",
    campaign_id: str = "",
    phase: str = "",
    stage: str = "",
    variant: str = "",
    steps: int = 0,
    track: str = "",
    axis: str = "",
    family: str = "",
    question: str = "",
    hypothesis: str = "",
    brief_path: str = "",
    baseline_experiment_id: str = "",
    owner: str = "",
    tags: str = "",
    input_wav_path: str = "/vol/data/raw/download.wav",
    text: str = "",
    lang: str = "en",
    sample_id: str = "download_001",
    max_seconds: float = 30.0,
    output_dir: str = "",
    checkpoint_ref: str = "",
    audio_codec_backend: str = "",
    audio_codec_source: str = "",
    audio_codec_model_id: str = "",
    audio_codec_ckpt_path: str = "",
    audio_codec_trust_remote_code: bool = False,
    spark_global_tokens: str = "",
    spark_global_tokens_file: str = "",
    spark_prompt_audio: str = "",
    download_local_dir: str = "",
    dataset_id: str = "amphion/Emilia-Dataset",
    data_files: str = "Emilia/EN/*.tar",
    source_split: str = "train",
    max_train_samples: int = 0,
    max_validation_samples: int = 0,
    max_test_samples: int = 0,
    min_seconds: float = 1.0,
    seq_len: int = 4096,
    reference_seq_len: int = 1024,
    export_format: str = "codec_only",
    mask_text_loss: bool = True,
    language_tokens: bool = False,
    keep_audio_codes: bool = False,
    emit_static_references: bool = False,
    validation_count: int = 0,
    test_count: int = 0,
    split_strategy: str = "train_only",
    parallel_workers: int = 8,
    max_data_files: int = 0,
    files_per_chunk: int = 16,
    max_samples: int = 0,
    batch_max_clips: int = 16,
    batch_max_audio_seconds: float = 180.0,
    shard_size: int = 10_000,
    controller_runtime_seconds: int = 60 * 60 * 6,
    controller_poll_seconds: float = 60.0,
    controller_handoff_buffer_seconds: int = 60 * 15,
    auto_continue: bool = True,
    seed: int = -1,
    deterministic: bool = False,
    dataset_path: str = "",
    checkpoint_interval: int = 0,
    checkpoint_keep_latest_k: int = 0,
    checkpoint_folder: str = "",
    checkpoint_async_mode: str = "async",
    wandb_group: str = "",
    wandb_tags: str = "",
    overrides_json: str = "",
    hf_dataset_repo: str = "emilia-en-mimi-q8-s4096",
    hf_namespace: str = "",
    hf_private: bool = True,
    hf_commit_message: str = "",
):
    if mode == "pretokenize":
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "official_fish"
        print(
            pretokenize_fleurs.remote(
                split=split,
                languages=languages,
                quantizers=quantizers,
                output_dir=output_dir,
                audio_codec_backend=resolved_codec_backend,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=audio_codec_model_id,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
            )
        )
        return
    if mode == "pretokenize_emilia":
        resolved_output_dir = (
            output_dir.strip() or f"/vol/data/emilia_en_full_mimi_q{quantizers}_s{seq_len}"
        )
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "hf_pretrained"
        resolved_codec_model = audio_codec_model_id.strip() or "kyutai/mimi"
        print(
            pretokenize_emilia_subset.remote(
                quantizers=quantizers,
                output_dir=resolved_output_dir,
                dataset_id=dataset_id,
                data_files=data_files,
                source_split=source_split,
                max_train_samples=max_train_samples,
                max_validation_samples=max_validation_samples,
                max_test_samples=max_test_samples,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                seq_len=seq_len,
                reference_seq_len=reference_seq_len,
                export_format=export_format,
                mask_text_loss=mask_text_loss,
                language_tokens=language_tokens,
                keep_audio_codes=keep_audio_codes,
                emit_static_references=emit_static_references,
                validation_count=validation_count,
                test_count=test_count,
                split_strategy=split_strategy,
                max_samples=max_samples,
                batch_max_clips=batch_max_clips,
                batch_max_audio_seconds=batch_max_audio_seconds,
                seed=seed if seed >= 0 else 42,
                shard_size=shard_size,
                audio_codec_backend=resolved_codec_backend,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=resolved_codec_model,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
            )
        )
        return
    if mode == "upload_dataset_hf":
        resolved_output_dir = (
            output_dir.strip() or f"/vol/data/emilia_en_full_mimi_q{quantizers}_s{seq_len}"
        )
        print(
            upload_dataset_folder_to_hf.remote(
                dataset_dir=resolved_output_dir,
                repo_name_or_id=hf_dataset_repo,
                namespace=hf_namespace,
                private=hf_private,
                commit_message=hf_commit_message,
            )
        )
        return
    if mode == "pretokenize_emilia_to_hf":
        resolved_output_dir = (
            output_dir.strip() or f"/vol/data/emilia_en_full_mimi_q{quantizers}_s{seq_len}"
        )
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "hf_pretrained"
        resolved_codec_model = audio_codec_model_id.strip() or "kyutai/mimi"
        print(
            pretokenize_emilia_subset_to_hf.remote(
                quantizers=quantizers,
                output_dir=resolved_output_dir,
                dataset_id=dataset_id,
                data_files=data_files,
                source_split=source_split,
                max_train_samples=max_train_samples,
                max_validation_samples=max_validation_samples,
                max_test_samples=max_test_samples,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                seq_len=seq_len,
                reference_seq_len=reference_seq_len,
                export_format=export_format,
                mask_text_loss=mask_text_loss,
                language_tokens=language_tokens,
                keep_audio_codes=keep_audio_codes,
                emit_static_references=emit_static_references,
                validation_count=validation_count,
                test_count=test_count,
                split_strategy=split_strategy,
                max_samples=max_samples,
                batch_max_clips=batch_max_clips,
                batch_max_audio_seconds=batch_max_audio_seconds,
                seed=seed if seed >= 0 else 42,
                shard_size=shard_size,
                audio_codec_backend=resolved_codec_backend,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=resolved_codec_model,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
                repo_name_or_id=hf_dataset_repo,
                namespace=hf_namespace,
                private=hf_private,
                commit_message=hf_commit_message,
            )
        )
        return
    if mode == "pretokenize_emilia_parallel_to_hf":
        resolved_output_dir = (
            output_dir.strip() or f"/vol/data/emilia_en_full_mimi_q{quantizers}_s{seq_len}"
        )
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "hf_pretrained"
        resolved_codec_model = audio_codec_model_id.strip() or "kyutai/mimi"
        result = pretokenize_emilia_parallel_to_hf.remote(
            quantizers=quantizers,
            output_dir=resolved_output_dir,
            dataset_id=dataset_id,
            data_files=data_files,
            source_split=source_split,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            max_test_samples=max_test_samples,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            seq_len=seq_len,
            reference_seq_len=reference_seq_len,
            export_format=export_format,
            mask_text_loss=mask_text_loss,
            language_tokens=language_tokens,
            keep_audio_codes=keep_audio_codes,
            emit_static_references=emit_static_references,
            validation_count=validation_count,
            test_count=test_count,
            split_strategy=split_strategy,
            max_samples=max_samples,
            batch_max_clips=batch_max_clips,
            batch_max_audio_seconds=batch_max_audio_seconds,
            seed=seed if seed >= 0 else 42,
            shard_size=shard_size,
            audio_codec_backend=resolved_codec_backend,
            audio_codec_source=resolved_codec_source,
            audio_codec_model_id=resolved_codec_model,
            audio_codec_ckpt_path=audio_codec_ckpt_path,
            audio_codec_trust_remote_code=audio_codec_trust_remote_code,
            repo_name_or_id=hf_dataset_repo,
            namespace=hf_namespace,
            private=hf_private,
            commit_message=hf_commit_message,
            parallel_workers=parallel_workers,
            max_data_files=max_data_files,
            files_per_chunk=files_per_chunk,
            controller_runtime_seconds=controller_runtime_seconds,
            controller_poll_seconds=controller_poll_seconds,
            controller_handoff_buffer_seconds=controller_handoff_buffer_seconds,
            auto_continue=auto_continue,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if mode == "pretokenize_single":
        resolved_output_dir = output_dir.strip() or f"/vol/data/custom_download_q{quantizers}"
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "hf_pretrained"
        print(
            pretokenize_single_wav.remote(
                input_wav_path=input_wav_path,
                text=text,
                lang=lang,
                sample_id=sample_id,
                quantizers=quantizers,
                max_seconds=max_seconds,
                output_dir=resolved_output_dir,
                audio_codec_backend=resolved_codec_backend,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=audio_codec_model_id,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
            )
        )
        return
    if mode == "reconstruct_single":
        input_path = Path(input_wav_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input WAV not found: {input_path}")
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "hf_pretrained"
        resolved_codec_model_id = audio_codec_model_id.strip() or "kyutai/mimi"
        resolved_output_dir = output_dir.strip() or (
            f"/vol/outputs/codec_recon/{_safe_slug(input_path.stem)}"
            f"-{resolved_codec_backend}-q{int(quantizers)}"
        )
        result = codec_reconstruction_report.remote(
            input_wav_bytes=input_path.read_bytes(),
            input_filename=input_path.name,
            text=text,
            lang=lang,
            sample_id=sample_id,
            quantizers=quantizers,
            max_seconds=max_seconds,
            output_dir=resolved_output_dir,
            audio_codec_backend=resolved_codec_backend,
            audio_codec_source=resolved_codec_source,
            audio_codec_model_id=resolved_codec_model_id,
            audio_codec_ckpt_path=audio_codec_ckpt_path,
            audio_codec_trust_remote_code=audio_codec_trust_remote_code,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if download_local_dir.strip():
            local_dir = Path(download_local_dir).expanduser().resolve()
            local_dir.mkdir(parents=True, exist_ok=True)
            remote_volume_path = resolved_output_dir
            if remote_volume_path.startswith("/vol/"):
                remote_volume_path = remote_volume_path[len("/vol") :]
            elif remote_volume_path == "/vol":
                remote_volume_path = "/"
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    DATA_VOL_NAME,
                    remote_volume_path,
                    str(local_dir),
                ],
                check=True,
            )
            print(
                json.dumps(
                    {
                        "downloaded_to": str(local_dir),
                        "remote_output_dir": resolved_output_dir,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return
    if mode == "pretokenize_s1":
        resolved_quantizers = quantizers if int(quantizers) > 1 else 9
        resolved_output_dir = output_dir.strip() or f"/vol/data/custom_download_s1_q{resolved_quantizers}"
        resolved_codec_source = audio_codec_source.strip() or "official_fish"
        resolved_codec_model = audio_codec_model_id.strip() or "jordand/fish-s1-dac-min"
        print(
            pretokenize_single_wav_s1.remote(
                input_wav_path=input_wav_path,
                text=text,
                lang=lang,
                sample_id=sample_id,
                quantizers=resolved_quantizers,
                max_seconds=max_seconds,
                output_dir=resolved_output_dir,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=resolved_codec_model,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
            )
        )
        return
    if mode == "pretokenize_fleurs_s1":
        resolved_quantizers = quantizers if int(quantizers) > 1 else 9
        resolved_output_dir = output_dir.strip() or f"/vol/data/fleurs_pretok_s1_q{resolved_quantizers}"
        resolved_codec_source = audio_codec_source.strip() or "official_fish"
        resolved_codec_model = audio_codec_model_id.strip() or "jordand/fish-s1-dac-min"
        print(
            pretokenize_fleurs_s1.remote(
                split=split,
                languages=languages,
                quantizers=resolved_quantizers,
                output_dir=resolved_output_dir,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=resolved_codec_model,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
            )
        )
        return
    if mode == "train":
        print(
            train.remote(
                path=path,
                experiment_id=experiment_id,
                campaign_id=campaign_id,
                phase=phase,
                stage=stage,
                variant=variant,
                steps=steps,
                track=track,
                axis=axis,
                family=family,
                question=question,
                hypothesis=hypothesis,
                brief_path=brief_path,
                baseline_experiment_id=baseline_experiment_id,
                owner=owner,
                tags=tags,
                seed=seed,
                deterministic=deterministic,
                dataset_path=dataset_path,
                checkpoint_interval=checkpoint_interval,
                checkpoint_keep_latest_k=checkpoint_keep_latest_k,
                checkpoint_folder=checkpoint_folder,
                checkpoint_async_mode=checkpoint_async_mode,
                audio_codec_backend=audio_codec_backend,
                audio_codec_source=audio_codec_source,
                audio_codec_model_id=audio_codec_model_id,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
                wandb_group=wandb_group,
                wandb_tags=wandb_tags,
                overrides_json=overrides_json,
            )
        )
        return
    if mode == "infer":
        resolved_checkpoint_ref = checkpoint_ref.strip()
        if not resolved_checkpoint_ref:
            raise ValueError("infer mode requires checkpoint_ref.")
        resolved_codec_backend = audio_codec_backend.strip() or "mimi"
        resolved_codec_source = audio_codec_source.strip() or "official_fish"
        print(
            infer.remote(
                checkpoint_ref=resolved_checkpoint_ref,
                text=text,
                lang=lang,
                num_quantizers=quantizers,
                audio_codec_backend=resolved_codec_backend,
                audio_codec_source=resolved_codec_source,
                audio_codec_model_id=audio_codec_model_id,
                audio_codec_ckpt_path=audio_codec_ckpt_path,
                audio_codec_trust_remote_code=audio_codec_trust_remote_code,
                spark_global_tokens=spark_global_tokens,
                spark_global_tokens_file=spark_global_tokens_file,
                spark_prompt_audio=spark_prompt_audio,
            )
        )
        return
    raise ValueError(
        "mode must be one of: pretokenize, pretokenize_emilia, upload_dataset_hf, "
        "pretokenize_emilia_to_hf, pretokenize_emilia_parallel_to_hf, "
        "pretokenize_single, reconstruct_single, "
        "pretokenize_s1, pretokenize_fleurs_s1, train, infer"
    )
