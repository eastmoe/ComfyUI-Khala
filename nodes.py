from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import folder_paths
except ImportError as exc:
    raise ImportError("Comfy-Khala must be imported from a ComfyUI runtime.") from exc


ROOT = Path(__file__).resolve().parent
KHALA_SOURCE_ROOT = ROOT / "Khala"
KHALA_INFERENCE = KHALA_SOURCE_ROOT / "inference.py"
CATEGORY = "eastmoe/Comfy-Khala"

GENRE_OPTIONS = [
    "Pop",
    "Rock",
    "R&B",
    "Hip-Hop",
    "Electronic",
    "Jazz",
    "Classical",
    "Folk",
    "Country",
    "Metal",
    "Latin",
    "Reggae",
    "Blues",
    "Funk",
    "Soul",
    "Indie",
    "Alternative",
    "Dance",
    "Acoustic",
]

LANGUAGE_OPTIONS = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "Cantonese",
    "Instrumental",
]

SUPERRES_TEXT_MODES = [
    "same_as_backbone",
    "same_as_backbone_no_description",
    "separate",
]

ATTENTION_BACKENDS = ["sdpa", "auto", "flash", "fused", "unfused", "local"]
DTYPES = ["bf16", "fp16", "fp32"]
KHALA_HF_REPO_ID = "liujiafeng/Khala-MusicGeneration-v1.0"
KHALA_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
KHALA_HF_REQUIRED_PATTERNS = [
    "backbone/*",
    "backbone/**/*",
    "superresolution/*",
    "superresolution/**/*",
    "dac_rvq_2490000.ckpt",
]


@dataclass
class KhalaModelConfig:
    model_dir: str
    tokenizer_path: str
    backbone_name: str
    backbone_path: str
    backbone_vocab_size: int
    superres_name: str
    superres_path: str
    superres_vocab_size: int
    decoder_config_path: str
    decoder_checkpoint_path: str
    decoder_sample_rate: int
    decoder_chunk_size: int
    decoder_chunk_overlap: int
    codec_fps: float
    output_dir: str
    output_format: str
    mp3_bitrate: str
    ffmpeg_bin: str
    python_executable: str
    gpu_id: str
    master_addr: str
    master_port: str
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    tokenizer_type: str
    norm_epsilon: float
    inference_max_seq_length: int
    inference_max_requests: int
    inference_batch_times_seqlen_threshold: int
    transformer_impl: str
    attention_backend: str
    dtype: str
    stream: bool
    enable_cuda_graph: bool
    flash_decode: bool
    tokens_per_minute: int
    backbone_max_prompt_len: int
    superres_max_prompt_len: int


def khala_models_root() -> Path:
    root = Path(folder_paths.models_dir) / "Khala"
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_folder_options() -> list[str]:
    root = khala_models_root()
    options = ["."]
    options.extend(
        child.name
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir()
    )
    return options


def default_output_dir() -> str:
    return str(Path(folder_paths.get_output_directory()) / "Khala")


def safe_child_path(base: Path, value: str, *, allow_empty: bool = False) -> Path | None:
    value = (value or "").strip()
    if not value:
        return None if allow_empty else base
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Khala model paths must be relative to ComfyUI/models/Khala.")
    resolved = (base / path).resolve()
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"Path escapes ComfyUI/models/Khala: {value}")
    return resolved


def first_existing(base: Path, requested: str, alternatives: list[str], fallback: Path | None = None) -> Path:
    requested_path = safe_child_path(base, requested)
    if requested_path.exists():
        return requested_path
    for item in alternatives:
        candidate = safe_child_path(base, item)
        if candidate.exists():
            return candidate
    if fallback is not None and fallback.exists():
        return fallback
    return requested_path


def require_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def split_patterns(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,]+", value or "") if item.strip()]


def summarize_download(path: Path) -> dict[str, Any]:
    files = [
        item
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    ]
    total_size = sum(item.stat().st_size for item in files)

    def group_summary(relative_prefix: str) -> dict[str, Any]:
        group = [item for item in files if item.relative_to(path).as_posix().startswith(relative_prefix)]
        return {
            "files": len(group),
            "distcp_files": sum(1 for item in group if item.suffix == ".distcp"),
            "size_bytes": sum(item.stat().st_size for item in group),
        }

    return {
        "path": str(path),
        "files": len(files),
        "size_bytes": total_size,
        "size_gib": round(total_size / (1024**3), 3),
        "backbone": group_summary("backbone/"),
        "superresolution": group_summary("superresolution/"),
        "decoder_checkpoint": {
            "exists": (path / "dac_rvq_2490000.ckpt").is_file(),
            "size_bytes": (path / "dac_rvq_2490000.ckpt").stat().st_size
            if (path / "dac_rvq_2490000.ckpt").is_file()
            else 0,
        },
    }


def configure_hf_tls(verify_tls: bool) -> None:
    try:
        if not verify_tls:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            import httpx
            import huggingface_hub.utils._http as hf_http
            from huggingface_hub import set_async_client_factory, set_client_factory

            def client_factory() -> httpx.Client:
                return httpx.Client(
                    event_hooks={"request": [hf_http.hf_request_event_hook]},
                    follow_redirects=True,
                    timeout=None,
                    verify=bool(verify_tls),
                )

            def async_client_factory() -> httpx.AsyncClient:
                return httpx.AsyncClient(
                    event_hooks={
                        "request": [hf_http.async_hf_request_event_hook],
                        "response": [hf_http.async_hf_response_event_hook],
                    },
                    follow_redirects=True,
                    timeout=None,
                    verify=bool(verify_tls),
                )

            set_client_factory(client_factory)
            set_async_client_factory(async_client_factory)
            return
        except ImportError:
            pass

        import requests
        from huggingface_hub import configure_http_backend

        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = bool(verify_tls)
            return session

        configure_http_backend(backend_factory=backend_factory)
    except Exception as exc:
        raise RuntimeError(f"Failed to disable Hugging Face TLS verification: {exc}") from exc


def append_value(argv: list[str], flag: str, value: Any) -> None:
    if value is not None and str(value) != "":
        argv.extend([flag, str(value)])


def append_positive_int(argv: list[str], flag: str, value: int) -> None:
    if int(value) > 0:
        argv.extend([flag, str(int(value))])


def normalize_request_sampling(
    sampling_method: str,
    top_k_bb: int,
    top_p_bb: float,
) -> tuple[int, float]:
    if sampling_method == "greedy":
        return 1, 0.0
    if sampling_method == "top_p":
        return 0, float(top_p_bb or 0.95)
    if int(top_k_bb) > 0 and float(top_p_bb) > 0.0:
        raise ValueError("Backbone top-k and top-p are mutually exclusive; choose top_k or top_p.")
    return int(top_k_bb), 0.0


def wav_to_comfy_audio(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = audio.reshape(-1, channels).T
    waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def load_audio_result(wav_path: Path, mp3_path: Path) -> dict[str, Any]:
    if wav_path.is_file():
        return wav_to_comfy_audio(wav_path)
    if mp3_path.is_file():
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(str(mp3_path))
            return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
        except Exception as exc:
            raise RuntimeError(
                "Only MP3 was generated and torchaudio could not load it. "
                "Use output_format='wav' or 'both' for direct AUDIO output."
            ) from exc
    raise FileNotFoundError("Khala finished without producing a WAV or MP3 file.")


def build_command(
    config: KhalaModelConfig,
    request_json: Path,
    result_json: Path,
    seed: int,
    auto_release_vram: bool,
    num_tokens_to_generate: int,
    extra_megatron_args: str,
) -> list[str]:
    python_executable = config.python_executable.strip() or sys.executable
    runtime_mode = "one_shot" if auto_release_vram else "keep_loaded"
    argv = [
        python_executable,
        str(KHALA_INFERENCE),
        "--request-json",
        str(request_json),
        "--result-json",
        str(result_json),
        "--output-dir",
        config.output_dir,
        "--output-format",
        config.output_format,
        "--mp3-bitrate",
        config.mp3_bitrate,
        "--ffmpeg-bin",
        config.ffmpeg_bin,
        "--gpus",
        str(config.gpu_id),
        "--seed",
        str(int(seed)),
        "--runtime-mode",
        runtime_mode,
        "--python",
        python_executable,
        "--master-addr",
        config.master_addr,
        "--master-port",
        config.master_port,
        "--tokenizer-path",
        config.tokenizer_path,
        "--backbone-name",
        config.backbone_name,
        "--backbone-path",
        config.backbone_path,
        "--backbone-vocab-size",
        str(config.backbone_vocab_size),
        "--superres-name",
        config.superres_name,
        "--superres-path",
        config.superres_path,
        "--superres-vocab-size",
        str(config.superres_vocab_size),
        "--decoder-config-path",
        config.decoder_config_path,
        "--decoder-checkpoint-path",
        config.decoder_checkpoint_path,
        "--decoder-sample-rate",
        str(config.decoder_sample_rate),
        "--decoder-chunk-size",
        str(config.decoder_chunk_size),
        "--decoder-chunk-overlap",
        str(config.decoder_chunk_overlap),
        "--codec-fps",
        str(config.codec_fps),
        "--tokens-per-minute",
        str(config.tokens_per_minute),
        "--backbone-max-prompt-len",
        str(config.backbone_max_prompt_len),
        "--superres-max-prompt-len",
        str(config.superres_max_prompt_len),
        "--tensor-model-parallel-size",
        str(config.tensor_model_parallel_size),
        "--pipeline-model-parallel-size",
        str(config.pipeline_model_parallel_size),
        "--tokenizer-type",
        config.tokenizer_type,
        "--norm-epsilon",
        str(config.norm_epsilon),
        "--inference-max-seq-length",
        str(config.inference_max_seq_length),
        "--num-tokens-to-generate",
        str(int(num_tokens_to_generate)),
        "--transformer-impl",
        config.transformer_impl,
        "--attention-backend",
        config.attention_backend,
    ]
    append_positive_int(argv, "--inference-max-requests", config.inference_max_requests)
    append_positive_int(
        argv,
        "--inference-batch-times-seqlen-threshold",
        config.inference_batch_times_seqlen_threshold,
    )
    argv.append("--stream" if config.stream else "--no-stream")
    if config.enable_cuda_graph:
        argv.append("--enable-cuda-graph")
    if config.flash_decode:
        argv.append("--flash-decode")
    argv.append(f"--{config.dtype}")
    if extra_megatron_args.strip():
        argv.append("--")
        argv.extend(shlex.split(extra_megatron_args, posix=False))
    return argv


class KhalaModelDownloader:
    DESCRIPTION = (
        "Downloads the official Khala Hugging Face checkpoint into ComfyUI/models/Khala. "
        "Supports hf-mirror.com, disabled TLS verification, and disabled Hugging Face Xet."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": KHALA_HF_REPO_ID}),
                "revision": ("STRING", {"default": "main"}),
                "model_folder": ("STRING", {"default": "Khala-MusicGeneration-v1.0"}),
                "download_scope": (
                    ["required_weights_only", "full_repository", "custom_patterns"],
                    {"default": "required_weights_only"},
                ),
                "use_hf_mirror": ("BOOLEAN", {"default": True}),
                "mirror_endpoint": ("STRING", {"default": KHALA_HF_MIRROR_ENDPOINT}),
                "verify_tls": ("BOOLEAN", {"default": True}),
                "disable_hf_xet": ("BOOLEAN", {"default": True}),
                "force_download": ("BOOLEAN", {"default": False}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "custom_allow_patterns": (
                    "STRING",
                    {
                        "default": "\n".join(KHALA_HF_REQUIRED_PATTERNS),
                        "multiline": True,
                    },
                ),
                "ignore_patterns": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_folder", "download_path", "summary_json")
    OUTPUT_TOOLTIPS = (
        "Folder name relative to ComfyUI/models/Khala. Use this in Khala Model Loader.",
        "Absolute path where the model was downloaded.",
        "Download summary JSON with file counts and sizes.",
    )
    FUNCTION = "download"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def download(
        self,
        repo_id: str,
        revision: str,
        model_folder: str,
        download_scope: str,
        use_hf_mirror: bool,
        mirror_endpoint: str,
        verify_tls: bool,
        disable_hf_xet: bool,
        force_download: bool,
        max_workers: int,
        custom_allow_patterns: str,
        ignore_patterns: str,
    ):
        target = safe_child_path(khala_models_root(), model_folder)
        assert target is not None
        target.mkdir(parents=True, exist_ok=True)

        if disable_hf_xet:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        else:
            os.environ.pop("HF_HUB_DISABLE_XET", None)

        endpoint = mirror_endpoint.strip() if use_hf_mirror else "https://huggingface.co"
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Khala model downloading requires huggingface_hub. "
                f"Install it with: {sys.executable} -m pip install huggingface_hub"
            ) from exc

        try:
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HUB_DISABLE_XET = bool(disable_hf_xet)
        except Exception:
            pass

        configure_hf_tls(bool(verify_tls))

        if download_scope == "required_weights_only":
            allow_patterns = KHALA_HF_REQUIRED_PATTERNS
        elif download_scope == "full_repository":
            allow_patterns = None
        else:
            allow_patterns = split_patterns(custom_allow_patterns)
            if not allow_patterns:
                raise ValueError("custom_patterns requires at least one allow pattern.")

        ignore = split_patterns(ignore_patterns) or None
        print(
            "[Comfy-Khala] Downloading "
            f"{repo_id}@{revision} to {target} via {endpoint or 'default Hugging Face endpoint'} "
            f"(scope={download_scope}, verify_tls={verify_tls}, disable_hf_xet={disable_hf_xet})"
        )
        download_kwargs = {
            "repo_id": repo_id.strip() or KHALA_HF_REPO_ID,
            "repo_type": "model",
            "revision": revision.strip() or "main",
            "local_dir": str(target),
            "endpoint": endpoint or None,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore,
            "force_download": bool(force_download),
            "max_workers": int(max_workers),
        }
        try:
            snapshot_download(**download_kwargs)
        except OSError as exc:
            if "Consistency check failed" not in str(exc) or bool(force_download):
                raise
            print("[Comfy-Khala] Consistency check failed; retrying once with force_download=True.")
            download_kwargs["force_download"] = True
            snapshot_download(**download_kwargs)

        summary = summarize_download(target)
        summary.update(
            {
                "repo_id": repo_id.strip() or KHALA_HF_REPO_ID,
                "revision": revision.strip() or "main",
                "endpoint": endpoint or "https://huggingface.co",
                "download_scope": download_scope,
                "allow_patterns": allow_patterns or ["<all>"],
                "ignore_patterns": ignore or [],
                "verify_tls": bool(verify_tls),
                "disable_hf_xet": bool(disable_hf_xet),
                "max_workers": int(max_workers),
            }
        )
        return model_folder.strip() or ".", str(target), json.dumps(summary, ensure_ascii=False, indent=2)


class KhalaModelLoader:
    DESCRIPTION = (
        "Builds a Khala model configuration rooted at ComfyUI/models/Khala. "
        "The selected folder should contain checkpoints/backbone, "
        "checkpoints/superresolution, and the DAC RVQ checkpoint."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_folder": (model_folder_options(), {"default": "."}),
                "backbone_name": ("STRING", {"default": "q01_354k_tag_desc_v0 (0217)"}),
                "backbone_subdir": ("STRING", {"default": "checkpoints/backbone"}),
                "backbone_vocab_size": ("INT", {"default": 130304, "min": 1, "max": 1000000, "step": 1}),
                "superres_name": ("STRING", {"default": "q01_ft5k_super_v2 (0215)"}),
                "superres_subdir": ("STRING", {"default": "checkpoints/superresolution"}),
                "superres_vocab_size": ("INT", {"default": 193792, "min": 1, "max": 1000000, "step": 1}),
                "tokenizer_subdir": ("STRING", {"default": "models/Tokenizer"}),
                "decoder_config_path": ("STRING", {"default": "models/Decoder/dac_rvq_1024_64_golden.yaml"}),
                "decoder_checkpoint_path": ("STRING", {"default": "checkpoints/dac_rvq_2490000.ckpt"}),
                "decoder_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 384000, "step": 1}),
                "decoder_chunk_size": ("INT", {"default": 1920, "min": 1, "max": 262144, "step": 1}),
                "decoder_chunk_overlap": ("INT", {"default": 480, "min": 0, "max": 262144, "step": 1}),
                "codec_fps": ("FLOAT", {"default": 21.5, "min": 1.0, "max": 240.0, "step": 0.01}),
                "output_dir": ("STRING", {"default": default_output_dir()}),
                "output_format": (["both", "wav", "mp3"], {"default": "both"}),
                "mp3_bitrate": ("STRING", {"default": "320k"}),
                "ffmpeg_bin": ("STRING", {"default": "ffmpeg"}),
                "python_executable": ("STRING", {"default": sys.executable}),
                "gpu_id": ("STRING", {"default": "0"}),
                "master_addr": ("STRING", {"default": "127.0.0.1"}),
                "master_port": ("STRING", {"default": os.environ.get("MASTER_PORT", "8791")}),
                "tensor_model_parallel_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "pipeline_model_parallel_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "tokenizer_type": ("STRING", {"default": "NullTokenizer"}),
                "norm_epsilon": ("FLOAT", {"default": 1e-6, "min": 1e-9, "max": 1e-3, "step": 1e-7}),
                "inference_max_seq_length": ("INT", {"default": 25600, "min": 1, "max": 1048576, "step": 1}),
                "inference_max_requests": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "inference_batch_times_seqlen_threshold": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2147483647, "step": 1},
                ),
                "transformer_impl": (["local", "transformer_engine"], {"default": "local"}),
                "attention_backend": (ATTENTION_BACKENDS, {"default": "sdpa"}),
                "dtype": (DTYPES, {"default": "bf16"}),
                "stream": ("BOOLEAN", {"default": True}),
                "enable_cuda_graph": ("BOOLEAN", {"default": False}),
                "flash_decode": ("BOOLEAN", {"default": False}),
                "tokens_per_minute": ("INT", {"default": 2584, "min": 1, "max": 1000000, "step": 1}),
                "backbone_max_prompt_len": ("INT", {"default": 4096, "min": 1, "max": 1048576, "step": 1}),
                "superres_max_prompt_len": ("INT", {"default": 2048, "min": 1, "max": 1048576, "step": 1}),
            }
        }

    RETURN_TYPES = ("KHALA_MODEL", "STRING")
    RETURN_NAMES = ("khala_model", "model_info")
    OUTPUT_TOOLTIPS = (
        "Resolved Khala model/runtime configuration for the inference node.",
        "Human-readable JSON summary of the resolved paths and runtime choices.",
    )
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    def load_model(self, **kwargs):
        base = safe_child_path(khala_models_root(), kwargs["model_folder"])
        assert base is not None

        tokenizer_path = first_existing(
            base,
            kwargs["tokenizer_subdir"],
            ["Tokenizer"],
            KHALA_SOURCE_ROOT / "models" / "Tokenizer",
        )
        backbone_path = first_existing(base, kwargs["backbone_subdir"], ["backbone"])
        superres_path = first_existing(base, kwargs["superres_subdir"], ["superresolution"])
        decoder_config_path = first_existing(
            base,
            kwargs["decoder_config_path"],
            ["Decoder/dac_rvq_1024_64_golden.yaml"],
            KHALA_SOURCE_ROOT / "models" / "Decoder" / "dac_rvq_1024_64_golden.yaml",
        )
        decoder_checkpoint_path = first_existing(
            base,
            kwargs["decoder_checkpoint_path"],
            ["dac_rvq_2490000.ckpt"],
        )

        require_exists(str(backbone_path), "Backbone checkpoint directory")
        require_exists(str(superres_path), "Super-resolution checkpoint directory")
        require_exists(str(decoder_checkpoint_path), "Decoder checkpoint")
        require_exists(str(tokenizer_path), "Tokenizer directory")
        require_exists(str(decoder_config_path), "Decoder config")

        output_dir = kwargs["output_dir"].strip() or default_output_dir()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config = KhalaModelConfig(
            model_dir=str(base),
            tokenizer_path=str(tokenizer_path),
            backbone_name=kwargs["backbone_name"],
            backbone_path=str(backbone_path),
            backbone_vocab_size=int(kwargs["backbone_vocab_size"]),
            superres_name=kwargs["superres_name"],
            superres_path=str(superres_path),
            superres_vocab_size=int(kwargs["superres_vocab_size"]),
            decoder_config_path=str(decoder_config_path),
            decoder_checkpoint_path=str(decoder_checkpoint_path),
            decoder_sample_rate=int(kwargs["decoder_sample_rate"]),
            decoder_chunk_size=int(kwargs["decoder_chunk_size"]),
            decoder_chunk_overlap=int(kwargs["decoder_chunk_overlap"]),
            codec_fps=float(kwargs["codec_fps"]),
            output_dir=output_dir,
            output_format=kwargs["output_format"],
            mp3_bitrate=kwargs["mp3_bitrate"],
            ffmpeg_bin=kwargs["ffmpeg_bin"],
            python_executable=kwargs["python_executable"],
            gpu_id=kwargs["gpu_id"],
            master_addr=kwargs["master_addr"],
            master_port=kwargs["master_port"],
            tensor_model_parallel_size=int(kwargs["tensor_model_parallel_size"]),
            pipeline_model_parallel_size=int(kwargs["pipeline_model_parallel_size"]),
            tokenizer_type=kwargs["tokenizer_type"],
            norm_epsilon=float(kwargs["norm_epsilon"]),
            inference_max_seq_length=int(kwargs["inference_max_seq_length"]),
            inference_max_requests=int(kwargs["inference_max_requests"]),
            inference_batch_times_seqlen_threshold=int(kwargs["inference_batch_times_seqlen_threshold"]),
            transformer_impl=kwargs["transformer_impl"],
            attention_backend=kwargs["attention_backend"],
            dtype=kwargs["dtype"],
            stream=bool(kwargs["stream"]),
            enable_cuda_graph=bool(kwargs["enable_cuda_graph"]),
            flash_decode=bool(kwargs["flash_decode"]),
            tokens_per_minute=int(kwargs["tokens_per_minute"]),
            backbone_max_prompt_len=int(kwargs["backbone_max_prompt_len"]),
            superres_max_prompt_len=int(kwargs["superres_max_prompt_len"]),
        )
        return config, json.dumps(asdict(config), ensure_ascii=False, indent=2)


class KhalaInference:
    DESCRIPTION = (
        "Runs Khala music inference from a Khala model configuration. "
        "When auto_release_vram is enabled, each generation stage runs in a short-lived child process."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "khala_model": ("KHALA_MODEL",),
                "description": ("STRING", {"default": "A bright pop song with emotional vocals.", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "genre": (GENRE_OPTIONS, {"default": "Pop"}),
                "language": (LANGUAGE_OPTIONS, {"default": "Chinese"}),
                "tags": ("STRING", {"default": ""}),
                "duration": ("INT", {"default": 2, "min": 1, "max": 30, "step": 1}),
                "superres_text_mode": (SUPERRES_TEXT_MODES, {"default": "same_as_backbone"}),
                "sampling_method": (["top_k", "top_p", "greedy"], {"default": "top_k"}),
                "top_k_bb": ("INT", {"default": 50, "min": 0, "max": 10000, "step": 1}),
                "top_p_bb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k_sr": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "num_tokens_to_generate": ("INT", {"default": 23552, "min": 1, "max": 1048576, "step": 1}),
                "return_log_probs": ("BOOLEAN", {"default": False}),
                "top_n_logprobs": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "return_prompt_top_n_logprobs": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1283, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "auto_release_vram": ("BOOLEAN", {"default": True}),
                "raw_user_input": ("STRING", {"default": "", "multiline": True}),
                "raw_mode": ("STRING", {"default": ""}),
                "raw_prompt_mode": ("STRING", {"default": ""}),
                "extra_megatron_args": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_path", "metadata_json", "result_json")
    OUTPUT_TOOLTIPS = (
        "Generated ComfyUI AUDIO object.",
        "Generated WAV path when available, otherwise MP3 path.",
        "Khala metadata JSON written beside the audio.",
        "Full Khala result JSON from the backend worker.",
    )
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def generate(
        self,
        khala_model: KhalaModelConfig,
        description: str,
        lyrics: str,
        genre: str,
        language: str,
        tags: str,
        duration: int,
        superres_text_mode: str,
        sampling_method: str,
        top_k_bb: int,
        top_p_bb: float,
        top_k_sr: int,
        temperature: float,
        num_tokens_to_generate: int,
        return_log_probs: bool,
        top_n_logprobs: int,
        return_prompt_top_n_logprobs: bool,
        seed: int,
        seed_override: int,
        auto_release_vram: bool,
        raw_user_input: str,
        raw_mode: str,
        raw_prompt_mode: str,
        extra_megatron_args: str,
    ):
        top_k_bb, top_p_bb = normalize_request_sampling(sampling_method, top_k_bb, top_p_bb)
        request = {
            "genre": genre,
            "language": language,
            "tags": tags,
            "description": description,
            "duration": int(duration),
            "lyrics": lyrics,
            "backbone_name": khala_model.backbone_name,
            "superres_name": khala_model.superres_name,
            "top_k_bb": int(top_k_bb),
            "top_p_bb": float(top_p_bb),
            "top_k_sr": int(top_k_sr),
            "temperature": float(temperature),
            "return_log_probs": bool(return_log_probs),
            "top_n_logprobs": int(top_n_logprobs),
            "return_prompt_top_n_logprobs": bool(return_prompt_top_n_logprobs),
            "superres_text_mode": superres_text_mode,
            "raw_user_input": raw_user_input,
            "raw_mode": raw_mode,
            "raw_prompt_mode": raw_prompt_mode,
            "seed_override": int(seed_override),
        }

        with tempfile.TemporaryDirectory(prefix="comfy_khala_") as temp_dir:
            request_json = Path(temp_dir) / "request.json"
            result_json = Path(temp_dir) / "result.json"
            request_json.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

            command = build_command(
                khala_model,
                request_json,
                result_json,
                seed,
                bool(auto_release_vram),
                int(num_tokens_to_generate),
                extra_megatron_args,
            )

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.run(
                command,
                cwd=str(KHALA_SOURCE_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if process.returncode != 0:
                stderr_tail = (process.stderr or process.stdout or "")[-6000:]
                raise RuntimeError(f"Khala inference failed with exit code {process.returncode}:\n{stderr_tail}")
            if not result_json.is_file():
                stdout_tail = (process.stdout or "")[-6000:]
                raise RuntimeError(f"Khala inference did not write result JSON.\n{stdout_tail}")
            result = json.loads(result_json.read_text(encoding="utf-8"))

        if result.get("status") != "ok":
            raise RuntimeError(result.get("error") or json.dumps(result, ensure_ascii=False, indent=2))

        output_dir = Path(result["output_dir"])
        wav_name = result.get("wav_filename") or ""
        mp3_name = result.get("mp3_filename") or ""
        metadata_name = result.get("metadata_filename") or ""
        wav_path = output_dir / wav_name if wav_name else Path()
        mp3_path = output_dir / mp3_name if mp3_name else Path()
        audio_path = wav_path if wav_path.is_file() else mp3_path
        metadata_path = output_dir / metadata_name if metadata_name else Path()
        audio = load_audio_result(wav_path, mp3_path)
        metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.is_file() else ""

        return (
            audio,
            str(audio_path) if audio_path else "",
            metadata,
            json.dumps(result, ensure_ascii=False, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "KhalaModelDownloader": KhalaModelDownloader,
    "KhalaModelLoader": KhalaModelLoader,
    "KhalaInference": KhalaInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KhalaModelDownloader": "Khala Model Downloader",
    "KhalaModelLoader": "Khala Model Loader",
    "KhalaInference": "Khala Inference",
}
