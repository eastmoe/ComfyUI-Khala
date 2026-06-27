#!/usr/bin/env python3
"""
Unified Khala inference CLI.

This wrapper turns command-line options into the JSON request consumed by
backend/backend_worker.py, sets the runtime environment, and forwards model /
Megatron options to the core worker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
BACKEND_WORKER = BACKEND_DIR / "backend_worker.py"


REQUEST_DEFAULTS: dict[str, Any] = {
    "genre": "Pop",
    "language": "Chinese",
    "tags": "",
    "description": "",
    "duration": 2,
    "lyrics": "",
    "backbone_name": "",
    "superres_name": "",
    "top_k_bb": 50,
    "top_p_bb": 0.0,
    "top_k_sr": 10,
    "temperature": 1.0,
    "return_log_probs": False,
    "top_n_logprobs": 0,
    "return_prompt_top_n_logprobs": False,
    "superres_text_mode": "same_as_backbone",
    "raw_user_input": "",
    "raw_mode": "",
    "raw_prompt_mode": "",
    "seed_override": 0,
}


def read_text_arg(value: str | None, file_path: str | None) -> str | None:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return value


def add_bool_pair(
    parser: argparse.ArgumentParser,
    positive: str,
    negative: str,
    dest: str,
    default: bool,
    help_text: str,
    negative_help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(positive, dest=dest, action="store_true", help=help_text)
    group.add_argument(negative, dest=dest, action="store_false", help=negative_help_text)
    parser.set_defaults(**{dest: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Khala core music inference from one Python CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Arguments after a standalone '--' are forwarded to backend_worker.py / Megatron.",
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument("--request-json", "--input-json", dest="request_json", default="")
    io_group.add_argument("--request-json-out", default="", help="Persist the generated worker request JSON.")
    io_group.add_argument("--result-json", "--output-json", dest="result_json", default="")
    io_group.add_argument("--output-dir", default=str(BACKEND_DIR / "generated_audio"))
    io_group.add_argument("--output-format", choices=["wav", "mp3", "both"], default="both")
    io_group.add_argument("--mp3-bitrate", default="320k")
    io_group.add_argument("--ffmpeg-bin", default="ffmpeg")
    io_group.add_argument("--print-command", action="store_true")
    io_group.add_argument("--dry-run", action="store_true", help="Build request/command without running models.")

    prompt_group = parser.add_argument_group("prompt")
    prompt_group.add_argument("--prompt", "--description", dest="description", default=None)
    prompt_group.add_argument("--prompt-file", "--description-file", dest="description_file", default=None)
    prompt_group.add_argument("--lyrics", default=None)
    prompt_group.add_argument("--lyrics-file", default=None)
    prompt_group.add_argument("--genre", default=None)
    prompt_group.add_argument("--language", default=None)
    prompt_group.add_argument("--tags", default=None)
    prompt_group.add_argument("--duration", type=int, default=None, help="Requested duration in minutes.")
    prompt_group.add_argument(
        "--superres-text-mode",
        choices=["same_as_backbone", "same_as_backbone_no_description", "separate"],
        default=None,
    )
    prompt_group.add_argument("--raw-user-input", default=None)
    prompt_group.add_argument("--raw-mode", default=None)
    prompt_group.add_argument("--raw-prompt-mode", default=None)

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument(
        "--sampling-method",
        choices=["top_k", "top_p", "greedy"],
        default="top_k",
        help="Backbone sampler. Superres currently uses top-k projection sampling.",
    )
    sampling_group.add_argument("--top-k-bb", type=int, default=None)
    sampling_group.add_argument("--top-p-bb", type=float, default=None)
    sampling_group.add_argument("--top-k-sr", type=int, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None)
    sampling_group.add_argument(
        "--num-tokens-to-generate",
        "--steps",
        "--bb-steps",
        dest="num_tokens_to_generate",
        type=int,
        default=23552,
    )
    sampling_group.add_argument("--return-log-probs", action="store_true")
    sampling_group.add_argument("--top-n-logprobs", type=int, default=None)
    sampling_group.add_argument("--return-prompt-top-n-logprobs", action="store_true")

    model_group = parser.add_argument_group("model paths")
    model_group.add_argument("--tokenizer-path", default=str(PROJECT_ROOT / "models" / "Tokenizer"))
    model_group.add_argument("--backbone-name", default="")
    model_group.add_argument("--backbone-path", default=str(PROJECT_ROOT / "checkpoints" / "backbone"))
    model_group.add_argument("--backbone-vocab-size", type=int, default=130304)
    model_group.add_argument("--superres-name", default="")
    model_group.add_argument("--superres-path", default=str(PROJECT_ROOT / "checkpoints" / "superresolution"))
    model_group.add_argument("--superres-vocab-size", type=int, default=193792)
    model_group.add_argument(
        "--decoder-config-path",
        default=str(PROJECT_ROOT / "models" / "Decoder" / "dac_rvq_1024_64_golden.yaml"),
    )
    model_group.add_argument(
        "--decoder-checkpoint-path",
        default=str(PROJECT_ROOT / "checkpoints" / "dac_rvq_2490000.ckpt"),
    )

    decode_group = parser.add_argument_group("decoder")
    decode_group.add_argument("--decoder-sample-rate", type=int, default=44100)
    decode_group.add_argument("--decoder-chunk-size", type=int, default=1920)
    decode_group.add_argument("--decoder-chunk-overlap", type=int, default=480)
    decode_group.add_argument("--codec-fps", type=float, default=21.5)

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--gpus", "--gpu", dest="gpus", default="0")
    runtime_group.add_argument("--seed", type=int, default=1283)
    runtime_group.add_argument("--seed-override", type=int, default=0)
    runtime_group.add_argument("--runtime-mode", choices=["one_shot", "keep_loaded"], default="one_shot")
    runtime_group.add_argument("--python", default=sys.executable)
    runtime_group.add_argument("--master-addr", default="127.0.0.1")
    runtime_group.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "8791"))
    runtime_group.add_argument("--tokens-per-minute", type=int, default=2584)
    runtime_group.add_argument("--backbone-max-prompt-len", type=int, default=4096)
    runtime_group.add_argument("--superres-max-prompt-len", type=int, default=2048)

    megatron_group = parser.add_argument_group("Megatron inference")
    megatron_group.add_argument("--tensor-model-parallel-size", type=int, default=1)
    megatron_group.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    megatron_group.add_argument("--tokenizer-type", default="NullTokenizer")
    megatron_group.add_argument("--norm-epsilon", type=float, default=1e-6)
    megatron_group.add_argument("--inference-max-seq-length", type=int, default=25600)
    megatron_group.add_argument("--inference-max-requests", type=int, default=None)
    megatron_group.add_argument("--inference-batch-times-seqlen-threshold", type=int, default=None)
    add_bool_pair(
        megatron_group,
        "--stream",
        "--no-stream",
        "stream",
        True,
        "Stream backbone progress.",
        "Disable backbone progress streaming.",
    )
    add_bool_pair(
        megatron_group,
        "--enable-cuda-graph",
        "--disable-cuda-graph",
        "enable_cuda_graph",
        True,
        "Enable CUDA graph warmup for backbone.",
        "Disable CUDA graph warmup.",
    )
    add_bool_pair(
        megatron_group,
        "--flash-decode",
        "--no-flash-decode",
        "flash_decode",
        True,
        "Enable Megatron flash decode for backbone.",
        "Disable Megatron flash decode.",
    )
    dtype_group = megatron_group.add_mutually_exclusive_group()
    dtype_group.add_argument("--bf16", dest="dtype", action="store_const", const="bf16")
    dtype_group.add_argument("--fp16", dest="dtype", action="store_const", const="fp16")
    dtype_group.add_argument("--fp32", dest="dtype", action="store_const", const="fp32")
    parser.set_defaults(dtype="bf16")

    return parser


def load_request(path: str) -> dict[str, Any]:
    request = REQUEST_DEFAULTS.copy()
    if path:
        with open(path, "r", encoding="utf-8") as file:
            request.update(json.load(file))
    return request


def apply_cli_overrides(request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    description = read_text_arg(args.description, args.description_file)
    lyrics = read_text_arg(args.lyrics, args.lyrics_file)
    overrides = {
        "description": description,
        "lyrics": lyrics,
        "genre": args.genre,
        "language": args.language,
        "tags": args.tags,
        "duration": args.duration,
        "superres_text_mode": args.superres_text_mode,
        "raw_user_input": args.raw_user_input,
        "raw_mode": args.raw_mode,
        "raw_prompt_mode": args.raw_prompt_mode,
        "top_k_bb": args.top_k_bb,
        "top_p_bb": args.top_p_bb,
        "top_k_sr": args.top_k_sr,
        "temperature": args.temperature,
        "top_n_logprobs": args.top_n_logprobs,
    }
    for key, value in overrides.items():
        if value is not None:
            request[key] = value

    if args.backbone_name:
        request["backbone_name"] = args.backbone_name
    if args.superres_name:
        request["superres_name"] = args.superres_name
    if args.seed_override > 0:
        request["seed_override"] = args.seed_override
    if args.return_log_probs:
        request["return_log_probs"] = True
    if args.return_prompt_top_n_logprobs:
        request["return_prompt_top_n_logprobs"] = True

    if args.sampling_method == "greedy":
        request["top_k_bb"] = 1
        request["top_p_bb"] = 0.0
    elif args.sampling_method == "top_p":
        request["top_k_bb"] = 0
        request["top_p_bb"] = float(request.get("top_p_bb") or 0.95)
    else:
        request["top_p_bb"] = 0.0 if request.get("top_p_bb") is None else request["top_p_bb"]

    if int(request.get("top_k_bb", 0)) > 0 and float(request.get("top_p_bb", 0.0)) > 0.0:
        raise ValueError("Backbone top-k and top-p are mutually exclusive; use one sampling method.")

    return request


def append_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_worker_command(args: argparse.Namespace, request_path: Path) -> list[str]:
    command = [
        args.python,
        str(BACKEND_WORKER),
        "--runtime-mode",
        args.runtime_mode,
        "--seed",
        str(args.seed),
        "--request-json",
        str(request_path),
        "--tokenizer-path",
        args.tokenizer_path,
        "--output-dir",
        args.output_dir,
        "--output-format",
        args.output_format,
        "--mp3-bitrate",
        args.mp3_bitrate,
        "--ffmpeg-bin",
        args.ffmpeg_bin,
        "--backbone-path",
        args.backbone_path,
        "--backbone-vocab-size",
        str(args.backbone_vocab_size),
        "--superres-path",
        args.superres_path,
        "--superres-vocab-size",
        str(args.superres_vocab_size),
        "--decoder-config-path",
        args.decoder_config_path,
        "--decoder-checkpoint-path",
        args.decoder_checkpoint_path,
        "--decoder-sample-rate",
        str(args.decoder_sample_rate),
        "--decoder-chunk-size",
        str(args.decoder_chunk_size),
        "--decoder-chunk-overlap",
        str(args.decoder_chunk_overlap),
        "--codec-fps",
        str(args.codec_fps),
        "--tokens-per-minute",
        str(args.tokens_per_minute),
        "--backbone-max-prompt-len",
        str(args.backbone_max_prompt_len),
        "--superres-max-prompt-len",
        str(args.superres_max_prompt_len),
        "--tensor-model-parallel-size",
        str(args.tensor_model_parallel_size),
        "--pipeline-model-parallel-size",
        str(args.pipeline_model_parallel_size),
        "--tokenizer-type",
        args.tokenizer_type,
        "--norm-epsilon",
        str(args.norm_epsilon),
        "--num-tokens-to-generate",
        str(args.num_tokens_to_generate),
        "--inference-max-seq-length",
        str(args.inference_max_seq_length),
    ]

    if args.backbone_name:
        command.extend(["--backbone-name", args.backbone_name])
    if args.superres_name:
        command.extend(["--superres-name", args.superres_name])
    if args.result_json:
        command.extend(["--result-json", args.result_json])
    append_value(command, "--inference-max-requests", args.inference_max_requests)
    append_value(
        command,
        "--inference-batch-times-seqlen-threshold",
        args.inference_batch_times_seqlen_threshold,
    )
    if args.stream:
        command.append("--stream")
    if args.enable_cuda_graph:
        command.append("--enable-cuda-graph")
    if args.flash_decode:
        command.append("--flash-decode")
    if args.dtype != "fp32":
        command.append(f"--{args.dtype}")
    return command


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "models" / "Megatron"),
        str(PROJECT_ROOT / "models" / "Decoder"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    env["MASTER_ADDR"] = str(args.master_addr)
    env["MASTER_PORT"] = str(args.master_port)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    try:
        request = apply_cli_overrides(load_request(args.request_json), args)
    except Exception as exc:
        parser.error(str(exc))
    if not args.backbone_name and request.get("backbone_name"):
        args.backbone_name = str(request["backbone_name"])
    if not args.superres_name and request.get("superres_name"):
        args.superres_name = str(request["superres_name"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json) if args.result_json else None
    if result_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="khala_inference_") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.request_json_out:
            out_path = Path(args.request_json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(request_path, out_path)

        command = build_worker_command(args, request_path)
        command.extend(passthrough)
        if args.print_command:
            print(" ".join(command))
        if args.dry_run:
            if not args.print_command:
                print(" ".join(command))
            if not args.request_json_out:
                print(json.dumps(request, ensure_ascii=False, indent=2))
            return 0

        completed = subprocess.run(command, cwd=BACKEND_DIR, env=build_env(args), text=True)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
