#!/usr/bin/env python3
"""
Cross-platform launcher for one Khala backend request.

This replaces the former bash wrapper with Python-only argument handling and
directly invokes backend_worker.main() after configuring the runtime process.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MEGATRON_ROOT = PROJECT_ROOT / "models" / "Megatron"
DECODER_ROOT = PROJECT_ROOT / "models" / "Decoder"

ATTENTION_BACKEND_MAP = {
    "sdpa": "unfused",
    "auto": "auto",
    "flash": "flash",
    "fused": "fused",
    "unfused": "unfused",
    "local": "local",
}

DEFAULT_MEGATRON_ARGS = [
    "--tensor-model-parallel-size",
    "1",
    "--pipeline-model-parallel-size",
    "1",
    "--tokenizer-type",
    "NullTokenizer",
    "--norm-epsilon",
    "1e-6",
    "--num-tokens-to-generate",
    "23552",
    "--inference-max-seq-length",
    "25600",
    "--stream",
    "--bf16",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Khala backend_worker.py from a portable Python launcher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Arguments after a standalone '--' are forwarded to backend_worker.py / Megatron.",
    )
    parser.add_argument("--gpus", default="0", help="Physical GPU id to use.")
    parser.add_argument(
        "--request-json",
        required=True,
        help="JSON request file consumed by backend_worker.py.",
    )
    parser.add_argument(
        "--result-json",
        default="",
        help="Optional JSON result file. If omitted, result prints to stdout.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=["one_shot", "keep_loaded"],
        default="one_shot",
    )
    parser.add_argument("--seed", type=int, default=1283, help="Generation seed.")
    parser.add_argument(
        "--transformer-impl",
        choices=["local", "transformer_engine"],
        default="local",
    )
    parser.add_argument(
        "--attention-backend",
        choices=list(ATTENTION_BACKEND_MAP),
        default="sdpa",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Enable Megatron CUDA graph warmup.",
    )
    parser.add_argument(
        "--flash-decode",
        action="store_true",
        help="Enable Megatron flash decode.",
    )
    return parser


def split_launcher_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    delimiter = argv.index("--")
    return argv[:delimiter], argv[delimiter + 1 :]


def configure_runtime_env(gpu_id: str, attention_backend: str) -> None:
    pythonpath_parts = [
        str(PROJECT_ROOT),
        str(MEGATRON_ROOT),
        str(DECODER_ROOT),
    ]
    if os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(os.environ["PYTHONPATH"])

    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "8791")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["KHALA_ATTENTION_BACKEND"] = attention_backend
    os.environ.setdefault("KHALA_PYTHON", sys.executable)

    for path in reversed(pythonpath_parts):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


def build_worker_argv(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    worker_argv = [
        str(SCRIPT_DIR / "backend_worker.py"),
        "--runtime-mode",
        args.runtime_mode,
        "--seed",
        str(args.seed),
        "--request-json",
        args.request_json,
        *DEFAULT_MEGATRON_ARGS,
        "--transformer-impl",
        args.transformer_impl,
        "--attention-backend",
        ATTENTION_BACKEND_MAP[args.attention_backend],
    ]
    if args.enable_cuda_graph:
        worker_argv.append("--enable-cuda-graph")
    if args.flash_decode:
        worker_argv.append("--flash-decode")
    worker_argv.extend(passthrough)
    if args.result_json:
        worker_argv.extend(["--result-json", args.result_json])
    return worker_argv


def validate_gpu_id(value: str, parser: argparse.ArgumentParser) -> None:
    if not value.isdigit():
        parser.error("--gpus expects one integer GPU id.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    launcher_args, passthrough = split_launcher_args(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(launcher_args)
    validate_gpu_id(args.gpus, parser)

    configure_runtime_env(args.gpus, args.attention_backend)
    os.chdir(SCRIPT_DIR)

    import backend_worker

    worker_argv = build_worker_argv(args, passthrough)
    try:
        backend_worker.main(worker_argv)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
