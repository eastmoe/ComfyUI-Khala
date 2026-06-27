#!/usr/bin/env bash
set -euo pipefail

# Core inference launcher.
#
# Usage:
#   bash run_backend.sh --request-json request.json --result-json result.json
#   bash run_backend.sh --gpus 1 --request-json request.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MEGATRON_ROOT="$PROJECT_ROOT/models/Megatron"
DECODER_ROOT="$PROJECT_ROOT/models/Decoder"

GPU_ID=0
REQUEST_JSON=""
RESULT_JSON=""
RUNTIME_MODE=one_shot
BASE_SEED=1283

MEGATRON_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --tokenizer-type NullTokenizer
    --norm-epsilon 1e-6
    --num-tokens-to-generate 23552
    --inference-max-seq-length 25600
    --stream
    --enable-cuda-graph
    --flash-decode
    --bf16
)

usage() {
    cat <<'EOF'
Usage:
  bash run_backend.sh --request-json request.json [--result-json result.json]

Options:
  --gpus <id>             Physical GPU id to use. Default: 0
  --request-json <path>   JSON request file consumed by backend_worker.py
  --result-json <path>    Optional JSON result file. If omitted, result prints to stdout.
  --runtime-mode <mode>   one_shot or keep_loaded. Default: one_shot
  --seed <int>            Generation seed. Default: 1283
  --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || { echo "ERROR: --gpus requires a value."; exit 1; }
            GPU_ID="$2"
            shift 2
            ;;
        --request-json)
            [[ $# -ge 2 ]] || { echo "ERROR: --request-json requires a value."; exit 1; }
            REQUEST_JSON="$2"
            shift 2
            ;;
        --result-json)
            [[ $# -ge 2 ]] || { echo "ERROR: --result-json requires a value."; exit 1; }
            RESULT_JSON="$2"
            shift 2
            ;;
        --runtime-mode)
            [[ $# -ge 2 ]] || { echo "ERROR: --runtime-mode requires a value."; exit 1; }
            RUNTIME_MODE="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "ERROR: --seed requires a value."; exit 1; }
            BASE_SEED="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

[[ -n "$REQUEST_JSON" ]] || { echo "ERROR: --request-json is required."; usage; exit 1; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpus expects one integer GPU id."; exit 1; }
[[ "$RUNTIME_MODE" == "one_shot" || "$RUNTIME_MODE" == "keep_loaded" ]] || {
    echo "ERROR: --runtime-mode must be one_shot or keep_loaded."
    exit 1
}

cd "$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT:$MEGATRON_ROOT:$DECODER_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-8791}"
export PYTHONUNBUFFERED=1

cmd=(
    python backend_worker.py
    --runtime-mode "$RUNTIME_MODE"
    --seed "$BASE_SEED"
    --request-json "$REQUEST_JSON"
    "${MEGATRON_ARGS[@]}"
)

if [[ -n "$RESULT_JSON" ]]; then
    cmd+=(--result-json "$RESULT_JSON")
fi

exec "${cmd[@]}"
