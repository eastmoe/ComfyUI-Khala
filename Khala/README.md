# Khala Core Inference

This repository has been trimmed to keep only the Khala inference path:

- text tokenizer loading
- backbone acoustic-token generation
- super-resolution token generation
- decoder waveform reconstruction

Training, evaluation, dataset preprocessing, WebUI, GUI, frontend dispatcher, tests, examples, and related dependencies have been removed.

## Runtime Requirements

- NVIDIA GPU with a CUDA/PyTorch/Transformer Engine stack compatible with the original Khala checkpoints.
- Python runtime based on the NVIDIA NGC PyTorch image is recommended.
- `ffmpeg` for MP3 export.
- Model weights under `checkpoints/` at the repository root.

Install the remaining Python dependencies:

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

## Model Layout

```text
Khala/
├── backend/
├── core/
├── models/
│   ├── Decoder/
│   ├── Megatron/
│   └── Tokenizer/
└── checkpoints/
    ├── backbone/
    ├── superresolution/
    └── dac_rvq_2490000.ckpt
```

## Run Inference

Create a JSON request:

```json
{
  "genre": "Pop",
  "language": "Chinese",
  "tags": "",
  "description": "A bright pop song with emotional vocals.",
  "duration": 2,
  "lyrics": "",
  "top_k_bb": 50,
  "top_k_sr": 10,
  "temperature": 1.0
}
```

Run the core inference launcher:

```bash
cd backend
bash run_backend.sh --gpus 0 --request-json request.json --result-json result.json
```

The default inference path uses Megatron's local transformer implementation and
the portable `sdpa` attention setting. The launcher passes Megatron's `unfused`
backend for compatibility and enables a Khala PyTorch SDPA fallback inside the
local attention layer. This avoids requiring NVIDIA Transformer Engine.

To opt into Transformer Engine and faster attention paths, pass them explicitly:

```bash
python inference.py --prompt "A bright pop song." \
  --transformer-impl transformer_engine \
  --attention-backend auto \
  --enable-cuda-graph \
  --flash-decode
```

Available `--attention-backend` values are `sdpa`, `auto`, `flash`, `fused`,
`unfused`, and `local`. `flash` / `auto` require the matching CUDA attention
stack in the runtime. This vendored Megatron tree does not expose Sage Attention
or a separate Triton attention backend for this GPT inference path.

Generated audio is written under `backend/generated_audio/`; structured metadata is written to the result JSON.

## Remaining Code

- `backend/backend_worker.py`: JSON-driven core inference runner.
- `backend/run_backend.sh`: thin launcher for one GPU and one request.
- `core/`: Khala-specific model patches.
- `models/Decoder/`: RVQ decoder implementation.
- `models/Tokenizer/`: tokenizer files.
- `models/Megatron/`: reduced Megatron runtime needed to construct models, initialize distributed state, and load checkpoints for inference.
