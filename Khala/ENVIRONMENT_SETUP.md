# Environment Setup

This inference-only tree expects a CUDA/PyTorch runtime compatible with the original Khala checkpoints. The recommended base is still the NVIDIA NGC PyTorch `25.02-py3` image.

## Python Dependencies

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

The repository no longer needs Node.js, Vite, React, FastAPI, Uvicorn, or httpx.

## System Dependency

Install `ffmpeg` for WAV/MP3 export:

```bash
apt update
apt install -y ffmpeg
```

## Checkpoints

Place model weights at the repository root:

```text
checkpoints/
├── backbone/
├── superresolution/
└── dac_rvq_2490000.ckpt
```

## Run

```bash
cd backend
bash run_backend.sh --gpus 0 --request-json request.json --result-json result.json
```
