# 环境配置

当前仓库是仅推理版本，需要与原始 Khala checkpoint 兼容的 CUDA / PyTorch 运行环境。仍推荐使用 NVIDIA NGC PyTorch `25.02-py3` 镜像。

## Python 依赖

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

仓库不再需要 Node.js、Vite、React、FastAPI、Uvicorn 或 httpx。

## Transformer / Attention Backend

CLI 默认使用 `--transformer-impl local --attention-backend sdpa`。在当前
vendored Megatron runtime 中，launcher 会向 Megatron 传入兼容的 `unfused`
backend，并在本地 attention 层内启用 Khala 的 PyTorch SDPA fallback，用作
Transformer Engine 或 flash attention kernel 不可用时的便携托底路径。

只有在运行环境提供 NVIDIA Transformer Engine 时，才建议使用
`--transformer-impl transformer_engine`。`--attention-backend auto` 或 `flash`
也需要对应 CUDA attention 栈。当前 GPT 推理路径没有暴露 Sage Attention 或独立的
Triton attention backend。

## 系统依赖

安装 `ffmpeg` 用于导出 WAV / MP3：

```bash
apt update
apt install -y ffmpeg
```

## Checkpoints

模型权重放在仓库根目录：

```text
checkpoints/
├── backbone/
├── superresolution/
└── dac_rvq_2490000.ckpt
```

## 运行

```bash
cd backend
bash run_backend.sh --gpus 0 --request-json request.json --result-json result.json
```
