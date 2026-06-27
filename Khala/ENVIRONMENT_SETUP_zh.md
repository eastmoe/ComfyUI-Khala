# 环境配置

当前仓库是仅推理版本，需要与原始 Khala checkpoint 兼容的 CUDA / PyTorch 运行环境。仍推荐使用 NVIDIA NGC PyTorch `25.02-py3` 镜像。

## Python 依赖

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

仓库不再需要 Node.js、Vite、React、FastAPI、Uvicorn 或 httpx。

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
