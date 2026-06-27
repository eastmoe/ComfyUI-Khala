# Khala 核心推理版

本仓库已裁剪为只保留 Khala 推理链路：

- 文本 tokenizer 加载
- backbone 声学 token 生成
- super-resolution token 生成
- decoder 波形重建

训练、评估、数据集预处理、WebUI、GUI、前端调度层、测试、示例以及相关依赖已移除。

## 运行环境

- NVIDIA GPU，以及与原始 Khala checkpoint 兼容的 CUDA / PyTorch / Transformer Engine 环境。
- 推荐基于 NVIDIA NGC PyTorch 镜像运行。
- 需要 `ffmpeg` 用于导出 MP3。
- 模型权重放在仓库根目录的 `checkpoints/` 下。

安装剩余 Python 依赖：

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

## 模型目录

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

## 执行推理

创建 JSON 请求：

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

运行核心推理入口：

```bash
cd backend
bash run_backend.sh --gpus 0 --request-json request.json --result-json result.json
```

生成音频会写入 `backend/generated_audio/`，结构化结果会写入指定的 result JSON。
