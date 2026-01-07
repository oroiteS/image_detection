# 电力巡检图像智能检测系统 (image-detection)

本项目是一个基于 YOLO 的电力巡检图像智能检测系统，旨在利用深度学习技术自动识别电力设备（如绝缘子）的缺陷或状态。

## 🚀 项目特点

- **高效检测**：基于 Ultralytics YOLO 框架，提供高性能的目标检测能力。
- **环境适配**：针对 RTX 4070s 等现代 GPU 进行了优化，支持 CUDA 12.1。
- **Web 交互**：集成 Gradio 和 FastAPI，方便用户通过浏览器进行交互式检测。
- **模型导出**：支持导出为 ONNX 格式，便于在不同平台上部署。

## 🛠️ 环境配置

本项目建议使用 Python 3.10 环境，并推荐使用 `uv` 进行包管理。

### 1. 安装依赖

确保已安装 `uv`，然后在项目根目录下运行：

```bash
uv sync
```

或者使用 pip 安装（请确保 CUDA 版本匹配）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi gradio uvicorn ultralytics onnx onnxsim netron opencv-python matplotlib pandas
```

### 2. 环境自检

运行以下脚本以验证 GPU 和相关库是否配置正确：

```bash
python verify_env.py
```

## 📂 项目结构

```text
image_detection/
├── datasets/           # 数据集目录
│   └── raw_data/       # 原始标注数据
├── main.py             # 项目主入口
├── yolo_train.py       # YOLO 模型训练脚本
├── split_dataset.py    # 数据集划分脚本（训练集/验证集/测试集）
├── verify_env.py       # 环境检查工具
├── pyproject.toml      # 项目配置文件及依赖管理
└── README.md           # 项目说明文档
```

## 📖 使用说明

### 数据准备

1. 将原始图像和标注文件放入 `datasets/raw_data/`。
2. 运行 `split_dataset.py` 进行数据集划分。

### 模型训练

使用 `yolo_train.py` 开始训练模型。

### 运行 Web 界面

运行 `main.py` 启动基于 Gradio/FastAPI 的可视化检测界面。

## 🛠️ 技术栈

- **语言**: Python 3.10
- **深度学习**: PyTorch, Ultralytics YOLO
- **Web 框架**: FastAPI, Gradio
- **部署**: ONNX
- **工具**: OpenCV, Pandas, Matplotlib
