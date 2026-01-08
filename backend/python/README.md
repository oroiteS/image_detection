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

确保已安装 `uv`。在 `backend/python` 目录下，运行以下命令自动同步环境并安装支持 CUDA 12.1 的 PyTorch：

```bash
# 同步环境（自动读取 pyproject.toml 中的 pytorch 源配置）
uv sync
```

如果你需要手动添加或更新 PyTorch 依赖，可以使用以下命令：

```bash
# 指定使用 pytorch 索引安装 CUDA 12.1 版本
uv add torch torchvision torchaudio --index pytorch
```

### 2. 环境自检

运行以下脚本以验证 GPU 和相关库是否配置正确：

```bash
python src/image_detection/utils/verify_env.py
```

## 📂 项目结构

```text
image_detection/
├── src/
│   └── image_detection/
│       ├── core/           # 核心逻辑
│       ├── data/           # 数据处理 (split_dataset.py, inflect.csv)
│       ├── training/       # 训练逻辑 (yolo_train.py)
│       ├── utils/          # 工具类 (verify_env.py)
│       └── web/            # Web 界面 (app.py)
├── tests/                  # 测试用例
├── datasets/               # 数据集目录
│   ├── raw_data/           # 原始标注数据
│   └── power_inspection/   # 划分后的数据集
├── runs/                   # 训练结果与权重
├── pyproject.toml          # 项目配置文件
└── README.md               # 项目说明文档
```

## 📖 使用说明

### 数据准备
1. 将原始图像和标注文件放入 `datasets/raw_data/`。
2. 运行 `python src/image_detection/data/split_dataset.py` 进行数据集划分。

### 模型训练
运行 `python src/image_detection/training/yolo_train.py` 开始训练、评估或导出模型。

### 运行 Web 界面
运行 `python src/image_detection/web/app.py` 启动可视化检测界面。

## 🛠️ 技术栈

- **语言**: Python 3.10
- **深度学习**: PyTorch (CUDA 12.1), Ultralytics YOLO
- **Web 框架**: FastAPI, Gradio
- **部署**: ONNX
- **工具**: OpenCV, Pandas, Matplotlib
