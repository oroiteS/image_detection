# 电力巡检图像智能检测系统 (Image Detection)

本项目是一个基于 YOLO 的电力巡检图像智能检测系统，采用前后端分离架构，后端支持 Python 和 C++ 混合开发。

## 项目结构

```
id/
├── backend/
│   ├── python/          # Python后端 (YOLO训练、API服务)
│   └── cpp/             # C++后端 (高性能推理)
├── frontend/            # 前端界面
├── .github/
│   └── workflows/       # CI/CD配置
├── pyproject.toml       # 项目配置
└── README.md
```

## 快速开始

### Python Backend

```bash
cd backend/python
uv sync
uv run python -m image_detection.utils.verify_env
```

### C++ Backend

```bash
cd backend/cpp
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ ..
make
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

## 技术栈

- **Python Backend**: PyTorch, Ultralytics YOLO, FastAPI, Gradio (包管理: uv)
- **C++ Backend**: C++17, CMake, g++, OpenCV
- **Frontend**: (待定) (包管理: pnpm)
- **CI/CD**: GitHub Actions

## 开发指南

详细文档请查看各子项目的 README：
- [Python Backend](./backend/python/README.md)
- [C++ Backend](./backend/cpp/README.md)
- [Frontend](./frontend/README.md)

