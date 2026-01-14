# 电力巡检图像智能检测系统 (Image Detection System)

本项目是一个基于 YOLO 的电力巡检图像智能检测系统，采用前后端分离架构。系统结合了 Python 的灵活性与 C++ TensorRT 的高性能推理能力，并配备了现代化的 Vue 3 前端界面。

## ✨ 核心特性

*   **高性能推理 (Backend)**:
    *   **Python**: 集成 Ultralytics YOLO，支持动态模型切换与 RESTful API (FastAPI)。
    *   **C++**: 基于 TensorRT 的深度优化，支持 FP16/INT8 量化、自定义 CUDA 后处理算子融合及多 Batch 并行推理。
*   **现代化交互 (Frontend)**:
    *   基于 Vue 3 + Vite + Tailwind CSS 4.0 构建。
    *   提供极简 UI、图片缩放查看器及本地历史记录管理功能。

## 📂 项目结构

| 目录 | 说明 | 关键技术 |
| :--- | :--- | :--- |
| [`backend/python/`](backend/python/README.md) | 业务后端与训练 | FastAPI, PyTorch, Ultralytics YOLO |
| [`backend/cpp/`](backend/cpp/README.md) | 高性能推理引擎 | TensorRT, CUDA, C++ |
| [`frontend/`](frontend/README.md) | 用户交互界面 | Vue 3, Vite, Tailwind CSS |

## 🐳 容器化部署 (Docker)

本项目支持 Docker Compose 一键部署完整系统（包含 GPU 支持）。

### 前置要求
*   Docker & Docker Compose
*   NVIDIA Driver & NVIDIA Container Toolkit (用于 GPU 加速)

### 一键启动
在项目根目录下执行：

```bash
docker-compose up --build
```

启动后访问：
*   **前端界面**: `http://localhost:5173`
*   **后端 API 文档**: `http://localhost:8000/docs`

## 🚀 本地开发快速开始

### 1. Python 后端环境配置

推荐使用 `uv` 进行包管理，以快速安装支持 CUDA 12.1 的 PyTorch 环境：

```bash
cd backend/python

# 同步环境 (自动安装所有依赖)
uv sync

# 启动 API 服务
uv run python src/image_detection/web/api.py
```

### 2. C++ 推理加速 (可选)

如需极致性能，可编译 C++ 模块。详细步骤请参考 [C++ Backend README](backend/cpp/README.md)。

### 3. 前端启动

```bash
cd frontend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev
```

访问 `http://localhost:5173` 即可使用系统。

---

详细开发文档请参考各子目录下的 `README.md`。
