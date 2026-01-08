# 电力巡检图像智能检测系统 (Image Detection)

本项目是一个基于 YOLO 的电力巡检图像智能检测系统，采用前后端分离架构。

## 🚀 快速环境配置 (Python Backend)

推荐使用 `uv` 进行包管理，以快速安装支持 CUDA 12.1 的 PyTorch 环境：

```bash
cd backend/python

# 1. 同步环境 (自动安装所有依赖)
uv sync

# 2. 如果需要手动更新 PyTorch (CUDA 12.1)
uv add torch torchvision torchaudio --index pytorch
```

## 📂 项目结构

- `backend/python/`: YOLO 训练、API 服务及 Web 交互界面。
- `backend/cpp/`: 高性能推理实现。
- `frontend/`: 系统前端展示。

详细开发文档请参考各目录下的 `README.md`。
