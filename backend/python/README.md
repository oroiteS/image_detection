# 电力巡检图像智能检测系统 (Backend)

本项目是一个基于 YOLO 的电力巡检图像智能检测系统后端，采用 FastAPI 构建高性能 RESTful API。

## 🚀 核心功能

- **AI 推理引擎**：集成 Ultralytics YOLO，支持实时图像检测。
- **动态模型管理**：支持运行时切换不同的 `.pt` 权重文件。
- **中文映射支持**：通过 `inflect.csv` 自动将检测类别转换为中文描述。
- **高性能 API**：基于 FastAPI，支持异步处理与跨域请求 (CORS)。

## 🛠️ 环境配置

本项目建议使用 Python 3.10，推荐使用 `uv` 进行包管理。

### 1. 安装依赖
```bash
cd backend/python
uv sync
```

### 2. 启动服务
```bash
# 运行 API 服务
python src/image_detection/web/api.py
```
默认运行在 `http://127.0.0.1:8000`。

## 📂 API 接口说明

- `GET /models`: 获取可用模型列表。
- `POST /set_model`: 切换当前使用的模型。
- `POST /detect`: 上传图片进行检测，支持 `conf` 阈值参数。

## 🛠️ 技术栈
- **核心**: Python 3.10, PyTorch (CUDA 12.1)
- **模型**: Ultralytics YOLO
- **框架**: FastAPI, Uvicorn
- **图像处理**: OpenCV, Pillow, NumPy
