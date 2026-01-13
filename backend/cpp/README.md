# C++ Backend

电力巡检图像智能检测系统的C++后端部分。

## 模型优化与部署

使用TensorRT C++进行模型优化部署和加速。模型格式转换优化链：PyTorch模型 → ONNX格式 → TensorRT引擎，确保转换过程精度损失最小。

### TensorRT深度优化 (已全部实现)

*   **图优化**：合并层、消除冗余计算、常量折叠等图级别优化 (TensorRT 自动完成)。
*   **精度优化**：支持FP32、FP16、INT8混合精度推理，平衡精度与速度。
*   **INT8量化**：采用 `IInt8EntropyCalibrator2` 校准技术，最小化量化过程中的精度损失。
*   **内核自动调优**：根据硬件特性自动选择最优计算内核 (TensorRT 自动完成)。
*   **后处理算子CUDA融合**：编写了自定义 CUDA Kernel (`decode_yolo_kernel`)，将复杂的边界框解码和置信度过滤操作融合为单一CUDA内核，大幅减少 Device-to-Host 内存传输开销 (减少约 90%)。
*   **多批次并行推理**：支持动态 Batch Size (1-16)，利用 `Optimization Profile` 实现批量图片同时推理，显著提升GPU利用率。

## 🚀 快速运行指南 (Docker 一体化方案)

由于 C++ 编译产物 (`.so`) 对系统环境（OpenCV/CUDA 版本）依赖极强，**强烈建议直接在 Docker 容器中运行 Python API 服务**。这样可以确保 C++ 模块能被正确加载。

### 1. 启动容器
在 Windows PowerShell 或 Linux 终端中执行：

```bash
# 请将 D:\Github\image_detection 替换为你的实际项目根路径
docker run --gpus all -it --rm -p 8000:8000 -v D:\Github\image_detection:/workspace/project image-detection-backend
```

### 2. 配置环境 (容器内)
进入容器后，推荐使用 `uv` 来快速配置 Python 环境：

```bash
# 1. 更新系统源并安装基础库 (OpenCV 需要 libgl1)
apt update && apt install -y libgl1 curl

# 2. 安装 uv (使用官方脚本)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 关键修正：uv 默认安装在 .local/bin
source $HOME/.local/bin/env

# 3. 配置国内镜像源 (可选，加速下载)
export UV_PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 同步项目环境
cd /workspace/project/backend/python
uv sync
```

### 3. 编译 C++ 模块 (如果尚未编译)
```bash
cd /workspace/project/backend/cpp
mkdir -p build && cd build
cmake ..
make image_detection_cpp

# 将生成的 .so 复制到 Python 目录
cp image_detection_cpp*.so /workspace/project/backend/python/src/image_detection/core/image_detection_cpp.so
```

### 4. 启动 API 服务
```bash
cd /workspace/project/backend/python

# 设置 PYTHONPATH 以便 Python 能找到 image_detection 包
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 使用 uv 启动服务
uv run python src/image_detection/web/api.py
```

服务启动后，即可在宿主机通过 `http://localhost:8000` 访问 API，前端项目也能正常连接。

---

## 目录结构

```
backend/cpp/
├── src/
│   ├── cuda/             # CUDA Kernel 源码
│   │   └── postprocess.cu # YOLO 解码与过滤 Kernel
│   ├── data/             # 模型与测试数据
│   │   ├── models/       # 多模型目录 (model, model2, ...)
│   │   └── calib/        # INT8 校准数据集
│   ├── Int8Calibrator.cpp # INT8 校准器实现
│   ├── ObjectDetector.cpp # 核心推理类 (Engine构建、推理流程)
│   └── main.cpp          # 程序入口与测试
├── include/              # 头文件
│   ├── cuda_utils.hpp    # CUDA 辅助函数与结构体
│   ├── Int8Calibrator.hpp
│   └── ObjectDetector.hpp
├── tests/                # 单元测试
├── build/                # 构建目录
├── CMakeLists.txt        # CMake配置
└── Dockerfile            # Docker构建文件
```
