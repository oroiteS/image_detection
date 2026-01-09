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

## 环境配置 (推荐：Docker)

由于您的宿主机 (Fedora 43) 的 GCC 版本 (15) 过新，与 CUDA 12.6 不兼容，且无法轻松安装旧版 GCC，**强烈建议使用 Docker 进行开发和构建**。

### 1. 安装与配置 Docker (已完成)

### 2. 使用提供的 Dockerfile (已构建)

镜像名称: `image-detection-backend`

**运行容器：**

**关键更新：**
1.  对于 Fedora 用户，挂载卷时必须添加 `:z` 选项 (SELinux)。
2.  使用 CDI 语法 `--device=nvidia.com/gpu=all` 启用 GPU。
3.  如果遇到权限错误，请使用 `sudo` 或确保用户组权限已生效。

```bash
# 在 backend/cpp 目录下运行
sudo docker run --device=nvidia.com/gpu=all -it --rm -v $(pwd):/workspace/project:z image-detection-backend
```

### 3. 在容器内构建

进入容器后，执行以下命令：

```bash
# 1. 清理旧的构建目录 (如果存在)
rm -rf build

# 2. 创建并进入构建目录
mkdir -p build && cd build

# 3. 运行 CMake
cmake ..

# 4. 编译
make
```

### 4. 运行测试

```bash
./ImageDetection
```

程序将自动执行以下流程：
1.  检查 `best_int8_batch.engine` 是否存在。
2.  如果不存在，加载 `src/data/calib` 下的图片进行 INT8 校准。
3.  构建支持多 Batch 的 TensorRT Engine。
4.  执行 Batch=4 的推理测试，并输出检测结果。

## 启用 Docker GPU 支持 (已完成)

您已成功生成 CDI 规范 (`/etc/cdi/nvidia.yaml`) 并验证了 GPU 支持。

**验证 GPU 支持：**
```bash
sudo docker run --rm --device=nvidia.com/gpu=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -benchmark
```

## 目录结构

```
backend/cpp/
├── src/
│   ├── cuda/             # CUDA Kernel 源码
│   │   └── postprocess.cu # YOLO 解码与过滤 Kernel
│   ├── data/             # 模型与测试数据
│   │   ├── model/        # ONNX 模型
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
