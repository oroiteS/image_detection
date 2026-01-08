# C++ Backend

电力巡检图像智能检测系统的C++后端部分。

## 目录结构

```
backend/cpp/
├── CMakeLists.txt            # 项目构建配置
├── include/                  # 头文件
│   ├── logging.hpp           # TensorRT 日志类
│   ├── engine.hpp             # 推理引擎类定义
│   └── utils.hpp             # OpenCV 图像处理工具
├── src/                      # 源文件
│   ├── main.cpp              # 最小化测试主程序
│   ├── engine.cpp             # TensorRT 构建与推理实现
│   └── preprocess.cu         # CUDA 预处理/后处理融合算子
└── models/                   # 存放模型
    └── power_inspection.onnx # 你的原始 ONNX 模型
```

## 构建

```bash
cd backend/cpp
```

使用CMAKE进行build，之后执行env_check环境检测：

```bash
./env_check.exe
```

- 2022 MSVC
- CMake 3.20+
- CUDA Toolkit 12.1（根据自己的显卡情况选择）
- TensorRT 8.6 GA
- OpenCV 4.8.0 / 4.9.0

这里给出windows下的安装地址

1. **Visual Studio 2022 Community**: [下载地址](https://visualstudio.microsoft.com/zh-hans/vs/)

2. **CMake**: [下载地址](https://cmake.org/download/)

3. **CUDA Toolkit 11.8**: [下载地址](https://developer.nvidia.com/cuda-11-8-0-download-archive)

4. **TensorRT 8.6 GA**: [下载地址](https://developer.nvidia.com/tensorrt)

5. **OpenCV 4.8.0 / 4.9.0**: [下载地址](https://opencv.org/releases/)
