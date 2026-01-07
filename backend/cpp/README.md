# C++ Backend

电力巡检图像智能检测系统的C++后端部分。

## 目录结构

```
backend/cpp/
├── src/              # 源代码
├── include/          # 头文件
├── tests/            # 单元测试
├── build/            # 构建目录
└── CMakeLists.txt    # CMake配置
```

## 构建

```bash
cd backend/cpp
```

使用CMAKE进行build，之后执行env_check环境检测：

```bash
./env_check.exe
```

- MSVC C++17
- CMake 3.20+
- CUDA Toolkit 11.8（根据自己的显卡情况选择）
- TensorRT 8.6 GA
- OpenCV 4.8.0 / 4.9.0

这里给出windows下的安装地址

1. **Visual Studio 2022 Community**: [下载地址](https://visualstudio.microsoft.com/zh-hans/vs/)

2. **CMake**: [下载地址](https://cmake.org/download/)

3. **CUDA Toolkit 11.8**: [下载地址](https://developer.nvidia.com/cuda-11-8-0-download-archive)

4. **TensorRT 8.6 GA**: [下载地址](https://developer.nvidia.com/tensorrt)

5. **OpenCV 4.8.0 / 4.9.0**: [下载地址](https://opencv.org/releases/)
