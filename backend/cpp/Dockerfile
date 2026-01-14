# 基础镜像：NVIDIA TensorRT (包含 CUDA, cuDNN, TensorRT)
# 这是一个基于 Ubuntu 的镜像，提供了稳定的开发环境
FROM nvcr.io/nvidia/tensorrt:24.03-py3

# 避免安装过程中的交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云镜像源以加速下载 (针对 Ubuntu 22.04 Jammy)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更新包列表并安装依赖
# - cmake: 构建工具
# - libopencv-dev: OpenCV 开发库
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopencv-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace/project

# 默认命令
CMD ["/bin/bash"]
