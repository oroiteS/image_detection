# 使用 NVIDIA TensorRT 基础镜像 (包含 CUDA, cuDNN, TensorRT)
FROM nvcr.io/nvidia/tensorrt:24.03-py3

# 设置工作目录
WORKDIR /app

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 1. 修改 APT 源为阿里云镜像 (解决 apt-get update 连接超时/EOF 问题)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 2. 配置 Python/UV 镜像源 (加速依赖下载)
ENV UV_PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目依赖文件
COPY pyproject.toml uv.lock ./

# 复制源代码
COPY src ./src
COPY README.md ./

# 同步环境
RUN uv sync --frozen

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "python", "src/image_detection/web/api.py"]
