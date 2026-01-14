# 构建阶段
FROM node:20-slim AS builder

# 设置工作目录
WORKDIR /app

# 1. 配置 npm 镜像源 (解决 corepack/pnpm 下载超时问题)
RUN npm config set registry https://registry.npmmirror.com

# 2. 手动安装 pnpm (替代 corepack，避免网络问题)
RUN npm install -g pnpm

# 复制依赖文件
COPY package.json pnpm-lock.yaml ./

# 3. 配置 pnpm 镜像源并安装依赖
RUN pnpm config set registry https://registry.npmmirror.com && \
    pnpm install

# 复制源代码
COPY . .

# 构建项目
RUN pnpm build

# 运行阶段
FROM nginx:alpine

# 复制构建产物到 Nginx 目录
COPY --from=builder /app/dist /usr/share/nginx/html

# 暴露端口
EXPOSE 80

# 启动 Nginx
CMD ["nginx", "-g", "daemon off;"]
