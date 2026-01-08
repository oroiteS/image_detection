# 电力巡检系统 - 前端界面 (Vue3 + TS)

本项目是基于 Vue3、Vite 和 Tailwind CSS 构建的智能检测系统前端，通过 RESTful API 与后端 FastAPI 通信。

## 🛠️ 环境要求

- **Node.js**: 18.x 或更高版本
- **包管理器**: 推荐使用 `pnpm` (也可使用 `npm` 或 `yarn`)

## 🚀 快速部署

### 1. 安装 pnpm (可选)
如果你还没有安装 pnpm，可以通过 npm 全局安装：
```bash
npm install -g pnpm
```

### 2. 安装项目依赖
在 `frontend` 目录下运行以下命令，它会自动安装 `package.json` 中定义的所有依赖（包括 Vue, Vite, Tailwind, Axios 等）：
```bash
pnpm install
```

### 3. 启动开发服务器
```bash
pnpm dev
```
启动后，在浏览器访问 `http://localhost:5173`。

---

## 📦 核心依赖清单 (已包含在 package.json 中)

如果需要手动维护或升级，本项目主要依赖以下库：

- **框架**: `vue` (v3.5+), `vite` (v7.0+)
- **样式**: `tailwindcss`, `autoprefixer`, `postcss`
- **通信**: `axios` (用于调用后端 API)
- **脚本**: `typescript`
- **配置**: `@vue/tsconfig`, `@tsconfig/node24` (解决 TS 环境兼容性)

## ✨ 主要功能

- **实时检测**: 上传巡检图片，秒级获取 AI 识别结果。
- **模型切换**: 支持在界面上直接切换后端不同的 YOLO 训练权重。
- **置信度过滤**: 前端支持通过滑动条实时过滤低置信度的检测结果。
- **响应式设计**: 适配不同尺寸的屏幕显示。
