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
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ ..
make
```

## 编译器要求

- g++ 支持 C++17 或更高版本
- CMake 3.15+
