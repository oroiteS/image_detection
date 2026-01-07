cmake_minimum_required(VERSION 3.20)
project(PowerInspection)

set(CMAKE_CXX_STANDARD 17)

# --- 自动寻找 CUDA ---
find_package(CUDAToolkit REQUIRED)

# --- 手动配置 OpenCV (填入你解压的 build 目录) ---
# 尝试指向这个更深层的路径（vc16 对应 VS2019/2022）
set(OpenCV_DIR "D:/Code/Others/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# --- 手动配置 TensorRT (填入你解压的目录) ---
set(TRT_DIR "C:/Program Files/TensorRT-8.6.1.6")
set(TRT_INCLUDE_DIRS "${TRT_DIR}/include")
set(TRT_LIBS "${TRT_DIR}/lib/nvinfer.lib" "${TRT_DIR}/lib/nvonnxparser.lib")

# 包含头文件
include_directories(
    include 
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS} 
    ${TRT_INCLUDE_DIRS}
)

# 生成可执行文件
add_executable(inference_test src/main.cpp)

# 链接库
target_link_libraries(inference_test 
    ${OpenCV_LIBS} 
    CUDA::cudart 
    ${TRT_LIBS}
)