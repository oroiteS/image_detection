cmake_minimum_required(VERSION 3.14)
project(ImageDetection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 关键修复：设置 CUDA 架构
# 75=Turing(20系), 80=Ampere(30系), 86=Ampere(30系), 89=Ada(40系)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

# ==========================================
# 依赖查找
# ==========================================
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_library(NVINFER_LIB nvinfer)
find_library(NVONNXPARSER_LIB nvonnxparser)

# ==========================================
# Pybind11 集成
# ==========================================
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# ==========================================
# 核心源文件列表
# ==========================================
set(CORE_SOURCES
    src/ObjectDetector.cpp
    src/Int8Calibrator.cpp
    src/cuda/postprocess.cu
)

# ==========================================
# 可执行文件 (C++ 测试用)
# ==========================================
add_executable(ImageDetection src/main.cpp ${CORE_SOURCES})

target_include_directories(ImageDetection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ImageDetection PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
)

# ==========================================
# Python 扩展模块 (Pybind11)
# ==========================================
pybind11_add_module(image_detection_cpp src/bindings.cpp ${CORE_SOURCES})

target_include_directories(image_detection_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(image_detection_cpp PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
)

set_target_properties(image_detection_cpp PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
