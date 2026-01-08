cmake_minimum_required(VERSION 3.20)

# 强制使用正确的编译器版本（根据你的需要选择）
# 如果你有VS 2022的14.3版本，建议使用v143
set(CMAKE_GENERATOR_TOOLSET "v143" CACHE STRING "Toolset to use")

project(PowerInspection LANGUAGES CXX CUDA)

# ========== 字符编码设置 ==========
# 设置源文件编码为UTF-8（解决中文乱码）
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 设置Windows SDK版本（避免版本冲突）
set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "10.0" CACHE STRING "Windows SDK Version")

# ========== 设置CUDA架构 ==========
# RTX 3060的计算能力是8.6，所以设置为86
set(CMAKE_CUDA_ARCHITECTURES "86")
message(STATUS "CUDA Architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
# [建议] 设置 CUDA 架构，"native" 会自动检测你当前显卡的架构
# 如果编译报错，可以手动指定，例如 set(CMAKE_CUDA_ARCHITECTURES 75) 对应 RTX 20系
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#     set(CMAKE_CUDA_ARCHITECTURES "native")
# endif()

# ========== C++标准设置 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ========== 设置编译类型 ==========
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ========== 查找依赖包 ==========
# 1. 首先设置OpenCV路径，然后查找
set(OpenCV_DIR "D:/Code/Others/opencv/build")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 2. 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# ========== 配置TensorRT ==========
set(TRT_DIR "C:/Program Files/TensorRT-8.6.1.6" CACHE PATH "TensorRT directory")

# 检查TensorRT路径是否存在
if(NOT EXISTS "${TRT_DIR}")
    message(FATAL_ERROR "TensorRT directory not found: ${TRT_DIR}")
endif()

message(STATUS "TensorRT directory: ${TRT_DIR}")

# 包含TensorRT头文件
include_directories(${TRT_DIR}/include)

# 添加TensorRT库目录（32位和64位都要考虑）
link_directories(
    ${TRT_DIR}/lib
    ${TRT_DIR}/lib/x64
)

# ========== 收集源代码 ==========
# 显式列出所有源文件，避免使用GLOB（更可靠）
set(CORE_SOURCES
    src/main.cpp
    src/engine.cpp
    # src/preprocess.cu
)

# 检查源文件是否存在
foreach(source_file ${CORE_SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file})
        message(WARNING "Source file not found: ${source_file}")
    endif()
endforeach()

# ========== 包含目录设置 ==========
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# ========== 创建可执行文件 ==========
add_executable(inference_demo ${CORE_SOURCES})

# ========== CUDA特定设置 ==========
# 启用可分离编译
set_target_properties(inference_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ========== 链接库 ==========
target_link_libraries(inference_demo
    PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudadevrt  # 添加CUDA设备运行时库
)

# ========== Windows特定设置 ==========
if(WIN32)
    # 添加Windows系统库
    target_link_libraries(inference_demo PRIVATE
        ws2_32
        advapi32
    )
    
    # 设置运行时库（MT/MTd for 静态，MD/MDd for 动态）
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(inference_demo PRIVATE /MDd)
    else()
        target_compile_options(inference_demo PRIVATE /MD)
    endif()
endif()

# ========== 输出信息 ==========
message(STATUS "======================================")
message(STATUS "Build Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT Path: ${TRT_DIR}")
message(STATUS "  Source Files: ${CORE_SOURCES}")
message(STATUS "======================================")

# ========== 安装后构建命令（可选） ==========
# 复制必要的DLL文件到输出目录
add_custom_command(TARGET inference_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TRT_DIR}/lib/nvinfer.dll"
    $<TARGET_FILE_DIR:inference_demo>
    COMMENT "Copying TensorRT DLLs..."
)

# 复制OpenCV DLLs（根据你的OpenCV安装路径调整）
if(WIN32)
    add_custom_command(TARGET inference_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/Code/Others/opencv/build/x64/vc16/bin/opencv_world480.dll"
        $<TARGET_FILE_DIR:inference_demo>
        COMMENT "Copying OpenCV DLLs..."
    )
endif()
