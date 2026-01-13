cmake_minimum_required(VERSION 3.15)
project(ImageDetection VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_DIR}")

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA found: ${CUDA_TOOLKIT_ROOT_DIR}")

# TensorRT Configuration
if(DEFINED ENV{TENSORRT_ROOT} AND NOT DEFINED TENSORRT_ROOT)
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

if(DEFINED TENSORRT_ROOT)
    include_directories(${TENSORRT_ROOT}/include)
    link_directories(${TENSORRT_ROOT}/lib)
endif()

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64)
find_library(NVPARSERS_LIB nvparsers HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64)

if (NOT NVINFER_LIB OR NOT NVPARSERS_LIB OR NOT NVONNXPARSER_LIB)
    message(WARNING "TensorRT libraries not found.")
endif()

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE CXX_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")

# Create executable
add_executable(${PROJECT_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${NVINFER_LIB}
    ${NVPARSERS_LIB}
    ${NVONNXPARSER_LIB}
)
