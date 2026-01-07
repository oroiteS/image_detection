cmake_minimum_required(VERSION 3.15)
project(ImageDetection VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 头文件目录
include_directories(include)

# 可执行文件或库
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# 测试
enable_testing()
add_subdirectory(tests)
