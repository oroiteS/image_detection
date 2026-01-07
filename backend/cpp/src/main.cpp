#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h> // 需要包含这个头文件

int main()
{
    SetConsoleOutputCP(65001); // 设置控制台为 UTF-8 编码
    // 1. 测试 OpenCV
    cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
    std::cout << "Starting test..." << std::endl;
    std::cout << "OpenCV 版本: " << CV_VERSION << std::endl;

    // 2. 测试 CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "检测到 CUDA 设备数量: " << deviceCount << std::endl;

    // 3. 测试 TensorRT
    std::cout << "TensorRT 版本: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << std::endl;
    // ... 其他代码 ...
    std::cout << "End of test." << std::endl;
    // system("pause"); // 这一行能强制让窗口停住

    return 0;
}