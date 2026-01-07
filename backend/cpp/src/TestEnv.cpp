#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>

int main()
{
    SetConsoleOutputCP(65001); // 解决乱码

    std::cout << "--- 电力巡检系统环境检测 ---" << std::endl;

    // 1. GPU 硬件检测
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA 初始化失败: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // 2. 显存压力简单测试
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "GPU 显存总量: " << total_byte / 1024 / 1024 << " MB" << std::endl;
    std::cout << "可用显存: " << free_byte / 1024 / 1024 << " MB" << std::endl;

    // 3. OpenCV 测试
    cv::Mat test_img(512, 512, CV_8UC3, cv::Scalar(0, 255, 0));
    cv::putText(test_img, "Environment Ready!", cv::Point(50, 250),
        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);

    std::cout << "环境检测完成，即将弹出测试窗口..." << std::endl;
    cv::imshow("Check", test_img);
    cv::waitKey(0);

    return 0;
}