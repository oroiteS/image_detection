#pragma once
#include <opencv2/opencv.hpp>
// utils.hpp
inline cv::Mat load_image(const std::string& path, int target = 640)
{
    cv::Mat img = cv::imread(path);
    if (img.empty())
        throw std::runtime_error("Failed to load image: " + path);

    // 1. 转换为 RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 2. Resize (保持比例的 Letterbox 处理略，这里先强制缩放方便跑通)
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target, target));

    // 3. 归一化 + HWC 转 CHW
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 这里的 Mat 依然是 HWC，我们需要在 main.cpp 里或者这里手动重排
    // 为了简单，我们使用 OpenCV 的 dnn::blobFromImage 来完成 HWC->CHW 的重排
    // 注意：blobFromImage 返回的是 NCHW
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(target, target), cv::Scalar(), true, false);

    // blob 是连续的内存，可以直接拷贝
    return blob;
}