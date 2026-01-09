#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "ObjectDetector.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::cout << "Image Detection Backend (Multi-Batch)" << std::endl;

    std::string onnxPath = "../src/data/model/best.onnx";
    std::string enginePath = "best_int8_batch.engine"; // 使用新的文件名，避免与单 Batch Engine 冲突
    std::string imagePath = "../src/data/test.jpg";
    std::string calibDataDir = "../src/data/calib";

    bool useInt8 = false;
    if (fs::exists(calibDataDir) && !fs::is_empty(calibDataDir)) {
        useInt8 = true;
        std::cout << "Calibration data found. Enabling INT8 quantization." << std::endl;
    } else {
        std::cout << "No calibration data found at " << calibDataDir << ". Using FP16/FP32." << std::endl;
        enginePath = "best_fp16_batch.engine";
    }

    ObjectDetector detector;

    if (!detector.init(onnxPath, enginePath, useInt8, calibDataDir)) {
        std::cerr << "Failed to initialize detector." << std::endl;
        return -1;
    }
    std::cout << "Detector initialized successfully." << std::endl;

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "No test image found at " << imagePath << ". Using dummy image." << std::endl;
        image = cv::Mat(640, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    }

    // 准备多 Batch 输入 (例如 Batch=4)
    int batchSize = 4;
    std::vector<cv::Mat> batchImages;
    for (int i = 0; i < batchSize; ++i) {
        batchImages.push_back(image.clone());
    }

    std::cout << "Running inference with Batch Size = " << batchSize << "..." << std::endl;
    auto batchResults = detector.detect(batchImages);
    std::cout << "Inference finished." << std::endl;

    for (int b = 0; b < batchSize; ++b) {
        std::cout << "Batch [" << b << "] Detected " << batchResults[b].size() << " objects." << std::endl;
        for (const auto& det : batchResults[b]) {
            std::cout << "  - Class: " << det.classId << ", Conf: " << det.confidence
                      << ", Box: [" << det.box.x << ", " << det.box.y << ", "
                      << det.box.width << ", " << det.box.height << "]" << std::endl;
        }
    }

    return 0;
}
