#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "ObjectDetector.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::cout << "Image Detection Backend (Multi-Batch)" << std::endl;

    // 默认模型索引
    int modelIndex = 1;
    if (argc > 1) {
        try {
            modelIndex = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid model index provided. Using default: " << modelIndex << std::endl;
        }
    }

    // 构建模型路径
    // 假设模型目录结构为: src/data/models/model{index}/best.onnx
    // 例如: src/data/models/model/best.onnx (对应 index 1)
    //       src/data/models/model2/best.onnx (对应 index 2)
    std::string modelDirName = (modelIndex == 1) ? "model" : "model" + std::to_string(modelIndex);
    std::string baseDir = "../src/data/models/" + modelDirName;
    std::string onnxPath = baseDir + "/best.onnx";

    // Engine 文件保存在同级目录下
    std::string enginePath = baseDir + "/best_int8_batch.engine";
    std::string calibDataDir = "../src/data/calib";
    std::string imagePath = "../src/data/test.jpg";

    std::cout << "Selected Model Index: " << modelIndex << std::endl;
    std::cout << "ONNX Path: " << onnxPath << std::endl;
    std::cout << "Engine Path: " << enginePath << std::endl;

    // 检查 ONNX 文件是否存在
    if (!fs::exists(onnxPath)) {
        std::cerr << "Error: ONNX model not found at " << onnxPath << std::endl;
        return -1;
    }

    bool useInt8 = false;
    if (fs::exists(calibDataDir) && !fs::is_empty(calibDataDir)) {
        useInt8 = true;
        std::cout << "Calibration data found. Enabling INT8 quantization." << std::endl;
    } else {
        std::cout << "No calibration data found at " << calibDataDir << ". Using FP16/FP32." << std::endl;
        enginePath = baseDir + "/best_fp16_batch.engine"; // 如果没有校准数据，使用 FP16 文件名
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
