#pragma once

#include <NvInfer.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

// 使用 EntropyCalibrator2，这是目前最推荐用于 CNN 的校准算法
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(int batchSize, const std::string& calibDataDir,
                   const std::string& calibCacheFileName,
                   int inputW, int inputH);

    virtual ~Int8Calibrator();

    // TensorRT 调用此方法获取 Batch Size
    int getBatchSize() const noexcept override;

    // TensorRT 调用此方法获取数据
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    // 读取校准缓存（避免每次都重新校准）
    const void* readCalibrationCache(size_t& length) noexcept override;

    // 写入校准缓存
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int m_batchSize;
    int m_inputW;
    int m_inputH;
    size_t m_inputSize;

    std::string m_calibCacheFileName;
    std::vector<std::string> m_imagePaths;
    size_t m_imgIdx = 0;

    void* m_deviceInput = nullptr;
    std::vector<float> m_hostInput;
    std::vector<char> m_calibrationCache;

    // 预处理函数 (必须与推理时的预处理完全一致)
    void preprocess(const cv::Mat& img, float* buffer);
};
