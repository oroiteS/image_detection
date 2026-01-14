#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "cuda_utils.hpp"

#define CHECK_CUDA(status) \
    do { \
        if (status != 0) { \
            std::cerr << "Cuda failure: " << status << " at line " << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();

    bool init(const std::string& onnxPath, const std::string& enginePath,
              bool useInt8 = false, const std::string& calibDataDir = "");

    // 修改：支持多 Batch 推理
    std::vector<std::vector<Detection>> detect(const std::vector<cv::Mat>& images, float confThreshold = 0.25f, float nmsThreshold = 0.45f);

private:
    struct InferDeleter {
        template <typename T>
        void operator()(T* obj) const {
            if (obj) delete obj;
        }
    };

    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    cudaStream_t m_cudaStream = nullptr;

    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;
    void* m_cudaInputBuffer = nullptr;
    void* m_cudaOutputBuffer = nullptr;
    void* m_cudaDecodeBuffer = nullptr;

    size_t m_inputSize = 0;
    size_t m_outputSize = 0;

    // 记录当前 Engine 支持的最大 Batch Size
    int m_maxBatchSize = 1;

    bool buildEngine(const std::string& onnxPath, const std::string& enginePath,
                     bool useInt8, const std::string& calibDataDir);
    bool loadEngine(const std::string& enginePath);

    // 修改：预处理多张图片
    void preprocess(const std::vector<cv::Mat>& images, float* gpuInput, const nvinfer1::Dims& dims);

    // 修改：后处理多张图片
    std::vector<std::vector<Detection>> postprocess(const BatchOutputBuffer& gpuOutput, int batchSize, float nmsThreshold);

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kINFO) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    } m_logger;
};
