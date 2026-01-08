#pragma once
#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

class TRTInferenceEngine {
public:
    TRTInferenceEngine(const std::string& onnx_path);
    ~TRTInferenceEngine();

    std::vector<float> infer(const float* input, int batch);

private:
    void build_from_onnx(const std::string& path);

    nvinfer1::ICudaEngine* engine_ {};
    nvinfer1::IExecutionContext* context_ {};
    nvinfer1::IRuntime* runtime_ {};
    void* device_input_ {};
    void* device_output_ {};
    int input_size_ {};
    int output_size_ {};
};
