#include "engine.hpp"
#include "logging.hpp"
#include <NvOnnxParser.h>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sys/stat.h>
#include <vector>
inline bool exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

static TRTLogger gLogger;

TRTInferenceEngine::TRTInferenceEngine(const std::string& path)
{
    build_from_onnx(path);
}

TRTInferenceEngine::~TRTInferenceEngine()
{
    cudaFree(device_input_);
    cudaFree(device_output_);
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

void TRTInferenceEngine::build_from_onnx(const std::string& path)
{
    using namespace nvinfer1;

    // 1. 检查文件是否存在
    if (!exists(path)) {
        throw std::runtime_error("Error: ONNX model file not found at: " + path);
    }
    std::cout << "[INFO] Loading ONNX model from: " << path << std::endl;

    IBuilder* builder = createInferBuilder(gLogger);
    // Explicit Batch 是 YOLOv5/v8/v11 必须的
    const auto explicitBatch = 1U << (int)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, gLogger);

    // 2. 解析 ONNX (增加错误检查)
    std::ifstream ifs(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    if (!parser->parse(data.data(), data.size())) {
        std::cerr << "[Error] Failed to parse ONNX file." << std::endl;
        // 打印具体哪里解析失败
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        throw std::runtime_error("ONNX Parsing Failed");
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    // 注意：如果显卡不支持 FP16 (如很老的卡)，这里会报错，建议先注释掉或检查硬件
    if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "[INFO] FP16 mode enabled." << std::endl;
    }

    std::cout << "[INFO] Building TensorRT Engine... (This may take a while)" << std::endl;

    // 3. 构建引擎 (增加空指针检查)
    // 注意：TensorRT 8.x+ 建议使用 buildSerializedNetwork，这里沿用你的旧 API 但增加检查
    engine_ = builder->buildEngineWithConfig(*network, *config);

    if (!engine_) {
        throw std::runtime_error("Error: Failed to build TensorRT Engine. Check network architecture.");
    }
    std::cout << "[INFO] Engine built successfully!" << std::endl;

    runtime_ = createInferRuntime(gLogger);
    context_ = engine_->createExecutionContext();

    // 4. 检查输入输出维度 (防止越界)
    // 假设只有一个输入和一个输出
    auto input_dims = engine_->getBindingDimensions(0);
    auto output_dims = engine_->getBindingDimensions(1);

    // 简单的体积计算 (假设是静态 Batch)
    auto volume = [](const Dims& d) {
        int v = 1;
        for (int i = 0; i < d.nbDims; i++)
            v *= (d.d[i] < 0 ? 1 : d.d[i]); // 处理动态维度 -1
        return v;
    };

    input_size_ = volume(input_dims);
    output_size_ = volume(output_dims);

    std::cout << "[INFO] Input Size: " << input_size_ << " floats" << std::endl;
    std::cout << "[INFO] Output Size: " << output_size_ << " floats" << std::endl;

    cudaMalloc(&device_input_, input_size_ * sizeof(float));
    cudaMalloc(&device_output_, output_size_ * sizeof(float));

    // 清理资源
    parser->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();
}

// 在 engine.cpp 的 build_from_onnx 函数后添加以下代码：

std::vector<float> TRTInferenceEngine::infer(const float* input, int batch_size)
{
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 1. 将输入数据复制到GPU
    cudaMemcpyAsync(device_input_, input, input_size_ * sizeof(float) * batch_size,
        cudaMemcpyHostToDevice, stream);

    // 2. 绑定输入输出
    void* bindings[2] = { device_input_, device_output_ };

    // 3. 执行推理
    bool status = context_->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        throw std::runtime_error("Failed to execute inference");
    }

    // 4. 将输出从GPU复制回CPU
    std::vector<float> host_output(output_size_ * batch_size);
    cudaMemcpyAsync(host_output.data(), device_output_,
        output_size_ * sizeof(float) * batch_size,
        cudaMemcpyDeviceToHost, stream);

    // 5. 同步流
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return host_output;
}
