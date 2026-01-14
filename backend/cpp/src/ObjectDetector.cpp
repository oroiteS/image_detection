#include "ObjectDetector.hpp"
#include "Int8Calibrator.hpp"
#include <NvOnnxParser.h>
#include <algorithm>
#include <numeric>

ObjectDetector::ObjectDetector() {
    CHECK_CUDA(cudaStreamCreate(&m_cudaStream));
    m_maxBatchSize = MAX_BATCH_SIZE;
}

ObjectDetector::~ObjectDetector() {
    if (m_cudaInputBuffer) cudaFree(m_cudaInputBuffer);
    if (m_cudaOutputBuffer) cudaFree(m_cudaOutputBuffer);
    if (m_cudaDecodeBuffer) cudaFree(m_cudaDecodeBuffer);
    if (m_cudaStream) cudaStreamDestroy(m_cudaStream);
}

bool ObjectDetector::init(const std::string& onnxPath, const std::string& enginePath,
                          bool useInt8, const std::string& calibDataDir) {
    std::ifstream f(enginePath);
    bool engineExists = f.good();
    f.close();

    if (engineExists) {
        std::cout << "Loading existing engine from: " << enginePath << std::endl;
        if (!loadEngine(enginePath)) {
            std::cerr << "Failed to load engine. Rebuilding..." << std::endl;
            if (!buildEngine(onnxPath, enginePath, useInt8, calibDataDir)) return false;
        }
    } else {
        std::cout << "Engine not found. Building from ONNX: " << onnxPath << std::endl;
        if (!buildEngine(onnxPath, enginePath, useInt8, calibDataDir)) return false;
    }

    m_inputDims = m_engine->getTensorShape(m_engine->getIOTensorName(0));
    m_outputDims = m_engine->getTensorShape(m_engine->getIOTensorName(1));

    m_inputSize = m_maxBatchSize;
    for(int i=0; i<m_inputDims.nbDims; ++i) {
        int dim = m_inputDims.d[i];
        if (dim < 0) dim = (i == 0) ? 1 : 640;
        m_inputSize *= dim;
    }
    m_inputSize *= sizeof(float);

    size_t outputElements = m_maxBatchSize;
    for(int i=0; i<m_outputDims.nbDims; ++i) {
        int dim = m_outputDims.d[i];
        if (dim < 0) dim = 1;
        outputElements *= dim;
    }
    if (outputElements < 100 * m_maxBatchSize) outputElements = m_maxBatchSize * 84 * 8400;
    m_outputSize = outputElements * sizeof(float);

    CHECK_CUDA(cudaMalloc(&m_cudaInputBuffer, m_inputSize));
    CHECK_CUDA(cudaMalloc(&m_cudaOutputBuffer, m_outputSize));
    CHECK_CUDA(cudaMalloc(&m_cudaDecodeBuffer, sizeof(BatchOutputBuffer)));

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) return false;

    m_context->setTensorAddress(m_engine->getIOTensorName(0), m_cudaInputBuffer);
    m_context->setTensorAddress(m_engine->getIOTensorName(1), m_cudaOutputBuffer);

    return true;
}

bool ObjectDetector::buildEngine(const std::string& onnxPath, const std::string& enginePath,
                                 bool useInt8, const std::string& calibDataDir) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) return false;

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) return false;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return false;
    }

    auto input = network->getInput(0);

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 640, 640});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{m_maxBatchSize > 1 ? m_maxBatchSize/2 : 1, 3, 640, 640});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{m_maxBatchSize, 3, 640, 640});
    config->addOptimizationProfile(profile);

    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    std::unique_ptr<Int8Calibrator> calibrator;
    if (useInt8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);

        auto calibProfile = builder->createOptimizationProfile();
        calibProfile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 640, 640});
        calibProfile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 640, 640});
        calibProfile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 640, 640});
        config->setCalibrationProfile(calibProfile);

        calibrator = std::make_unique<Int8Calibrator>(1, calibDataDir, enginePath + ".calib", 640, 640);
        config->setInt8Calibrator(calibrator.get());

        std::cout << "Applying Layer Fallback strategy for INT8..." << std::endl;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto layer = network->getLayer(i);
            std::string layerName = layer->getName();

            // 关键修复：跳过 Shape Layer 和 INT32 输出层
            if (layer->getType() == nvinfer1::LayerType::kSHAPE ||
                layer->getType() == nvinfer1::LayerType::kCONSTANT) {
                continue;
            }

            // 检查输出类型
            bool isInt32Output = false;
            for (int j = 0; j < layer->getNbOutputs(); ++j) {
                if (layer->getOutput(j)->getType() == nvinfer1::DataType::kINT32) {
                    isInt32Output = true;
                    break;
                }
            }
            if (isInt32Output) continue;

            // 策略：如果层名称包含 "detect" 或 "cv2" (YOLO Head 常用名)，或者是输出层的前置层
            // 则强制使用 FP16。
            if (layerName.find("detect") != std::string::npos ||
                layerName.find("cv2") != std::string::npos ||
                layerName.find("cv3") != std::string::npos ||
                layerName.find("dfl") != std::string::npos ||
                layer->getType() == nvinfer1::LayerType::kNON_ZERO) {

                layer->setPrecision(nvinfer1::DataType::kHALF);
                // 显式设置输出类型为 FP16，避免 TensorRT 自动推断错误
                for (int k = 0; k < layer->getNbOutputs(); ++k) {
                    layer->setOutputType(k, nvinfer1::DataType::kHALF);
                }
            }
        }
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) return false;

    std::ofstream outfile(enginePath, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()));

    return true;
}

bool ObjectDetector::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return false;

    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) return false;

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), size));
    return (m_engine != nullptr);
}

std::vector<std::vector<Detection>> ObjectDetector::detect(const std::vector<cv::Mat>& images, float confThreshold, float nmsThreshold) {
    int batchSize = images.size();
    if (batchSize == 0) return {};
    if (batchSize > m_maxBatchSize) return {};

    if (m_inputDims.d[0] == -1) {
        m_context->setInputShape(m_engine->getIOTensorName(0), nvinfer1::Dims4{batchSize, 3, 640, 640});
    }

    size_t currentInputSize = batchSize * 3 * 640 * 640 * sizeof(float);
    std::vector<float> hostInput(currentInputSize / sizeof(float));

    nvinfer1::Dims4 inputDims{batchSize, 3, 640, 640};
    preprocess(images, hostInput.data(), inputDims);

    CHECK_CUDA(cudaMemcpyAsync(m_cudaInputBuffer, hostInput.data(), currentInputSize, cudaMemcpyHostToDevice, m_cudaStream));

    if (!m_context->enqueueV3(m_cudaStream)) {
        std::cerr << "Inference failed." << std::endl;
        return {};
    }

    int inputW = 640;
    int inputH = 640;
    float scale = std::min((float)inputW / images[0].cols, (float)inputH / images[0].rows);

    int numChannels = m_outputDims.d[1];
    int numAnchors = m_outputDims.d[2];
    int numClasses = numChannels - 4;

    launch_yolo_decode(
        (float*)m_cudaOutputBuffer,
        m_cudaDecodeBuffer,
        batchSize,
        numAnchors, numChannels, numClasses,
        confThreshold, scale, inputW, inputH,
        m_cudaStream
    );

    BatchOutputBuffer hostOutput;
    CHECK_CUDA(cudaMemcpyAsync(&hostOutput, m_cudaDecodeBuffer, sizeof(BatchOutputBuffer), cudaMemcpyDeviceToHost, m_cudaStream));

    cudaStreamSynchronize(m_cudaStream);

    return postprocess(hostOutput, batchSize, nmsThreshold);
}

void ObjectDetector::preprocess(const std::vector<cv::Mat>& images, float* input, const nvinfer1::Dims& dims) {
    int inputH = dims.d[2];
    int inputW = dims.d[3];
    int channelSize = inputH * inputW;
    int imageSize = 3 * channelSize;

    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Mat& image = images[i];
        float* currentInput = input + i * imageSize;

        float scale = std::min((float)inputW / image.cols, (float)inputH / image.rows);
        int newW = image.cols * scale;
        int newH = image.rows * scale;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(newW, newH));

        cv::Mat canvas(inputH, inputW, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(0, 0, newW, newH)));

        cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
        canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(canvas, channels);

        memcpy(currentInput, channels[0].data, channelSize * sizeof(float));
        memcpy(currentInput + channelSize, channels[1].data, channelSize * sizeof(float));
        memcpy(currentInput + 2 * channelSize, channels[2].data, channelSize * sizeof(float));
    }
}

std::vector<std::vector<Detection>> ObjectDetector::postprocess(const BatchOutputBuffer& gpuOutput, int batchSize, float nmsThreshold) {
    std::vector<std::vector<Detection>> batchDetections;

    for (int b = 0; b < batchSize; ++b) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        int count = std::min(gpuOutput.counts[b], MAX_DETECTIONS);

        for (int i = 0; i < count; ++i) {
            const auto& det = gpuOutput.detections[b * MAX_DETECTIONS + i];

            int x = static_cast<int>(det.x1);
            int y = static_cast<int>(det.y1);
            int w = static_cast<int>(det.x2 - det.x1);
            int h = static_cast<int>(det.y2 - det.y1);

            boxes.push_back(cv::Rect(x, y, w, h));
            confidences.push_back(det.confidence);
            classIds.push_back(det.classId);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.0f, nmsThreshold, indices);

        std::vector<Detection> detections;
        for (int idx : indices) {
            Detection det;
            det.classId = classIds[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            detections.push_back(det);
        }
        batchDetections.push_back(detections);
    }

    return batchDetections;
}
