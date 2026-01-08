#pragma once
#include <NvInfer.h>
#include <iostream>

// logging.hpp
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // 修改为 kINFO，这样能看到详细的构建过程
        if (severity <= Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};