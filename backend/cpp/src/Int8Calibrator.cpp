#include "Int8Calibrator.hpp"
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <iterator> // 修复: 缺少这个头文件

namespace fs = std::filesystem;

Int8Calibrator::Int8Calibrator(int batchSize, const std::string& calibDataDir,
                               const std::string& calibCacheFileName,
                               int inputW, int inputH)
    : m_batchSize(batchSize), m_calibCacheFileName(calibCacheFileName),
      m_inputW(inputW), m_inputH(inputH)
{
    // 计算输入 Tensor 大小: Batch * 3 * H * W * sizeof(float)
    m_inputSize = m_batchSize * 3 * m_inputH * m_inputW * sizeof(float);

    // 分配 GPU 和 Host 内存
    cudaMalloc(&m_deviceInput, m_inputSize);
    m_hostInput.resize(m_inputSize / sizeof(float));

    // 读取校准图片路径
    if (fs::exists(calibDataDir)) {
        for (const auto& entry : fs::directory_iterator(calibDataDir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // 简单的后缀检查
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    m_imagePaths.push_back(entry.path().string());
                }
            }
        }
    }

    std::cout << "[Int8Calibrator] Found " << m_imagePaths.size() << " images for calibration." << std::endl;
    if (m_imagePaths.empty()) {
        std::cerr << "[Int8Calibrator] Warning: No images found in " << calibDataDir << std::endl;
    }
}

Int8Calibrator::~Int8Calibrator() {
    if (m_deviceInput) cudaFree(m_deviceInput);
}

int Int8Calibrator::getBatchSize() const noexcept {
    return m_batchSize;
}

bool Int8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (m_imgIdx + m_batchSize > m_imagePaths.size()) {
        return false; // 数据读完了
    }

    // 准备一个 Batch 的数据
    float* ptr = m_hostInput.data();
    // 单张图片的大小 (元素个数)
    size_t volImg = 3 * m_inputH * m_inputW;

    for (int i = 0; i < m_batchSize; ++i) {
        std::string path = m_imagePaths[m_imgIdx + i];
        // std::cout << "[Int8Calibrator] Processing: " << path << std::endl;

        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << path << std::endl;
            continue;
        }

        // 执行预处理，写入到 host buffer 的对应位置
        preprocess(img, ptr + i * volImg);
    }

    // 拷贝到 GPU
    cudaMemcpy(m_deviceInput, m_hostInput.data(), m_inputSize, cudaMemcpyHostToDevice);

    // 绑定输入 Tensor 地址
    // 注意：这里假设只有一个输入，且 bindings[0] 是输入
    // 更严谨的做法是检查 names[] 数组
    bindings[0] = m_deviceInput;

    m_imgIdx += m_batchSize;
    return true;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    m_calibrationCache.clear();
    std::ifstream input(m_calibCacheFileName, std::ios::binary);
    input >> std::noskipws;
    if (input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibrationCache));
    }
    length = m_calibrationCache.size();
    return length ? m_calibrationCache.data() : nullptr;
}

void Int8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream output(m_calibCacheFileName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

// 必须与 ObjectDetector::preprocess 保持一致
void Int8Calibrator::preprocess(const cv::Mat& image, float* buffer) {
    // Letterbox resize
    float scale = std::min((float)m_inputW / image.cols, (float)m_inputH / image.rows);
    int newW = image.cols * scale;
    int newH = image.rows * scale;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));

    cv::Mat canvas(m_inputH, m_inputW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(0, 0, newW, newH)));

    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);

    int channelSize = m_inputH * m_inputW;
    memcpy(buffer, channels[0].data, channelSize * sizeof(float));
    memcpy(buffer + channelSize, channels[1].data, channelSize * sizeof(float));
    memcpy(buffer + 2 * channelSize, channels[2].data, channelSize * sizeof(float));
}
