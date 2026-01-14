#include "Int8Calibrator.hpp"
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <set>

namespace fs = std::filesystem;

Int8Calibrator::Int8Calibrator(int batchSize, const std::string& calibDataDir,
                               const std::string& calibCacheFileName,
                               int inputW, int inputH)
    : m_batchSize(batchSize), m_calibCacheFileName(calibCacheFileName),
      m_inputW(inputW), m_inputH(inputH)
{
    m_inputSize = m_batchSize * 3 * m_inputH * m_inputW * sizeof(float);

    cudaMalloc(&m_deviceInput, m_inputSize);
    m_hostInput.resize(m_inputSize / sizeof(float));

    // 支持的后缀 (小写)
    std::set<std::string> validExts = {".jpg", ".jpeg", ".png", ".bmp"};

    if (fs::exists(calibDataDir)) {
        for (const auto& entry : fs::directory_iterator(calibDataDir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // 转小写
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (validExts.count(ext)) {
                    m_imagePaths.push_back(entry.path().string());
                }
            }
        }
    }

    std::cout << "[Int8Calibrator] Found " << m_imagePaths.size() << " images for calibration." << std::endl;
    if (m_imagePaths.empty()) {
        std::cerr << "[Int8Calibrator] Warning: No images found in " << calibDataDir << std::endl;
    }

    // 随机打乱，避免只校准某一类数据
    std::random_shuffle(m_imagePaths.begin(), m_imagePaths.end());
}

Int8Calibrator::~Int8Calibrator() {
    if (m_deviceInput) cudaFree(m_deviceInput);
}

int Int8Calibrator::getBatchSize() const noexcept {
    return m_batchSize;
}

bool Int8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (m_imgIdx + m_batchSize > m_imagePaths.size()) {
        return false;
    }

    float* ptr = m_hostInput.data();
    size_t volImg = 3 * m_inputH * m_inputW;

    for (int i = 0; i < m_batchSize; ++i) {
        std::string path = m_imagePaths[m_imgIdx + i];
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << path << std::endl;
            // 填充黑色，避免崩溃
            img = cv::Mat::zeros(m_inputH, m_inputW, CV_8UC3);
        }
        preprocess(img, ptr + i * volImg);
    }

    cudaMemcpy(m_deviceInput, m_hostInput.data(), m_inputSize, cudaMemcpyHostToDevice);
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

void Int8Calibrator::preprocess(const cv::Mat& image, float* buffer) {
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
