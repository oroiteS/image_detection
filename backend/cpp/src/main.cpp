#include "engine.hpp"
#include "utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    // 强制刷新缓冲区并输出
    std::cout << "--- PROGRAM START ---" << std::endl;
    std::cerr << "--- ERROR CHANNEL TEST ---" << std::endl;
    fflush(stdout);
    try {
        std::string model = "../models/best.onnx";
        std::string img_path = "../test.jpg";

        TRTInferenceEngine engine(model);

        cv::Mat img = load_image(img_path);

        // 检查图像是否正确加载
        if (img.empty()) {
            std::cerr << "Error: Failed to load image: " << img_path << std::endl;
            return -1;
        }

        // 注意：OpenCV图像的存储顺序是BGR，且是uint8类型
        // 你需要将图像转换为模型所需的格式（通常是归一化的RGB）
        std::cout << "Image size: " << img.cols << "x" << img.rows
                  << ", channels: " << img.channels() << std::endl;

        // 这里需要根据你的模型预处理要求调整
        // 假设模型需要3x640x640的归一化RGB图像
        std::vector<float> input(img.total() * img.channels());

        // 将图像数据转换为float并归一化
        for (int i = 0; i < img.total() * img.channels(); ++i) {
            input[i] = img.data[i] / 255.0f;
        }

        auto output = engine.infer(input.data(), 1);

        std::cout << "Inference OK, output size = " << output.size() << std::endl;

        cv::imshow("input", img);
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}