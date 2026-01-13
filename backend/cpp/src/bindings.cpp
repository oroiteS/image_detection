#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ObjectDetector.hpp"
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// ==========================================
// NumPy <-> cv::Mat 转换器
// ==========================================
cv::Mat numpy_to_mat(py::array_t<unsigned char>& input) {
    py::buffer_info buf = input.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Input image must be a 3-channel image (H, W, C)");
    }

    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.shape[2];

    if (channels != 3) {
        throw std::runtime_error("Input image must have 3 channels (BGR/RGB)");
    }

    cv::Mat mat(rows, cols, CV_8UC3, (unsigned char*)buf.ptr);
    return mat.clone();
}

// ==========================================
// Pybind11 模块定义
// ==========================================
PYBIND11_MODULE(image_detection_cpp, m) {
    m.doc() = "C++ TensorRT Inference Engine for Image Detection";

    // 绑定 Detection 结构体
    py::class_<Detection>(m, "Detection")
        .def_readwrite("box", &Detection::box)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("classId", &Detection::classId)
        // 辅助属性：返回 [x1, y1, x2, y2] 格式，方便 Python 端使用
        .def_property("bbox",
            [](const Detection& d) {
                return std::vector<float>{
                    (float)d.box.x,
                    (float)d.box.y,
                    (float)(d.box.x + d.box.width),
                    (float)(d.box.y + d.box.height)
                };
            },
            nullptr)
        .def_property("conf",
            [](const Detection& d) { return d.confidence; },
            nullptr)
        .def_property("class_id",
            [](const Detection& d) { return d.classId; },
            nullptr);

    // 绑定 cv::Rect
    py::class_<cv::Rect>(m, "Rect")
        .def_readwrite("x", &cv::Rect::x)
        .def_readwrite("y", &cv::Rect::y)
        .def_readwrite("width", &cv::Rect::width)
        .def_readwrite("height", &cv::Rect::height);

    // 绑定 ObjectDetector 类
    py::class_<ObjectDetector>(m, "ObjectDetector")
        .def(py::init<>())
        .def("init", &ObjectDetector::init,
             py::arg("onnx_path"),
             py::arg("engine_path"),
             py::arg("use_int8") = false,
             py::arg("calib_data_dir") = "")
        .def("detect_batch", [](ObjectDetector& self, std::vector<py::array_t<unsigned char>> inputs, float conf, float nms) {
            std::vector<cv::Mat> images;
            for (auto& input : inputs) {
                images.push_back(numpy_to_mat(input));
            }
            return self.detect(images, conf, nms);
        }, py::arg("images"), py::arg("conf")=0.25f, py::arg("nms")=0.45f)
        .def("detect_single", [](ObjectDetector& self, py::array_t<unsigned char> input, float conf, float nms) {
            std::vector<cv::Mat> images;
            images.push_back(numpy_to_mat(input));
            auto results = self.detect(images, conf, nms);
            // detect 返回的是 vector<vector<Detection>> (batch results)
            // 我们取第一个 batch 的结果
            if (results.empty()) return std::vector<Detection>{};
            return results[0];
        }, py::arg("image"), py::arg("conf")=0.25f, py::arg("nms")=0.45f);
}
