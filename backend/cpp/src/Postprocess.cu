#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

// 定义一个简单的结构体用于 GPU/CPU 数据交换
// 注意：不要包含 OpenCV 头文件，保持纯净
struct AlignDetection {
    float bbox[4]; // x, y, w, h
    float conf;
    float classId;
};

// CUDA Kernel: 解码 + 阈值过滤
// grid_size: 8400 (anchors)
// num_classes: 4
__global__ void decode_kernel(float* output, int num_anchors, int num_classes,
                              float conf_thresh, AlignDetection* detections, int* count,
                              int max_objects) {
    // 计算当前线程处理的 anchor 索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    // YOLOv11 输出布局: [Batch, Channels, Anchors] -> [1, 8, 8400]
    // 内存中是连续的 8400 个 cx，然后是 8400 个 cy...
    // output[c * num_anchors + idx]

    // 1. 找出最大置信度的类别
    float max_score = -1.0f;
    int max_class_id = -1;

    // 类别分数从第 4 个通道开始 (0-3 是坐标)
    for (int c = 0; c < num_classes; ++c) {
        float score = output[(4 + c) * num_anchors + idx];
        if (score > max_score) {
            max_score = score;
            max_class_id = c;
        }
    }

    // 2. 阈值过滤
    if (max_score > conf_thresh) {
        // 3. 原子操作获取写入位置
        int write_idx = atomicAdd(count, 1);

        if (write_idx < max_objects) {
            // 4. 读取坐标
            float cx = output[0 * num_anchors + idx];
            float cy = output[1 * num_anchors + idx];
            float w  = output[2 * num_anchors + idx];
            float h  = output[3 * num_anchors + idx];

            // 5. 写入结果
            detections[write_idx].bbox[0] = cx;
            detections[write_idx].bbox[1] = cy;
            detections[write_idx].bbox[2] = w;
            detections[write_idx].bbox[3] = h;
            detections[write_idx].conf = max_score;
            detections[write_idx].classId = (float)max_class_id;
        }
    }
}

// Host 包装函数
void launch_decode_kernel(float* output, int num_anchors, int num_classes,
                          float conf_thresh, void* detections, void* count,
                          int max_objects, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_anchors + threads - 1) / threads;

    decode_kernel<<<blocks, threads, 0, stream>>>(
        output, num_anchors, num_classes, conf_thresh,
        (AlignDetection*)detections, (int*)count, max_objects
    );
}
