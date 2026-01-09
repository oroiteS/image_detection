#pragma once
#include <cuda_runtime_api.h>

// 定义最大检测数量 (单张图片)
#define MAX_DETECTIONS 1024
// 定义最大支持的 Batch Size (用于分配固定大小的 buffer，或者动态分配)
#define MAX_BATCH_SIZE 16

struct DetectionGPU {
    float x1, y1, x2, y2;
    float confidence;
    int classId;
};

// 输出结构体：支持多 Batch
// 内存布局：
// counts[0], counts[1], ...
// detections[batch0_0], detections[batch0_1], ... detections[batch1_0] ...
struct BatchOutputBuffer {
    int counts[MAX_BATCH_SIZE];
    DetectionGPU detections[MAX_BATCH_SIZE * MAX_DETECTIONS];
};

extern "C" void launch_yolo_decode(const float* input, void* output,
                                   int batchSize,
                                   int numAnchors, int numChannels, int numClasses,
                                   float confThreshold, float scale, int inputW, int inputH,
                                   cudaStream_t stream);
