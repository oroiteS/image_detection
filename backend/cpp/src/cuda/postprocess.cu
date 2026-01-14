#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "../../include/cuda_utils.hpp" // 引入头文件以使用 BatchOutputBuffer 定义

// YOLOv11 解码 Kernel (支持多 Batch)
// 输入: [Batch, Channels, Anchors] -> [Batch, 8, 8400]
__global__ void decode_yolo_kernel(const float* input, BatchOutputBuffer* output,
                                   int batchSize,
                                   int numAnchors, int numChannels, int numClasses,
                                   float confThreshold, float scale, int inputW, int inputH) {
    // 计算当前线程处理的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 总任务数 = Batch * Anchors
    if (idx >= batchSize * numAnchors) return;

    // 计算当前线程属于哪个 Batch (b) 和哪个 Anchor (a)
    int b = idx / numAnchors;
    int a = idx % numAnchors;

    // 输入数据偏移: 每个 Batch 占用 numChannels * numAnchors
    // input[b][c][a] -> input[b * (numChannels * numAnchors) + c * numAnchors + a]
    size_t batchOffset = (size_t)b * numChannels * numAnchors;

    // 1. 找出最大置信度的类别
    float maxScore = -1.0f;
    int maxClassId = -1;

    // 类别分数从第 4 个通道开始 (0-3 是坐标)
    for (int c = 0; c < numClasses; ++c) {
        float score = input[batchOffset + (4 + c) * numAnchors + a];
        if (score > maxScore) {
            maxScore = score;
            maxClassId = c;
        }
    }

    // 2. 阈值过滤
    if (maxScore > confThreshold) {
        // 3. 原子操作获取输出索引 (针对当前 Batch)
        // output->counts 是一个数组，counts[b] 是当前 Batch 的计数器
        int outIdx = atomicAdd(&(output->counts[b]), 1);

        if (outIdx < MAX_DETECTIONS) {
            // 4. 解码坐标
            float cx = input[batchOffset + 0 * numAnchors + a];
            float cy = input[batchOffset + 1 * numAnchors + a];
            float w  = input[batchOffset + 2 * numAnchors + a];
            float h  = input[batchOffset + 3 * numAnchors + a];

            // 还原到原图坐标 (Letterbox 逆变换)
            float x1 = (cx - 0.5f * w) / scale;
            float y1 = (cy - 0.5f * h) / scale;
            float x2 = (cx + 0.5f * w) / scale;
            float y2 = (cy + 0.5f * h) / scale;

            // 写入输出
            // 输出偏移: b * MAX_DETECTIONS + outIdx
            int globalOutIdx = b * MAX_DETECTIONS + outIdx;

            output->detections[globalOutIdx].x1 = x1;
            output->detections[globalOutIdx].y1 = y1;
            output->detections[globalOutIdx].x2 = x2;
            output->detections[globalOutIdx].y2 = y2;
            output->detections[globalOutIdx].confidence = maxScore;
            output->detections[globalOutIdx].classId = maxClassId;
        }
    }
}

// Host 端调用函数
extern "C" void launch_yolo_decode(const float* input, void* output,
                                   int batchSize,
                                   int numAnchors, int numChannels, int numClasses,
                                   float confThreshold, float scale, int inputW, int inputH,
                                   cudaStream_t stream) {
    // 初始化计数器为 0 (清空所有 Batch 的计数器)
    // counts 数组在 BatchOutputBuffer 的最前面，大小为 batchSize * sizeof(int)
    // 但为了安全，我们清空整个 counts 数组区域 (MAX_BATCH_SIZE)
    cudaMemsetAsync(output, 0, MAX_BATCH_SIZE * sizeof(int), stream);

    int totalThreads = batchSize * numAnchors;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    decode_yolo_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        input, (BatchOutputBuffer*)output,
        batchSize,
        numAnchors, numChannels, numClasses,
        confThreshold, scale, inputW, inputH
    );
}
