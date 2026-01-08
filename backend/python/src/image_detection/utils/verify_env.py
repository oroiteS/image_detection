import torch
import sys
import os


def check_environment():
    print("=" * 30)
    print("环境自检报告 (Member B - RTX 4070s)")
    print("=" * 30)

    # 1. Check Python Version
    print(f"[Python 版本]: {sys.version.split()[0]}")

    # 2. Check PyTorch Version
    try:
        import torch
        print(f"[PyTorch 版本]: {torch.__version__}")
    except ImportError:
        print("[错误]: 未检测到 PyTorch，请执行安装命令！")
        return

    # 3. Check CUDA & GPU
    cuda_available = torch.cuda.is_available()
    print(f"[CUDA 是否可用]: {'✅ 是' if cuda_available else '❌ 否'}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[检测到显卡]: {gpu_name}")
        print(f"[CUDA 版本]: {torch.version.cuda}")

        # 验证是否为 RTX 4070s
        if "4070" in gpu_name:
            print("  -> 显卡型号匹配成功！")
        else:
            print(f"  -> 警告: 当前显卡 ({gpu_name}) 与计划表(4070s)不符，请确认。")

        # 简单的张量运算测试
        try:
            x = torch.rand(5, 3).cuda()
            print("[GPU 运算测试]: ✅ 成功 (Tensor已加载至显卡)")
        except Exception as e:
            print(f"[GPU 运算测试]: ❌ 失败 - {e}")
    else:
        print("[严重警告]: PyTorch 正在使用 CPU 运行！这将导致训练极慢。请检查 CUDA 驱动和 PyTorch 安装版本。")

    # 4. Check Ultralytics (YOLO)
    try:
        import ultralytics
        print(f"[YOLO (Ultralytics) 版本]: {ultralytics.__version__}")
    except ImportError:
        print("[警告]: 未检测到 ultralytics 库，无法进行 YOLO 训练。")

    # 5. Check Gradio
    try:
        import gradio
        print(f"[Gradio 版本]: {gradio.__version__}")
    except ImportError:
        print("[警告]: 未检测到 Gradio，无法开发 Web 界面。")

    # 6. Check ONNX
    try:
        import onnx
        print(f"[ONNX 版本]: {onnx.__version__}")
    except ImportError:
        print("[警告]: 未检测到 ONNX，无法进行模型导出。")

    print("=" * 30)
    if cuda_available:
        print("🎉 恭喜！基础环境配置完成。")
    else:
        print("⚠️ 请根据上述警告检查配置。")


if __name__ == "__main__":
    check_environment()
