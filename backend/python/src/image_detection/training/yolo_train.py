import os
import glob
import time
from ultralytics import YOLO
import cv2
import shutil

# ================= 配置区域 =================
# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 数据集配置文件
DATA_YAML = os.path.join(BASE_DIR, "datasets", "power_inspection", "data.yaml")
# 结果保存根目录
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
# 权重文件存放目录
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# 训练参数
MODEL_TYPE = os.path.join(WEIGHTS_DIR, "yolo11n.pt")
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0


# ===========================================

def get_all_trained_models():
    """
    扫描 runs/detect 下所有包含 weights/best.pt 的文件夹
    返回列表: [{'name': 'train4', 'path': '...', 'time': '...'}, ...]
    """
    if not os.path.exists(RUNS_DIR):
        return []

    # 找所有 train 开头的文件夹
    candidates = [d for d in os.listdir(RUNS_DIR) if d.startswith('train') and os.path.isdir(os.path.join(RUNS_DIR, d))]

    valid_models = []
    for folder in candidates:
        pt_path = os.path.join(RUNS_DIR, folder, "weights", "best.pt")
        if os.path.exists(pt_path):
            # 获取最后修改时间
            mtime = os.path.getmtime(pt_path)
            time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
            valid_models.append({
                'name': folder,
                'path': pt_path,
                'mtime': mtime,
                'time_str': time_str
            })

    # 按时间倒序排列 (最新的在前)
    valid_models.sort(key=lambda x: x['mtime'], reverse=True)
    return valid_models


def select_model_interactive():
    """
    交互式让用户选择模型
    """
    models = get_all_trained_models()

    if not models:
        print("❌ 未找到任何训练好的模型 (best.pt)。请先执行训练。")
        return None

    print("\n" + "=" * 50)
    print("📋 可用的训练记录:")
    print("=" * 50)
    for i, m in enumerate(models):
        print(f" [{i + 1}] {m['name']:<10} | 🕒 {m['time_str']} | 📂 {m['path']}")
    print("=" * 50)

    while True:
        choice = input(f"请输入序号选择模型 (1-{len(models)}, 回车默认选最新): ").strip()
        if choice == "":
            selected = models[0]
            break
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                break
        print("输入无效，请重新输入。")

    print(f"\n✅ 已锁定模型: {selected['name']} ({selected['path']})")
    return selected['path']


def train_model():
    print(f"🚀 开始训练: {MODEL_TYPE} | Epochs: {EPOCHS}")
    model = YOLO(MODEL_TYPE)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=2,
        project=RUNS_DIR,
        name="train"
    )
    print("✅ 训练完成！")


def evaluate_model():
    print("📊 [评估模式]")
    model_path = select_model_interactive()
    if not model_path: return

    print("正在加载模型进行评估...")
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, split='test', device=DEVICE)

    print(f"📈 mAP50:    {metrics.box.map50:.4f}")
    print(f"📈 mAP50-95: {metrics.box.map:.4f}")
    print(f"📂 评估结果已保存至: {metrics.save_dir}")


def predict_single_image():
    print("🖼️ [推理模式]")
    model_path = select_model_interactive()
    if not model_path: return

    model = YOLO(model_path)

    img_path = input("请输入图片路径 (支持拖入文件): ").strip().strip('"')
    if not os.path.exists(img_path):
        print("❌ 图片不存在。")
        return

    results = model.predict(img_path, save=True, conf=0.25)
    print(f"✅ 推理完成，结果保存在: {results[0].save_dir}")

    try:
        res_plot = results[0].plot()
        cv2.imshow("Result", res_plot)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass


def export_to_onnx():
    """
    将 .pt 转换为 .onnx (为 TensorRT 做准备)
    """
    print("📦 [模型导出模式 - ONNX]")
    print("此步骤生成的 .onnx 文件将交给成员C进行 C++ TensorRT 部署。")

    model_path = select_model_interactive()
    if not model_path: return

    print("\n⏳ 正在导出 ONNX，请稍候...")
    model = YOLO(model_path)

    # 核心导出代码
    # dynamic=True: 支持动态输入尺寸 and 动态Batch (TensorRT 关键要求)
    # simplify=True: 使用 onnxsim 简化图结构
    # opset=12: 兼容性最好的算子集
    success = model.export(
        format='onnx',
        dynamic=True,
        simplify=True,
        opset=12
    )

    if success:
        onnx_path = model_path.replace('.pt', '.onnx')
        print("\n" + "*" * 50)
        print("🎉 导出成功！")
        print(f"📂 ONNX 文件路径: {onnx_path}")
        print("*" * 50)
        print("👉 下一步: 请将此文件发送给成员C，他将使用 TensorRT C++ API 加载它。")
    else:
        print("❌ 导出失败，请检查错误日志。")


def main():
    while True:
        print("\n" + "=" * 30)
        print("   YOLOv11 任务管理器 (v3.0 Pro)")
        print("=" * 30)
        print("1. [训练] 新一轮训练 (Train)")
        print("2. [评估] 评估历史模型 (Evaluate)")
        print("3. [推理] 单图测试 (Predict)")
        print("4. [导出] 转为 ONNX 格式 (Export)")
        print("q. 退出")

        choice = input("请选择任务: ").lower()

        if choice == '1':
            train_model()
        elif choice == '2':
            evaluate_model()
        elif choice == '3':
            predict_single_image()
        elif choice == '4':
            export_to_onnx()
        elif choice == 'q':
            break


if __name__ == "__main__":
    main()
