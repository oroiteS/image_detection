import gradio as gr
from ultralytics import YOLO
import os
import time
import pandas as pd
from PIL import Image
import csv

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
DEFAULT_MODEL = os.path.join(BASE_DIR, "weights", "yolo11n.pt")
INFLECT_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "inflect.csv")

# ================= 逻辑函数 =================

def load_class_mapping():
    """从 CSV 加载中英文映射"""
    mapping = {}
    try:
        if os.path.exists(INFLECT_CSV):
            with open(INFLECT_CSV, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) >= 2:
                    en_names = rows[0][1:] # 跳过第一个 'en'
                    cn_names = rows[1][1:] # 跳过第一个 'cn'
                    mapping = dict(zip(en_names, cn_names))
    except Exception as e:
        print(f"⚠️ 加载中文映射失败: {e}")
    return mapping

# 预加载映射
CLASS_MAP = load_class_mapping()

def get_available_models():
    """扫描所有可用的模型文件"""
    models = {"yolo11n (Official)": DEFAULT_MODEL}
    if os.path.exists(RUNS_DIR):
        for folder in os.listdir(RUNS_DIR):
            pt_path = os.path.join(RUNS_DIR, folder, "weights", "best.pt")
            if os.path.exists(pt_path):
                models[f"Custom: {folder}"] = pt_path
    return models

# 全局变量缓存当前加载的模型
current_model = YOLO(DEFAULT_MODEL)

def load_model(model_name):
    global current_model
    models = get_available_models()
    path = models.get(model_name, DEFAULT_MODEL)
    current_model = YOLO(path)
    return f"✅ 已加载模型: {model_name}"

def detect_objects(image, conf_threshold, iou_threshold):
    if image is None:
        return None, None, "请先上传图片"
    
    start_time = time.time()
    results = current_model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold
    )
    inference_time = (time.time() - start_time) * 1000 # ms
    
    # 绘制结果图
    res_plotted = results[0].plot()
    res_rgb = res_plotted[:, :, ::-1]
    
    # 提取检测信息并转换中文
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        en_name = current_model.names[cls_id]
        # 查找中文名，找不到则用英文名
        cn_name = CLASS_MAP.get(en_name, en_name)
        detections.append({"类别": cn_name, "置信度": f"{conf:.2f}"})
    
    df = pd.DataFrame(detections) if detections else pd.DataFrame(columns=["类别", "置信度"])
    
    status = f"🚀 推理耗时: {inference_time:.1f}ms | 检测到 {len(detections)} 个目标"
    return Image.fromarray(res_rgb), df, status

# ================= UI 界面 =================

with gr.Blocks(title="电力巡检智能检测系统") as demo:
    gr.Markdown("# ⚡ 电力巡检图像智能检测系统")
    gr.Markdown("### 第3天任务：模型评估与 Web 原型展示")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(get_available_models().keys()),
                value="yolo11n (Official)",
                label="选择模型权重"
            )
            load_btn = gr.Button("🔄 重新加载模型", variant="secondary")
            load_status = gr.Markdown("当前模型: yolo11n (Official)")
            
            gr.Markdown("---")
            
            conf_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="置信度阈值 (Confidence)")
            iou_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.45, label="交并比阈值 (IoU)")
            
            detect_btn = gr.Button("🔍 开始检测", variant="primary")

        with gr.Column(scale=2):
            input_img = gr.Image(type="pil", label="上传待检测图片")
            output_img = gr.Image(type="pil", label="检测结果")
            
    with gr.Row():
        with gr.Column():
            status_output = gr.Textbox(label="运行状态", interactive=False)
            result_table = gr.Dataframe(label="检测详情 (中文映射已启用)")

    # 事件绑定
    load_btn.click(load_model, inputs=[model_dropdown], outputs=[load_status])
    detect_btn.click(
        detect_objects, 
        inputs=[input_img, conf_slider, iou_slider], 
        outputs=[output_img, result_table, status_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
