import gradio as gr
from ultralytics import YOLO
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 权重文件路径
MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolo11n.pt")

# 加载默认模型
model = YOLO(MODEL_PATH)

def predict(image):
    results = model.predict(image)
    return results[0].plot()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="电力巡检图像智能检测系统"
)

if __name__ == "__main__":
    demo.launch()
