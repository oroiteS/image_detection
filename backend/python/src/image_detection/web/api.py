from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import io
import time
import csv
import base64
import numpy as np
import cv2
from PIL import Image

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_MODEL = os.path.join(BASE_DIR, "weights", "yolo11n.pt")
INFLECT_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "inflect.csv")

app = FastAPI(title="电力巡检图像检测 API", version="1.0.0")

# 允许跨域请求 (Vue3 开发环境必备)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 逻辑初始化 =================

# 加载模型
model = YOLO(DEFAULT_MODEL)

# 加载中文映射
def load_class_mapping():
    mapping = {}
    try:
        if os.path.exists(INFLECT_CSV):
            with open(INFLECT_CSV, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) >= 2:
                    en_names = rows[0][1:]
                    cn_names = rows[1][1:]
                    mapping = dict(zip(en_names, cn_names))
    except Exception as e:
        print(f"⚠️ 加载中文映射失败: {e}")
    return mapping

CLASS_MAP = load_class_mapping()

# ================= API 路由 =================

@app.get("/")
async def root():
    return {"message": "电力巡检 API 服务已启动", "status": "running"}

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    """
    接收图片并返回检测结果
    """
    start_time = time.time()
    
    # 1. 读取上传的图片
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    
    # 2. YOLO 推理
    results = model.predict(source=img_array, conf=conf, iou=iou)
    inference_time = (time.time() - start_time) * 1000
    
    # 3. 解析结果
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        en_name = model.names[cls_id]
        cn_name = CLASS_MAP.get(en_name, en_name)
        
        # 坐标 [x1, y1, x2, y2]
        coords = box.xyxy[0].tolist()
        
        detections.append({
            "class_en": en_name,
            "class_cn": cn_name,
            "confidence": round(conf_score, 2),
            "bbox": [round(x, 1) for x in coords]
        })
    
    # 4. 生成带框的结果图 (Base64 编码)
    res_plotted = results[0].plot()
    # YOLO 返回 BGR，转为 RGB 再转为 JPEG
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "success": True,
        "inference_time_ms": round(inference_time, 1),
        "count": len(detections),
        "detections": detections,
        "image_base64": f"data:image/jpeg;base64,{img_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
