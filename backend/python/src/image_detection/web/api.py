from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolo11n.pt")
INFLECT_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "inflect.csv")

app = FastAPI(title="电力巡检图像检测 API", version="1.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 状态管理 =================

class ModelManager:
    def __init__(self):
        self.model = YOLO(DEFAULT_MODEL_PATH)
        self.model_name = "yolo11n (Official)"
        self.mapping = self.load_mapping()

    def load_mapping(self):
        mapping = {}
        try:
            if os.path.exists(INFLECT_CSV):
                with open(INFLECT_CSV, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        mapping = dict(zip(rows[0][1:], rows[1][1:]))
        except Exception as e:
            print(f"⚠️ 映射加载失败: {e}")
        return mapping

    def switch_model(self, name: str, path: str):
        try:
            if not os.path.exists(path):
                return False
            self.model = YOLO(path)
            self.model_name = name
            return True
        except Exception as e:
            print(f"❌ 模型切换失败: {e}")
            return False

manager = ModelManager()

# ================= API 路由 =================

@app.get("/models")
async def list_models():
    models = [{"name": "yolo11n (Official)", "path": DEFAULT_MODEL_PATH}]
    if os.path.exists(RUNS_DIR):
        for folder in os.listdir(RUNS_DIR):
            pt_path = os.path.join(RUNS_DIR, folder, "weights", "best.pt")
            if os.path.exists(pt_path):
                models.append({"name": f"Custom: {folder}", "path": pt_path})
    return models

@app.post("/set_model")
async def set_model(data: dict):
    name = data.get("name")
    path = data.get("path")
    if not name or not path:
        raise HTTPException(status_code=400, detail="参数缺失")
    
    success = manager.switch_model(name, path)
    if success:
        return {"message": f"已切换至 {name}"}
    raise HTTPException(status_code=500, detail="模型加载失败")

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    start_time = time.time()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    
    results = manager.model.predict(source=img_array, conf=conf, iou=iou)
    inference_time = (time.time() - start_time) * 1000
    
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        en_name = manager.model.names[cls_id]
        detections.append({
            "class_cn": manager.mapping.get(en_name, en_name),
            "confidence": float(box.conf[0]),
            "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
        })
    
    res_plotted = results[0].plot()
    # YOLO plot() 返回 BGR，cv2.imencode 需要 BGR，所以直接编码即可
    _, buffer = cv2.imencode(".jpg", res_plotted)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "success": True,
        "model_used": manager.model_name,
        "inference_time_ms": round(inference_time, 1),
        "detections": detections,
        "image_base64": f"data:image/jpeg;base64,{img_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
