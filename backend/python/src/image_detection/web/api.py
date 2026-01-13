from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import io
import time
import csv
import base64
import sqlite3
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel

# 引入统一推理引擎
from image_detection.core.engine import InferenceEngine

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
CPP_MODELS_DIR = os.path.join(BASE_DIR, "..", "cpp", "src", "data", "models")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolo11n.pt")
DB_PATH = os.path.join(BASE_DIR, "feedback.db")

INFLECT_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "inflect.csv")
MODEL_MAPPING_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "model_mapping.csv")

app = FastAPI(title="电力巡检图像检测 API", version="1.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 数据库初始化 =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  model_name TEXT,
                  feedback_type TEXT, -- 'false_positive' (误检) or 'false_negative' (漏检)
                  details TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ================= 状态管理 =================

class ModelManager:
    def __init__(self):
        self.engine = None
        self.model_name = "yolo11n (Official)"
        self.class_mapping = self.load_class_mapping()
        self.model_name_mapping = self.load_model_name_mapping()
        
        self.switch_model(self.model_name, DEFAULT_MODEL_PATH)

    def load_class_mapping(self):
        mapping = {}
        if os.path.exists(INFLECT_CSV):
            try:
                with open(INFLECT_CSV, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        keys = rows[0][1:]
                        values = rows[1][1:]
                        mapping = dict(zip(keys, values))
                print(f"✅ 已加载类别映射: {INFLECT_CSV}")
            except Exception as e:
                print(f"⚠️ 读取类别映射失败: {e}")
        return mapping

    def load_model_name_mapping(self):
        mapping = {}
        if os.path.exists(MODEL_MAPPING_CSV):
            try:
                with open(MODEL_MAPPING_CSV, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            mapping[row[0].strip()] = row[1].strip()
                print(f"✅ 已加载模型名映射: {MODEL_MAPPING_CSV}")
            except Exception as e:
                print(f"⚠️ 读取模型名映射失败: {e}")
        return mapping

    def switch_model(self, name: str, path: str):
        try:
            if not os.path.exists(path):
                print(f"❌ 模型文件不存在: {path}")
                return False
            
            use_trt = path.endswith(".engine")
            print(f"🔄 Switching to model: {name} (TRT={use_trt})")
            self.engine = InferenceEngine(path, use_tensorrt=use_trt)
            self.model_name = name
            return True
        except Exception as e:
            print(f"❌ 模型切换失败: {e}")
            import traceback
            traceback.print_exc()
            return False

manager = ModelManager()

# ================= API 路由 =================

@app.get("/models")
async def list_models():
    models = [{"name": "yolo11n (Official)", "path": DEFAULT_MODEL_PATH}]
    
    if os.path.exists(RUNS_DIR):
        for folder in os.listdir(RUNS_DIR):
            folder_path = os.path.join(RUNS_DIR, folder, "weights")
            if not os.path.exists(folder_path):
                continue
            
            display_name = manager.model_name_mapping.get(folder, folder)
            
            pt_path = os.path.join(folder_path, "best.pt")
            if os.path.exists(pt_path):
                models.append({"name": f"PyTorch: {display_name}", "path": pt_path})
            
            engine_path = os.path.join(folder_path, "best.engine")
            if os.path.exists(engine_path):
                models.append({"name": f"TensorRT: {display_name}", "path": engine_path})

    if os.path.exists(CPP_MODELS_DIR):
        for folder in os.listdir(CPP_MODELS_DIR):
            folder_path = os.path.join(CPP_MODELS_DIR, folder)
            if not os.path.isdir(folder_path): continue
            
            display_name = manager.model_name_mapping.get(folder, folder)
            
            for f in os.listdir(folder_path):
                if f.endswith(".engine"):
                    models.append({"name": f"CPP-TRT: {display_name}", "path": os.path.join(folder_path, f)})

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
    
    results = await manager.engine.infer(img_array, conf=conf, iou=iou)
    
    inference_time = (time.time() - start_time) * 1000
    
    detections = []
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for res in results:
        class_id = res['class_id']
        names = manager.engine.get_names()
        en_name = names.get(class_id, str(class_id))
        cn_name = manager.class_mapping.get(en_name, en_name)
        
        confidence = res['conf']
        bbox = res['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        detections.append({
            "class_cn": cn_name,
            "confidence": float(confidence),
            "bbox": [round(x, 1) for x in bbox],
            "dimensions": f"{int(w)} x {int(h)}"
        })
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{en_name} {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", img_bgr)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "success": True,
        "model_used": manager.model_name,
        "inference_time_ms": round(inference_time, 1),
        "detections": detections,
        "image_base64": f"data:image/jpeg;base64,{img_base64}"
    }

@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    start_time = time.time()
    
    images = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        images.append(np.array(image))
    
    batch_results = await manager.engine.infer_batch(images, conf=conf, iou=iou)
    
    inference_time = (time.time() - start_time) * 1000
    
    response_data = []
    
    for i, results in enumerate(batch_results):
        detections = []
        img_bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        
        for res in results:
            class_id = res['class_id']
            names = manager.engine.get_names()
            en_name = names.get(class_id, str(class_id))
            cn_name = manager.class_mapping.get(en_name, en_name)
            
            confidence = res['conf']
            bbox = res['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            detections.append({
                "class_cn": cn_name,
                "confidence": float(confidence),
                "bbox": [round(x, 1) for x in bbox],
                "dimensions": f"{int(w)} x {int(h)}"
            })
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{en_name} {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", img_bgr)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        
        response_data.append({
            "filename": files[i].filename,
            "detections": detections,
            "image_base64": f"data:image/jpeg;base64,{img_base64}"
        })
    
    return {
        "success": True,
        "model_used": manager.model_name,
        "total_inference_time_ms": round(inference_time, 1),
        "results": response_data
    }

class Feedback(BaseModel):
    filename: str
    model_name: str
    feedback_type: str
    details: str

@app.post("/feedback")
async def submit_feedback(feedback: Feedback):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO feedback (filename, model_name, feedback_type, details) VALUES (?, ?, ?, ?)",
                  (feedback.filename, feedback.model_name, feedback.feedback_type, feedback.details))
        conn.commit()
        conn.close()
        return {"success": True, "message": "反馈已提交"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
