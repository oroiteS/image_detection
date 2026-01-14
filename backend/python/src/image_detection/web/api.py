from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import io
import time
import csv
import base64
import sqlite3
import json
import hashlib
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
DB_PATH = os.path.join(BASE_DIR, "app.db") # 统一数据库文件

INFLECT_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "inflect.csv")
MODEL_MAPPING_CSV = os.path.join(BASE_DIR, "src", "image_detection", "data", "model_mapping.csv")

app = FastAPI(title="电力巡检图像检测 API", version="2.0.0")

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
    
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # 历史记录表
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  filename TEXT,
                  model_name TEXT,
                  inference_time REAL,
                  detection_count INTEGER,
                  result_image_base64 TEXT, -- 存储缩略图或完整Base64 (生产环境建议存路径)
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # 反馈表
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  model_name TEXT,
                  feedback_type TEXT,
                  details TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                  
    conn.commit()
    conn.close()

init_db()

# ================= 辅助函数 =================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ================= Pydantic Models =================
class UserAuth(BaseModel):
    username: str
    password: str

class Feedback(BaseModel):
    filename: str
    model_name: str
    feedback_type: str
    details: str

# ================= 状态管理 (ModelManager) =================
# (保持原有 ModelManager 代码不变)
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
            except: pass
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
            except: pass
        return mapping

    def switch_model(self, name: str, path: str):
        try:
            if not os.path.exists(path): return False
            use_trt = path.endswith(".engine")
            print(f"🔄 Switching to model: {name} (TRT={use_trt})")
            self.engine = InferenceEngine(path, use_tensorrt=use_trt)
            self.model_name = name
            return True
        except Exception as e:
            print(f"❌ 模型切换失败: {e}")
            return False

manager = ModelManager()

# ================= API 路由 =================

# --- 用户认证 ---
@app.post("/register")
async def register(user: UserAuth):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                  (user.username, hash_password(user.password)))
        conn.commit()
        conn.close()
        return {"success": True, "message": "注册成功"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="用户名已存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(user: UserAuth):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?",
              (user.username, hash_password(user.password)))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {"success": True, "username": user.username, "token": "dummy-token"} # 简单实现
    else:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

# --- 历史记录 ---
@app.get("/history")
async def get_history(username: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC LIMIT 50", (username,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- 模型相关 ---
@app.get("/models")
async def list_models():
    models = [{"name": "yolo11n (Official)", "path": DEFAULT_MODEL_PATH}]
    if os.path.exists(RUNS_DIR):
        for folder in os.listdir(RUNS_DIR):
            folder_path = os.path.join(RUNS_DIR, folder, "weights")
            if not os.path.exists(folder_path): continue
            display_name = manager.model_name_mapping.get(folder, folder)
            pt_path = os.path.join(folder_path, "best.pt")
            if os.path.exists(pt_path): models.append({"name": f"PyTorch: {display_name}", "path": pt_path})
            engine_path = os.path.join(folder_path, "best.engine")
            if os.path.exists(engine_path): models.append({"name": f"TensorRT: {display_name}", "path": engine_path})
    if os.path.exists(CPP_MODELS_DIR):
        for folder in os.listdir(CPP_MODELS_DIR):
            folder_path = os.path.join(CPP_MODELS_DIR, folder)
            if not os.path.isdir(folder_path): continue
            display_name = manager.model_name_mapping.get(folder, folder)
            for f in os.listdir(folder_path):
                if f.endswith(".engine"): models.append({"name": f"CPP-TRT: {display_name}", "path": os.path.join(folder_path, f)})
    return models

@app.post("/set_model")
async def set_model(data: dict):
    name = data.get("name")
    path = data.get("path")
    if not name or not path: raise HTTPException(status_code=400, detail="参数缺失")
    if manager.switch_model(name, path): return {"message": f"已切换至 {name}"}
    raise HTTPException(status_code=500, detail="模型加载失败")

# --- 检测接口 (修改以支持历史记录) ---
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    username: Optional[str] = Form(None) # 新增 username 参数
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
    
    # 保存历史记录
    if username:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            # 为了节省空间，这里可以只存缩略图，或者不存图片
            # 这里演示存完整图片
            c.execute("INSERT INTO history (username, filename, model_name, inference_time, detection_count, result_image_base64) VALUES (?, ?, ?, ?, ?, ?)",
                      (username, file.filename, manager.model_name, inference_time, len(detections), f"data:image/jpeg;base64,{img_base64}"))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ 保存历史记录失败: {e}")

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
    iou: float = Form(0.45),
    username: Optional[str] = Form(None)
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
    
    conn = sqlite3.connect(DB_PATH) if username else None
    c = conn.cursor() if conn else None

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
        final_base64 = f"data:image/jpeg;base64,{img_base64}"
        
        response_data.append({
            "filename": files[i].filename,
            "detections": detections,
            "image_base64": final_base64
        })

        if c:
            c.execute("INSERT INTO history (username, filename, model_name, inference_time, detection_count, result_image_base64) VALUES (?, ?, ?, ?, ?, ?)",
                      (username, files[i].filename, manager.model_name, inference_time / len(files), len(detections), final_base64))
    
    if conn:
        conn.commit()
        conn.close()

    return {
        "success": True,
        "model_used": manager.model_name,
        "total_inference_time_ms": round(inference_time, 1),
        "results": response_data
    }

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
