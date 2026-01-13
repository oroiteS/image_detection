import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    import image_detection_cpp
    CPP_AVAILABLE = True
    print("✅ C++ TensorRT module loaded successfully.")
except ImportError:
    try:
        from . import image_detection_cpp
        CPP_AVAILABLE = True
        print("✅ C++ TensorRT module loaded successfully (package import).")
    except ImportError as e:
        CPP_AVAILABLE = False
        print(f"⚠️ Warning: C++ TensorRT module not found ({e}). Running in PyTorch-only mode.")

class InferenceEngine:
    def __init__(self, model_path: str, use_tensorrt: bool = False):
        self.model_path = model_path
        if model_path.endswith(".engine"):
            use_tensorrt = True
            
        self.use_tensorrt = use_tensorrt
        self.model = None
        self.cpp_detector = None
        self.lock = asyncio.Lock()
        
        self._load_model()

    def _load_model(self):
        if self.use_tensorrt:
            if not CPP_AVAILABLE:
                raise RuntimeError(
                    "❌ 尝试加载 TensorRT 引擎，但 C++ 模块 (image_detection_cpp) 未找到。\n"
                    "请确保已编译 C++ 后端并将 .so 文件复制到 backend/python/src/image_detection/core/ 目录下。"
                )
                
            print(f"🚀 Loading TensorRT Engine (C++): {self.model_path}")
            engine_path = self.model_path
            onnx_path = self.model_path.replace('.engine', '.onnx')
            
            self.cpp_detector = image_detection_cpp.ObjectDetector()
            self.cpp_detector.init(onnx_path, engine_path, use_int8=False)
        else:
            print(f"🔥 Loading PyTorch Model: {self.model_path}")
            self.model = YOLO(self.model_path)

    async def infer(self, image, conf=0.25, iou=0.45):
        """单图推理"""
        results = []
        async with self.lock:
            if self.use_tensorrt:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                det_results = self.cpp_detector.detect_single(img_bgr, conf, iou)
                for det in det_results:
                    results.append({
                        'class_id': det.class_id,
                        'class_name': str(det.class_id),
                        'conf': det.conf,
                        'bbox': det.bbox
                    })
            else:
                yolo_res = self.model.predict(image, conf=conf, iou=iou)[0]
                for box in yolo_res.boxes:
                    cls_id = int(box.cls[0])
                    results.append({
                        'class_id': cls_id,
                        'class_name': self.model.names[cls_id],
                        'conf': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
        return results

    async def infer_batch(self, images, conf=0.25, iou=0.45):
        """批量推理"""
        batch_results = []
        async with self.lock:
            if self.use_tensorrt:
                # 转换所有图片为 BGR
                imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
                
                # 调用 C++ detect_batch
                # 返回 vector<vector<Detection>>
                all_dets = self.cpp_detector.detect_batch(imgs_bgr, conf, iou)
                
                for det_list in all_dets:
                    single_res = []
                    for det in det_list:
                        single_res.append({
                            'class_id': det.class_id,
                            'class_name': str(det.class_id),
                            'conf': det.conf,
                            'bbox': det.bbox
                        })
                    batch_results.append(single_res)
            else:
                # PyTorch 批量推理
                # YOLO predict 支持 list of images
                yolo_results = self.model.predict(images, conf=conf, iou=iou)
                for res in yolo_results:
                    single_res = []
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        single_res.append({
                            'class_id': cls_id,
                            'class_name': self.model.names[cls_id],
                            'conf': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        })
                    batch_results.append(single_res)
                    
        return batch_results

    def get_names(self):
        if self.use_tensorrt:
            return {0: "missing_insulator", 1: "burned_insulator", 2: "bird_nest", 3: "shifted_grading_ring"}
        else:
            return self.model.names
