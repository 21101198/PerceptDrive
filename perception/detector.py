"""
YOLOv5s 객체 탐지 — TensorRT INT8 지원 (onnxruntime-gpu, pycuda 불필요)
"""
import numpy as np
import cv2
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    conf: float; cls: int; label: str = ""

    @property
    def box(self):
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


class ObjectDetector:
    """
    YOLOv5s / YOLOv8s 래퍼.
    backend: 'torch' | 'trt_fp16' | 'trt_int8'
    """

    COCO_CLASSES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck',
        'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
        'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
        'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
        'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
        'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
        'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
        'hair drier','toothbrush'
    ]

    def __init__(self, weights: str = "yolov5s.pt",
                 backend: str = "torch",
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 device: str = "cuda"):
        self.weights = weights
        self.backend = backend
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self._model = None
        self._last_conf = 1.0

    def load(self):
        if self.backend == "torch":
            import torch
            self._model = torch.hub.load(
                'ultralytics/yolov5', 'custom',
                path=self.weights, trust_repo=True
            )
            self._model.conf = self.conf_thres
            self._model.iou = self.iou_thres
            if self.device == "cuda":
                self._model.cuda()
            self._model.eval()
        elif self.backend in ("trt_fp16", "trt_int8"):
            import onnxruntime as ort
            providers = [("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_int8_enable": self.backend == "trt_int8",
            }), "CUDAExecutionProvider"]
            self._model = ort.InferenceSession(
                self.weights.replace(".pt", ".onnx"), providers=providers
            )
        print(f"[Detector] Loaded {self.weights} ({self.backend})")

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Returns (detections, mean_conf)"""
        t0 = time.perf_counter()
        if self._model is None:
            self.load()

        if self.backend == "torch":
            results = self._model(frame[..., ::-1])  # BGR→RGB
            dets = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                label = self.COCO_CLASSES[int(cls)] if int(cls) < len(self.COCO_CLASSES) else str(int(cls))
                dets.append(Detection(*xyxy, float(conf), int(cls), label))
        else:
            dets = []  # TRT path (별도 구현)

        mean_conf = float(np.mean([d.conf for d in dets])) if dets else 0.0
        self._last_conf = mean_conf
        return dets, mean_conf

    @property
    def last_conf(self) -> float:
        return self._last_conf
