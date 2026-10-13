"""
단안 거리 추정 모듈
모델: MiDaS v2.1 Small (Intel) — 실시간 가능
     DepthAnything v2 Small — 정확도 우선
"""
import numpy as np
import cv2
import torch
from typing import Optional, Tuple


class DepthEstimator:
    """
    단안 카메라로 상대 거리 추정.
    객체 탐지 결과와 결합해 각 바운딩 박스의 추정 거리 반환.
    """

    MODELS = {
        "midas_small":    "Intel/dpt-hybrid-midas",
        "midas_v21":      "intel-isl/MiDaS",
        "depth_anything": "LiheYoung/depth-anything-small-hf",
    }

    def __init__(self, model_name: str = "midas_small",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._transform = None
        self._focal_length: Optional[float] = None  # 캘리브레이션 시 설정

    def load(self):
        import torch
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.MODELS.get(self.model_name, self.model_name),
            device=0 if self.device == "cuda" else -1
        )
        print(f"[Depth] Loaded {self.model_name}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns depth map (H×W float32), 값이 클수록 가까움 (상대값).
        절대 거리 변환은 calibrate() 필요.
        """
        if self._pipe is None:
            self.load()
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._pipe(pil)
        depth = np.array(result["depth"], dtype=np.float32)
        return depth

    def get_box_distance(self, depth_map: np.ndarray,
                         box: Tuple[float, float, float, float],
                         percentile: int = 20) -> float:
        """
        바운딩 박스 내 하위 percentile 깊이값 → 거리 추정.
        percentile=20: 노이즈 제거 위해 가장 가까운 20% 값 사용.
        """
        h, w = depth_map.shape
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return float('inf')
        # 상대 깊이값 → 작을수록 멀다
        val = float(np.percentile(roi, percentile))
        if self._focal_length and val > 0:
            return self._focal_length / val  # 미터 단위 (캘리브 후)
        return val  # 상대값

    def calibrate(self, known_distance_m: float,
                  known_depth_val: float):
        """focal_length = known_distance * known_depth_val"""
        self._focal_length = known_distance_m * known_depth_val

    def colorize(self, depth_map: np.ndarray) -> np.ndarray:
        """깊이 맵을 시각화용 BGR 이미지로 변환."""
        norm = cv2.normalize(depth_map, None, 0, 255,
                             cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
