"""시각화 유틸리티"""
import numpy as np
import cv2
from typing import List, Optional
from perception.detector import Detection
from perception.lane_detector import LaneResult
from control.steering import ControlOutput


def draw_hud(frame: np.ndarray,
             detections: List[Detection],
             lane: LaneResult,
             control: ControlOutput,
             fps: float,
             frame_type: str = 'I') -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 객체 박스
    for d in detections:
        color = (0, 255, 0) if d.cls not in {0, 2, 5, 7} else (0, 120, 255)
        cv2.rectangle(vis, (int(d.x1), int(d.y1)),
                           (int(d.x2), int(d.y2)), color, 2)
        cv2.putText(vis, f"{d.label} {d.conf:.2f}",
                    (int(d.x1), int(d.y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 차선
    if lane.left_line:
        x1, y1, x2, y2 = lane.left_line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 80), 4)
    if lane.right_line:
        x1, y1, x2, y2 = lane.right_line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 80), 4)

    # HUD 패널
    panel_color = (20, 20, 20)
    cv2.rectangle(vis, (0, 0), (340, 130), panel_color, -1)
    cv2.rectangle(vis, (0, 0), (340, 130), (80, 80, 80), 1)

    ft_color = (0, 200, 255) if frame_type == 'I' else (180, 180, 180)
    texts = [
        (f"FPS: {fps:.1f}  [{frame_type}-frame]", ft_color),
        (f"Steer: {control.steering_angle:+.1f}°", (255, 220, 0)),
        (f"Throttle: {control.throttle:.2f}  Brake: {control.brake:.2f}", (200, 255, 200)),
        (f"Offset: {lane.center_offset:+.0f}px", (200, 200, 255)),
        (control.reason[:45], (255, 100, 100) if lane.departure_warning else (200, 200, 200)),
    ]
    for i, (txt, color) in enumerate(texts):
        cv2.putText(vis, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return vis
