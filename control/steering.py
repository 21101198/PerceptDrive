"""
조향 제어 모듈
차선 이탈 오프셋 + 전방 장애물 거리 → 조향각 출력
"""
import numpy as np
from dataclasses import dataclass
from typing import List
from perception.lane_detector import LaneResult
from perception.detector import Detection


@dataclass
class ControlOutput:
    steering_angle: float    # 도 단위, 양수=우회전
    throttle: float          # 0.0 ~ 1.0
    brake: float             # 0.0 ~ 1.0
    reason: str = ""


class SteeringController:
    """
    PID 기반 차선 추종 제어기.
    전방 장애물 거리가 임계값 이하면 감속/정지.
    """

    def __init__(self,
                 Kp: float = 0.03,
                 safe_distance_m: float = 5.0,
                 stop_distance_m: float = 2.0,
                 max_steer_deg: float = 30.0):
        self.Kp = Kp
        self.safe_dist = safe_distance_m
        self.stop_dist = stop_distance_m
        self.max_steer = max_steer_deg
        self._prev_error = 0.0
        self._integral = 0.0

    def compute(self,
                lane: LaneResult,
                detections: List[Detection],
                depth_map: np.ndarray = None,
                frame_width: int = 1280) -> ControlOutput:
        # 조향 (차선 오프셋 기반 P 제어)
        error = -lane.center_offset / (frame_width / 2)  # -1 ~ +1 정규화
        steer = float(np.clip(self.Kp * error * self.max_steer,
                               -self.max_steer, self.max_steer))

        # 속도 (전방 장애물)
        throttle = 0.5
        brake = 0.0
        reason = "Normal"

        # 차량/사람 클래스 (COCO)
        obstacle_classes = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, ...
        obstacles = [d for d in detections if d.cls in obstacle_classes]

        if obstacles:
            # 가장 가까운 장애물 (y좌표 기준 근사)
            nearest = max(obstacles, key=lambda d: d.y2)
            if nearest.y2 > frame_width * 0.8:  # 화면 하단 80% = 근접
                if nearest.y2 > frame_width * 0.9:
                    brake = 1.0; throttle = 0.0
                    reason = f"Emergency Brake: {nearest.label}"
                else:
                    throttle = 0.2
                    reason = f"Slow Down: {nearest.label} close"

        if lane.departure_warning:
            reason += " | Lane Departure"

        return ControlOutput(steer, throttle, brake, reason)
