"""
차선 인식 모듈
방법 1 (기본): HSV 필터 + Hough 변환 (실시간, 의존성 없음)
방법 2 (고급): LaneNet 딥러닝 모델 (별도 설치 필요)
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class LaneResult:
    left_line: Optional[Tuple]  = None   # (x1, y1, x2, y2)
    right_line: Optional[Tuple] = None
    center_offset: float = 0.0           # 픽셀 단위, 양수=우측으로 치우침
    curvature: float = 0.0               # 반지름 (m), 직선=inf
    departure_warning: bool = False


class HoughLaneDetector:
    """
    HSV 흰색/노란색 필터 → Canny → Hough 변환으로 차선 검출.
    YOLOv5s와 동시에 실행 가능 (CPU 기반, ~2ms/frame on 720p).
    """

    def __init__(self,
                 roi_top_ratio: float = 0.55,
                 canny_low: int = 50,
                 canny_high: int = 150):
        self.roi_top = roi_top_ratio   # 상단 ROI 비율 (0~1)
        self.canny_low = canny_low
        self.canny_high = canny_high

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        roi = np.array([[
            (0, h),
            (w * 0.1, h * self.roi_top),
            (w * 0.9, h * self.roi_top),
            (w, h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        return mask

    def _filter_white_yellow(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 흰색
        white_mask = cv2.inRange(hsv,
            np.array([0, 0, 200]),
            np.array([180, 30, 255]))
        # 노란색
        yellow_mask = cv2.inRange(hsv,
            np.array([15, 80, 100]),
            np.array([35, 255, 255]))
        return cv2.bitwise_or(white_mask, yellow_mask)

    def detect(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        color_mask = self._filter_white_yellow(frame)
        roi_mask = self._extract_roi(frame)
        masked = cv2.bitwise_and(color_mask, roi_mask)

        blurred = cv2.GaussianBlur(masked, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(edges,
            rho=1, theta=np.pi/180,
            threshold=30, minLineLength=50, maxLineGap=100)

        if lines is None:
            return LaneResult()

        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  # 수평선 무시
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

        def average_line(lines_list, h):
            if not lines_list:
                return None
            x1s, y1s, x2s, y2s = zip(*lines_list)
            x1 = int(np.mean(x1s)); y1 = int(np.mean(y1s))
            x2 = int(np.mean(x2s)); y2 = int(np.mean(y2s))
            # y 범위를 ROI 하단~상단으로 정규화
            if x2 == x1: return None
            slope = (y2 - y1) / (x2 - x1)
            b = y1 - slope * x1
            y_bot = h; y_top = int(h * 0.6)
            x_bot = int((y_bot - b) / slope)
            x_top = int((y_top - b) / slope)
            return (x_bot, y_bot, x_top, y_top)

        left  = average_line(left_lines,  h)
        right = average_line(right_lines, h)

        # 차선 중심 이탈 계산
        center_offset = 0.0
        if left and right:
            lane_center = (left[0] + right[0]) / 2
            center_offset = lane_center - w / 2
        elif left:
            center_offset = left[0] - w / 2
        elif right:
            center_offset = right[0] - w / 2

        departure = abs(center_offset) > w * 0.15

        return LaneResult(
            left_line=left,
            right_line=right,
            center_offset=center_offset,
            departure_warning=departure
        )

    def draw(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        overlay = frame.copy()
        if result.left_line:
            x1, y1, x2, y2 = result.left_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        if result.right_line:
            x1, y1, x2, y2 = result.right_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        if result.departure_warning:
            cv2.putText(overlay, "⚠ LANE DEPARTURE",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return overlay
