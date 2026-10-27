"""
CodecInspiredScheduler — 캡스톤 설계에서 이식
SCS = w1 * Δpixel + w2 * (1 - conf)
최적 파라미터: w1=0.6, w2=0.4, θ=0.35
"""
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SchedulerConfig:
    w1: float = 0.6          # 픽셀 변화 가중치
    w2: float = 0.4          # 신뢰도 가중치
    theta: float = 0.35      # I-frame 임계값
    max_p_streak: int = 4    # 최대 연속 P-frame 수
    alert_sigma: float = 1.0 # Proactive 예측기 σ 배수


class CodecInspiredScheduler:
    """
    I/P-frame 기반 선택적 추론 스케줄러.
    5.4× FPS 향상 (134.9 → 728.9 FPS on RTX 4060)
    """

    def __init__(self, config: SchedulerConfig = None):
        self.cfg = config or SchedulerConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._p_streak: int = 0
        self._last_conf: float = 1.0

        # Proactive tail latency predictor
        self._latency_history: deque = deque(maxlen=30)
        self._alert_count: int = 0

    def decide(self, frame: np.ndarray, last_conf: float = 1.0) -> str:
        """
        Returns:
            'I' — 전체 추론 필요
            'P' — 이전 결과 재사용
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 90))

        if self._prev_gray is None or self._p_streak >= self.cfg.max_p_streak:
            self._prev_gray = gray_small
            self._p_streak = 0
            return 'I'

        # Scene Change Score
        diff = cv2.absdiff(gray_small, self._prev_gray).astype(np.float32)
        delta_pixel = float(diff.mean()) / 255.0
        scs = self.cfg.w1 * delta_pixel + self.cfg.w2 * (1.0 - last_conf)

        self._prev_gray = gray_small
        self._last_conf = last_conf

        if scs >= self.cfg.theta:
            self._p_streak = 0
            return 'I'
        else:
            self._p_streak += 1
            return 'P'

    def record_latency(self, ms: float) -> bool:
        """Proactive tail latency alert. Returns True if alert fired."""
        self._latency_history.append(ms)
        if len(self._latency_history) < 10:
            return False
        mu = np.mean(self._latency_history)
        sigma = np.std(self._latency_history)
        if ms > mu + self.cfg.alert_sigma * sigma:
            self._alert_count += 1
            return True
        return False

    @property
    def alert_count(self) -> int:
        return self._alert_count
