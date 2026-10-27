"""
FixedRatioScheduler — 장면을 보지 않고 일정 간격으로 I-frame을 내는 기준선.

I 비율 r을 정하면 I-frame을 최대한 균등하게 배치한다 (r=0.37 → 간격 2·3 교대).
프레임 n(0부터)은 floor(n·r) > floor((n-1)·r) 이면 I-frame이다.
N프레임 중 I-frame 수는 1 + floor((N-1)·r).

OL-SCS와 같은 I 비율로 돌려, 연산량이 같을 때 "언제 추론하느냐"만의 효과를 비교하는 데 쓴다.
"""
import math

import numpy as np

from core.scheduler import CodecInspiredScheduler


class FixedRatioScheduler(CodecInspiredScheduler):
    def __init__(self, ratio: float):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1]: {ratio}")
        self.ratio = ratio
        self._n = 0

    def decide(self, frame: np.ndarray = None, last_conf: float = 1.0) -> str:
        n, self._n = self._n, self._n + 1
        if n == 0 or math.floor(n * self.ratio) > math.floor((n - 1) * self.ratio):
            return 'I'
        return 'P'
