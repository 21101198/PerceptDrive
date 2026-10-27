"""
ObjectSelectiveScheduler — OL-SCS (Object-Level Scene Change Score)

"어떤 객체를 재추론할지"가 아니라 "이 프레임을 추론할지"를 객체 상태로 결정하는
I/P 2단계 스케줄러. (ROI만 재추론하는 O-frame은 모델 호출 고정비 때문에 P-frame
대비 ~20배 비싸 오히려 느려져 제거)

객체 i의 점수:
    OL-SCS_i = w1 * Δpixel_i + w2 * (1 - conf_i) * Δpixel_i
      Δpixel_i : 객체 i의 (패딩된) 박스 영역에서, 마지막 추론 시점 대비
                 변화 픽셀 비율 (|Δgray| > pixel_thresh). 평균 밝기 차분은 객체/배경
                 대비에 좌우돼 이동을 과소평가하므로 비율을 사용.
      conf_i   : 객체 i의 마지막 탐지 신뢰도. 움직임에 비례해서만 점수를 키운다 —
                 상수 항으로 더하면 자차 이동으로 늘 조금씩 변하는 작은 객체가
                 신뢰도만으로 θ_obj를 넘어 매 프레임 I-frame을 유발한다.

판정 대상: conf ≥ min_conf 이고 박스 면적 ≥ min_box_area 인 객체만.
    나머지(저신뢰·초소형, 오탐 다수)는 결과에는 남기되 I/P 판정에서 제외.

객체별 P-streak (판정 대상만):
    p_streak_i : 객체 i가 마지막으로 추론된 이후 경과 프레임
    limit_i    = min(min_obj_streak + stable_i, max_obj_age)
    stable_i   : 객체 i가 연속으로 같은 자리(IoU ≥ stable_iou)에서 재탐지된 I-frame 수.
                 I-frame 사이에 IoU로 객체를 이어 붙여 유지 — 오래 안정적인 객체일수록
                 길게 재사용하고, 새로 나타난 객체는 빨리 다시 확인한다.
    static_i   : 직전 I-frame 대비 bbox IoU ≥ static_iou (거의 안 움직임) → limit_i 판정
                 제외. P-frame에는 새 bbox가 없으므로 연속한 두 I-frame의 bbox로 판단.
                 움직이기 시작하면 OL-SCS_i, 최악의 경우 max_i_interval이 잡는다.

I-frame 조건 (하나라도 만족):
    첫 프레임 / max_i_interval 경과 / 배경 변화 ≥ θ_bg /
    판정 대상 객체 중 OL-SCS_i ≥ θ_obj 또는 (static_i 아님 and p_streak_i ≥ limit_i)
그 외 → P-frame (전체 결과 재사용)

배경(객체 박스 밖) 변화는 직전 프레임 대비 그리드 셀별 평균 차분의 최댓값으로
측정해 새 객체 진입처럼 국소적인 변화도 놓치지 않도록 한다. 마지막 I-frame과
비교하면 자차 이동(ego-motion)으로 차분이 누적돼 거의 매 프레임 I-frame이 된다.
"""
import numpy as np
import cv2
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.scheduler import CodecInspiredScheduler, SchedulerConfig
from perception.detector import Detection

Box = Tuple[int, int, int, int]


@dataclass
class ObjectSchedulerConfig(SchedulerConfig):
    theta_obj: float = 0.25      # 객체 OL-SCS 임계값 → I-frame
    theta_bg: float = 0.15       # 배경 셀 변화 임계값 → I-frame
    max_i_interval: int = 15     # 강제 I-frame 주기 (객체 없는 장면 포함)
    min_obj_streak: int = 4      # 새 객체의 허용 P-streak
    max_obj_age: int = 8         # 안정 객체의 최대 허용 P-streak
    stable_iou: float = 0.5      # I-frame 간 같은 객체로 이어 붙이는 IoU
    static_iou: float = 0.85     # 이 IoU 이상이면 정지 객체 → 재사용 한도 판정 제외
    pixel_thresh: int = 20       # Δpixel_i 변화 픽셀 판정 임계값 (gray 0~255)
    min_conf: float = 0.4        # 판정 대상 최소 신뢰도
    min_box_area: float = 0.001  # 판정 대상 최소 박스 면적 (프레임 대비, 1280×720 기준 ~30×30px)
    patch_pad: float = 0.2       # Δpixel_i 측정 영역 패딩 (박스 크기 대비)
    bg_grid: Tuple[int, int] = (8, 6)   # 배경 변화 측정 그리드 (cols, rows)
    proc_width: int = 160        # 차분 계산용 다운스케일 폭


@dataclass
class _Track:
    det: Detection
    ref_patch: np.ndarray        # 마지막 추론 시점의 gray patch (다운스케일 좌표)
    patch_box: Box               # ref_patch를 잘라낸 위치 (다운스케일 좌표)
    stable: int = 0              # 같은 자리에서 연속 재탐지된 I-frame 수
    p_streak: int = 0            # 마지막 추론 이후 경과 프레임
    judged: bool = True          # I/P 판정 대상 여부 (min_conf, min_box_area)
    static: bool = False         # 직전 I-frame 대비 IoU ≥ static_iou


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a.x2 - a.x1) * (a.y2 - a.y1)
             + (b.x2 - b.x1) * (b.y2 - b.y1) - inter)
    return inter / union if union > 0 else 0.0


class ObjectSelectiveScheduler(CodecInspiredScheduler):
    """
    OL-SCS 기반 I/P 스케줄러.
    record_latency / alert_count 는 CodecInspiredScheduler에서 상속.

    사용:
        if sched.decide(frame) == 'I':
            dets, _ = detector.infer(frame)
            sched.update(dets)
    """

    def __init__(self, config: ObjectSchedulerConfig = None):
        super().__init__(config or ObjectSchedulerConfig())
        self.cfg: ObjectSchedulerConfig
        self._tracks: List[_Track] = []
        self._scale: float = 1.0
        self._frame_hw: Tuple[int, int] = (0, 0)
        self._cur_gray: Optional[np.ndarray] = None
        self._bg_mask: Optional[np.ndarray] = None   # 객체 영역 제외 마스크 (float32 0/1)
        self._bg_valid: Optional[np.ndarray] = None  # 평가할 그리드 셀 (배경 비율 > 30%)
        self._bg_inv_cnt: Optional[np.ndarray] = None  # 유효 셀의 1 / (배경 비율 · 255)
        self._since_i: int = 0
        self.counts = Counter()      # 'I' / 'P'
        self.reasons = Counter()     # I-frame 원인

    # ── 결정 ──────────────────────────────────────────
    def decide(self, frame: np.ndarray, last_conf: float = 1.0) -> str:
        """Returns 'I' (전체 추론) or 'P' (재사용). last_conf는 인터페이스 호환용."""
        h, w = frame.shape[:2]
        self._frame_hw = (h, w)
        self._scale = self.cfg.proc_width / w
        # 축소 먼저, 회색 변환은 작은 이미지에서 (1280×720 전체 변환 회피)
        small = cv2.resize(frame, (self.cfg.proc_width, max(1, round(h * self._scale))),
                           interpolation=cv2.INTER_AREA)
        self._cur_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev_gray, self._prev_gray = self._prev_gray, self._cur_gray

        if self._bg_mask is None:
            return self._emit('first')
        if self._since_i >= self.cfg.max_i_interval:
            return self._emit('interval')
        if self._background_score(prev_gray) >= self.cfg.theta_bg:
            return self._emit('background')

        judged = [t for t in self._tracks if t.judged]
        for t in judged:
            t.p_streak += 1
        for t in judged:
            if not t.static and t.p_streak >= min(self.cfg.min_obj_streak + t.stable, self.cfg.max_obj_age):
                return self._emit('obj_streak')
            if self._ol_scs(t) >= self.cfg.theta_obj:
                return self._emit('obj_motion')
        return self._emit(None)

    def _emit(self, reason: Optional[str]) -> str:
        frame_type = 'P' if reason is None else 'I'
        self._since_i = 0 if frame_type == 'I' else self._since_i + 1
        self.counts[frame_type] += 1
        if reason:
            self.reasons[reason] += 1
        return frame_type

    def _ol_scs(self, t: _Track) -> float:
        x1, y1, x2, y2 = t.patch_box
        diff = cv2.absdiff(self._cur_gray[y1:y2, x1:x2], t.ref_patch)
        delta_pixel = float((diff > self.cfg.pixel_thresh).mean())
        return (self.cfg.w1 + self.cfg.w2 * (1.0 - t.det.conf)) * delta_pixel

    def _background_score(self, prev_gray: np.ndarray) -> float:
        """객체 영역 밖에서 직전 프레임 대비 그리드 셀별 평균 차분의 최댓값 (0~1).
        마스크 관련 값(셀별 배경 비율, 유효 셀)은 마스크가 바뀌는 update()에서 미리 계산."""
        if self._bg_inv_cnt.size == 0:
            return 0.0
        diff = cv2.absdiff(self._cur_gray, prev_gray).astype(np.float32)
        cell_sum = cv2.resize(diff * self._bg_mask, self.cfg.bg_grid,
                              interpolation=cv2.INTER_AREA)
        return float((cell_sum[self._bg_valid] * self._bg_inv_cnt).max())

    # ── 상태 갱신 ─────────────────────────────────────
    def update(self, dets: List[Detection]) -> List[Detection]:
        """I-frame 추론 결과 반영. 이전 객체와 IoU로 이어 붙여 stable 카운터 유지."""
        prev, tracks = self._tracks, []
        h, w = self._frame_hw
        for d in dets:
            best = max(prev, key=lambda t: _iou(t.det, d), default=None)
            iou = _iou(best.det, d) if best is not None and best.det.cls == d.cls else 0.0
            same = iou >= self.cfg.stable_iou
            pb = self._padded_box(d)
            x1, y1, x2, y2 = pb
            area = (d.x2 - d.x1) * (d.y2 - d.y1) / (h * w)
            tracks.append(_Track(d, self._cur_gray[y1:y2, x1:x2].copy(), pb,
                                 stable=best.stable + 1 if same else 0,
                                 static=iou >= self.cfg.static_iou,
                                 judged=d.conf >= self.cfg.min_conf
                                 and area >= self.cfg.min_box_area))
        self._tracks = tracks

        self._bg_mask = np.ones_like(self._cur_gray, dtype=np.float32)
        for t in self._tracks:
            x1, y1, x2, y2 = t.patch_box
            self._bg_mask[y1:y2, x1:x2] = 0.0
        cell_cnt = cv2.resize(self._bg_mask, self.cfg.bg_grid, interpolation=cv2.INTER_AREA)
        self._bg_valid = cell_cnt > 0.3   # 셀의 30% 이상이 배경일 때만 평가
        self._bg_inv_cnt = 1.0 / (cell_cnt[self._bg_valid] * 255.0)
        return self.detections

    @property
    def detections(self) -> List[Detection]:
        return [t.det for t in self._tracks]

    def _padded_box(self, d: Detection) -> Box:
        """patch_pad만큼 패딩·클리핑한 박스 (다운스케일 좌표)."""
        h, w = self._frame_hw
        s = self._scale
        px = (d.x2 - d.x1) * self.cfg.patch_pad
        py = (d.y2 - d.y1) * self.cfg.patch_pad
        x1 = min(int(max(0, d.x1 - px) * s), int(w * s) - 1)
        y1 = min(int(max(0, d.y1 - py) * s), int(h * s) - 1)
        x2 = int(np.ceil(min(w, d.x2 + px) * s))
        y2 = int(np.ceil(min(h, d.y2 + py) * s))
        return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)
