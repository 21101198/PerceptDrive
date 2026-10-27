"""
PerceptDrive 메인 데모
python demo.py --source test.mp4 [--scheduler olscs|frame] [--show-lanes] [--show-depth] [--trt-int8]
              [--output out.mp4]
"""
import argparse
import time
import cv2
import numpy as np
import sys

from core.scheduler import CodecInspiredScheduler, SchedulerConfig
from core.object_scheduler import ObjectSelectiveScheduler, ObjectSchedulerConfig
from core.fixed_scheduler import FixedRatioScheduler
from perception.detector import ObjectDetector
from perception.lane_detector import HoughLaneDetector, LaneResult
from control.steering import SteeringController, ControlOutput
from pd_utils.visualizer import draw_hud


class Pipeline:
    """스케줄러 + 탐지 + 차선 인식 1프레임 처리 (demo / benchmark 공용)."""

    def __init__(self, mode: str, detector: ObjectDetector, fixed_ratio: float = None):
        self.mode = mode
        self.detector = detector
        self.lane_det = HoughLaneDetector()
        if mode == "olscs":
            self.scheduler = ObjectSelectiveScheduler(ObjectSchedulerConfig())
        elif mode == "fixed":
            self.scheduler = FixedRatioScheduler(fixed_ratio)
        else:
            self.scheduler = CodecInspiredScheduler(SchedulerConfig())
        self._last_dets, self._last_lane = [], LaneResult()

    def step(self, frame: np.ndarray):
        """Returns (frame_type, dets, lane_result)"""
        frame_type = self.scheduler.decide(frame, last_conf=self.detector.last_conf)
        if frame_type == 'I':
            # 전체 추론
            self._last_dets, _ = self.detector.infer(frame)
            self._last_lane = self.lane_det.detect(frame)
            if self.mode == "olscs":
                self.scheduler.update(self._last_dets)
        # P-frame: 이전 결과 재사용
        return frame_type, self._last_dets, self._last_lane


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PerceptDrive Demo")
    p.add_argument("--source",      default="0",  help="영상 소스 (0=웹캠, 파일경로)")
    p.add_argument("--weights",     default="yolov5s.pt")
    p.add_argument("--backend",     default="torch",
                   choices=["torch", "trt_fp16", "trt_int8"])
    p.add_argument("--scheduler",   default="olscs",
                   choices=["olscs", "frame"],
                   help="olscs=객체 단위 OL-SCS, frame=프레임 단위 SCS")
    p.add_argument("--show-lanes",  action="store_true")
    p.add_argument("--show-depth",  action="store_true")
    p.add_argument("--no-lanes",    action="store_true", help="차선 라인 그리기 생략")
    p.add_argument("--no-display",  action="store_true")
    p.add_argument("--max-frames",  type=int, default=0)
    p.add_argument("--output",      default="", help="HUD 결과 영상 저장 경로 (.mp4)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = int(args.source) if args.source.isdigit() else args.source

    # 모듈 초기화
    detector     = ObjectDetector(args.weights, args.backend)
    pipeline     = Pipeline(args.scheduler, detector)
    scheduler    = pipeline.scheduler
    controller   = SteeringController()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 소스 열기 실패: {source}")
        sys.exit(1)

    writer = None
    if args.output:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (w, h))
        if not writer.isOpened():
            print(f"[ERROR] 출력 파일 열기 실패: {args.output}")
            sys.exit(1)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[PerceptDrive] 저장 → {args.output} ({w}x{h} @ {src_fps:.1f}fps, {total}프레임)")

    WARMUP_FRAMES = 30            # 모델 워밍업 구간은 FPS 계산에서 제외
    warmup_end = None
    fps_display = 0.0
    last_control = ControlOutput(0.0, 0.5, 0.0, "Init")

    print("[PerceptDrive] 시작 — 'q'로 종료")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        t_frame = time.perf_counter()

        frame_type, dets, lane_result = pipeline.step(frame)

        # 제어
        control = controller.compute(lane_result, dets,
                                     frame_width=frame.shape[1])

        # FPS
        infer_ms = (time.perf_counter() - t_frame) * 1000
        scheduler.record_latency(infer_ms)

        if frame_idx > WARMUP_FRAMES:
            fps_display = (frame_idx - WARMUP_FRAMES) / (time.perf_counter() - warmup_end)
        else:
            if frame_idx == WARMUP_FRAMES:
                warmup_end = time.perf_counter()
            fps_display = 0.0

        # 시각화
        if writer is not None or not args.no_display:
            vis = draw_hud(frame, dets, lane_result,
                           control, fps_display, frame_type,
                           draw_lanes=not args.no_lanes)
        if writer is not None:
            writer.write(vis)
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}프레임 처리", flush=True)
        if not args.no_display:
            cv2.imshow("PerceptDrive", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    summary = (f"\n[완료] 총 {frame_idx}프레임 | "
               f"Proactive 경보: {scheduler.alert_count}회")
    if args.scheduler == "olscs":
        c = scheduler.counts
        summary += f" | I/P = {c['I']}/{c['P']} | I-frame 원인: {dict(scheduler.reasons)}"
    print(summary)


if __name__ == "__main__":
    main()
