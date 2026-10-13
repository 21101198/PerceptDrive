"""
PerceptDrive 메인 데모
python demo.py --source test.mp4 [--show-lanes] [--show-depth] [--trt-int8]
"""
import argparse
import time
import cv2
import numpy as np
import sys

from core.scheduler import CodecInspiredScheduler, SchedulerConfig
from perception.detector import ObjectDetector
from perception.lane_detector import HoughLaneDetector, LaneResult
from control.steering import SteeringController, ControlOutput
from utils.visualizer import draw_hud


def parse_args():
    p = argparse.ArgumentParser(description="PerceptDrive Demo")
    p.add_argument("--source",      default="0",  help="영상 소스 (0=웹캠, 파일경로)")
    p.add_argument("--weights",     default="yolov5s.pt")
    p.add_argument("--backend",     default="torch",
                   choices=["torch", "trt_fp16", "trt_int8"])
    p.add_argument("--show-lanes",  action="store_true")
    p.add_argument("--show-depth",  action="store_true")
    p.add_argument("--no-display",  action="store_true")
    p.add_argument("--max-frames",  type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # 모듈 초기화
    scheduler    = CodecInspiredScheduler(SchedulerConfig())
    detector     = ObjectDetector(args.weights, args.backend)
    lane_det     = HoughLaneDetector()
    controller   = SteeringController()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 소스 열기 실패: {source}")
        sys.exit(1)

    fps_counter, t_start = 0, time.perf_counter()
    fps_display = 0.0
    last_dets, last_lane = [], LaneResult()
    last_control = ControlOutput(0.0, 0.5, 0.0, "초기화")

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

        # 스케줄러 결정
        frame_type = scheduler.decide(frame, last_conf=detector.last_conf)

        if frame_type == 'I':
            # 전체 추론
            dets, _ = detector.infer(frame)
            lane_result = lane_det.detect(frame)
            last_dets = dets
            last_lane = lane_result
        else:
            # P-frame: 이전 결과 재사용
            dets = last_dets
            lane_result = last_lane

        # 제어
        control = controller.compute(lane_result, dets,
                                     frame_width=frame.shape[1])

        # FPS
        infer_ms = (time.perf_counter() - t_frame) * 1000
        scheduler.record_latency(infer_ms)

        fps_counter += 1
        elapsed = time.perf_counter() - t_start
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            t_start = time.perf_counter()

        # 시각화
        if not args.no_display:
            vis = draw_hud(frame, dets, lane_result,
                           control, fps_display, frame_type)
            cv2.imshow("PerceptDrive", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[완료] 총 {frame_idx}프레임 | "
          f"Proactive 경보: {scheduler.alert_count}회")


if __name__ == "__main__":
    main()
