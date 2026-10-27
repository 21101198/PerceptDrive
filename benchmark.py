"""
PerceptDrive 벤치마크 — OL-SCS(olscs) vs 프레임 단위 SCS(frame)
같은 영상을 두 모드로 처리해 FPS, 지연시간, I/P 비율 비교 → CSV 저장

python benchmark.py [--videos a.mp4 b.mp4 ...] [--max-frames N] [--out results]

지연시간 = 1프레임 파이프라인 처리 시간 (스케줄러 + 탐지 + 차선 + 제어, 디코딩 제외)
pipeline_fps = 처리 프레임 / 지연시간 합,  wall_fps = 처리 프레임 / 디코딩 포함 전체 시간
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from demo import Pipeline
from perception.detector import ObjectDetector
from control.steering import SteeringController

DEFAULT_VIDEOS = [
    r"C:\Users\USER\OneDrive\Desktop\proj2\videos\crowd.mp4",
    r"C:\Users\USER\OneDrive\Desktop\proj2\videos\highspeed.mp4",
    r"C:\Users\USER\OneDrive\Desktop\proj2\videos\lowlight.mp4",
]
# SCS off(프레임 단위) / on(객체 단위 OL-SCS) / 고정 간격(같은 영상 olscs의 I 비율에 맞춤)
MODES = ["frame", "olscs", "fixed"]

SUMMARY_FIELDS = [
    "video", "mode", "frames", "pipeline_fps", "wall_fps", "speedup",
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms",
    "I_ratio", "P_ratio", "I", "P", "i_reasons", "proactive_alerts",
]
FRAME_FIELDS = ["video", "mode", "frame", "frame_type", "latency_ms", "n_dets"]


def parse_args():
    p = argparse.ArgumentParser(description="PerceptDrive Benchmark")
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    p.add_argument("--weights", default="yolov5s.pt")
    p.add_argument("--backend", default="torch",
                   choices=["torch", "trt_fp16", "trt_int8"])
    p.add_argument("--max-frames", type=int, default=0, help="영상당 최대 프레임 (0=전체)")
    p.add_argument("--warmup", type=int, default=10, help="측정 전 워밍업 추론 횟수")
    p.add_argument("--out", default="results")
    return p.parse_args()


def warmup(detector: ObjectDetector, video: str, n: int):
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    for _ in range(n):
        detector.infer(frame)


def run(video: str, mode: str, detector: ObjectDetector, max_frames: int, frame_rows: list,
        fixed_ratio: float = None):
    pipeline = Pipeline(mode, detector, fixed_ratio)
    controller = SteeringController()
    name = Path(video).stem

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"소스 열기 실패: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    latencies, counts = [], {"I": 0, "P": 0}
    t_wall = time.perf_counter()
    while not max_frames or len(latencies) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        frame_type, dets, lane = pipeline.step(frame)
        controller.compute(lane, dets, frame_width=frame.shape[1])
        ms = (time.perf_counter() - t0) * 1000
        pipeline.scheduler.record_latency(ms)

        latencies.append(ms)
        counts[frame_type] += 1
        frame_rows.append([name, mode, len(latencies), frame_type, round(ms, 3), len(dets)])
        if len(latencies) % 500 == 0:
            print(f"  [{name}/{mode}] {len(latencies)}/{total} "
                  f"({1000 / np.mean(latencies[-500:]):.1f} FPS)", flush=True)
    wall = time.perf_counter() - t_wall
    cap.release()

    lat = np.array(latencies)
    n = len(lat)
    return {
        "video": name, "mode": mode, "frames": n,
        "pipeline_fps": round(n / (lat.sum() / 1000), 2),
        "wall_fps": round(n / wall, 2),
        "speedup": None,
        "lat_mean_ms": round(lat.mean(), 3),
        "lat_p50_ms": round(np.percentile(lat, 50), 3),
        "lat_p95_ms": round(np.percentile(lat, 95), 3),
        "lat_p99_ms": round(np.percentile(lat, 99), 3),
        "lat_max_ms": round(lat.max(), 3),
        **{f"{k}_ratio": round(v / n, 4) for k, v in counts.items()},
        **counts,
        "i_reasons": dict(getattr(pipeline.scheduler, "reasons", {})),
        "proactive_alerts": pipeline.scheduler.alert_count,
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    detector = ObjectDetector(args.weights, args.backend)
    warmup(detector, args.videos[0], args.warmup)

    summary, frame_rows = [], []
    for video in args.videos:
        base, olscs_ratio = None, None
        for mode in MODES:
            print(f"[Benchmark] {Path(video).name} — {mode}", flush=True)
            r = run(video, mode, detector, args.max_frames, frame_rows, olscs_ratio)
            if mode == "frame":
                base = r["pipeline_fps"]
            elif mode == "olscs":
                olscs_ratio = r["I"] / r["frames"]
            r["speedup"] = round(r["pipeline_fps"] / base, 3) if base else None
            summary.append(r)
            print(f"  → {r['frames']}f | {r['pipeline_fps']} FPS | "
                  f"p95 {r['lat_p95_ms']}ms | I/P {r['I']}/{r['P']}", flush=True)

    with open(out / "benchmark_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    with open(out / "benchmark_frames.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FRAME_FIELDS)
        w.writerows(frame_rows)

    print(f"\n{'video':<12}{'mode':<7}{'FPS':>9}{'speedup':>9}{'mean':>9}"
          f"{'p95':>9}{'p99':>9}   I/P ratio")
    for r in summary:
        print(f"{r['video']:<12}{r['mode']:<7}{r['pipeline_fps']:>9}{r['speedup']:>9}"
              f"{r['lat_mean_ms']:>9}{r['lat_p95_ms']:>9}{r['lat_p99_ms']:>9}   "
              f"{r['I_ratio']:.2f}/{r['P_ratio']:.2f}  {r['i_reasons'] or ''}")
    print(f"\n[저장] {out / 'benchmark_summary.csv'}, {out / 'benchmark_frames.csv'}")


if __name__ == "__main__":
    main()
