"""
스케줄러 정확도 평가 — 매 프레임 전체 추론 결과를 기준으로, 각 스케줄러가
내보내는 결과(I-frame: 새 추론, P-frame: 재사용)가 얼마나 일치하는지 측정.

python eval_accuracy.py --video dashcam.mp4 [--max-frames N] [--out results/accuracy]

매칭: 같은 클래스, IoU ≥ --iou 인 1:1 그리디 매칭
recall    = 매칭된 기준 객체 / 기준 객체      (놓친 객체가 적을수록 ↑)
precision = 매칭된 출력 객체 / 출력 객체      (사라진 객체를 계속 내보낼수록 ↓)

I-frame 출력은 같은 프레임의 기준 추론과 동일하므로 (같은 모델, 같은 입력)
프레임당 추론 1회로 두 스케줄러를 함께 재생한다.

fixed: olscs와 같은 I 비율의 고정 간격 기준선. olscs의 I 비율은 영상을 다 본 뒤에야
정해지므로, 프레임별 기준 탐지를 저장해 두었다가 끝에서 재생한다.
"""
import argparse
import csv
from pathlib import Path

import cv2

from core.scheduler import CodecInspiredScheduler, SchedulerConfig
from core.object_scheduler import ObjectSelectiveScheduler, ObjectSchedulerConfig, _iou
from core.fixed_scheduler import FixedRatioScheduler
from perception.detector import ObjectDetector

FIELDS = ["video", "mode", "frames", "I_ratio",
          "recall", "precision", "recall_P", "precision_P", "ref_objects"]


def parse_args():
    p = argparse.ArgumentParser(description="PerceptDrive scheduler accuracy")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="yolov5s.pt")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out", default="results/accuracy")
    return p.parse_args()


def match(ref, out, thr):
    """같은 클래스 IoU ≥ thr 1:1 그리디 매칭 수."""
    pairs = sorted(((_iou(r, o), i, j) for i, r in enumerate(ref)
                    for j, o in enumerate(out) if r.cls == o.cls), reverse=True)
    used_r, used_o = set(), set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i not in used_r and j not in used_o:
            used_r.add(i)
            used_o.add(j)
    return len(used_r)


def main():
    args = parse_args()
    detector = ObjectDetector(args.weights)
    modes = {
        "frame": CodecInspiredScheduler(SchedulerConfig()),
        "olscs": ObjectSelectiveScheduler(ObjectSchedulerConfig()),
    }
    state = {m: {"dets": [], "conf": 1.0} for m in modes}
    # [기준 객체, 매칭, 출력 객체] — 전체 / P-frame만
    stats = {m: {"all": [0, 0, 0], "P": [0, 0, 0], "I": 0} for m in (*modes, "fixed")}
    refs = []

    cap = cv2.VideoCapture(args.video)
    n = 0
    while not args.max_frames or n < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        ref, ref_conf = detector.infer(frame)
        refs.append(ref)
        for m, sched in modes.items():
            st = state[m]
            if sched.decide(frame, last_conf=st["conf"]) == 'I':
                st["dets"], st["conf"] = ref, ref_conf
                if m == "olscs":
                    sched.update(ref)
                stats[m]["I"] += 1
                buckets = ("all",)
            else:
                buckets = ("all", "P")
            hit = match(ref, st["dets"], args.iou)
            for b in buckets:
                s = stats[m][b]
                s[0] += len(ref); s[1] += hit; s[2] += len(st["dets"])
        if n % 500 == 0:
            print(f"  {n} frames", flush=True)
    cap.release()

    fixed = FixedRatioScheduler(stats["olscs"]["I"] / n)
    dets = []
    for ref in refs:
        if fixed.decide() == 'I':
            dets = ref
            stats["fixed"]["I"] += 1
            buckets = ("all",)
        else:
            buckets = ("all", "P")
        hit = match(ref, dets, args.iou)
        for b in buckets:
            s = stats["fixed"][b]
            s[0] += len(ref); s[1] += hit; s[2] += len(dets)

    name = Path(args.video).stem
    rows = []
    for m in stats:
        a, p = stats[m]["all"], stats[m]["P"]
        rows.append({
            "video": name, "mode": m, "frames": n,
            "I_ratio": round(stats[m]["I"] / n, 4),
            "recall": round(a[1] / a[0], 4) if a[0] else None,
            "precision": round(a[1] / a[2], 4) if a[2] else None,
            "recall_P": round(p[1] / p[0], 4) if p[0] else None,
            "precision_P": round(p[1] / p[2], 4) if p[2] else None,
            "ref_objects": a[0],
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"accuracy_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'mode':<7}{'I_ratio':>9}{'recall':>9}{'prec':>9}{'recall_P':>10}{'prec_P':>9}")
    for r in rows:
        print(f"{r['mode']:<7}{r['I_ratio']:>9}{r['recall']:>9}{r['precision']:>9}"
              f"{r['recall_P']:>10}{r['precision_P']:>9}")
    print(f"\n[저장] {path}")


if __name__ == "__main__":
    main()
