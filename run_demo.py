"""
발표용 결과 영상 생성 (HUD 오버레이 포함 mp4 저장, 창 없이 실행)
python run_demo.py --video dashcam.mp4 --output demo_output.mp4 [--scheduler olscs|frame] [--max-frames N]
"""
import argparse
import sys
import demo

# Windows 콘솔/파이프에서 한글 출력 시 UnicodeEncodeError 방지
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description="PerceptDrive 결과 영상 생성")
    p.add_argument("--video",      required=True, help="입력 영상 경로")
    p.add_argument("--output",     default="demo_output.mp4")
    p.add_argument("--weights",    default="yolov5s.pt")
    p.add_argument("--backend",    default="torch", choices=["torch", "trt_fp16", "trt_int8"])
    p.add_argument("--scheduler",  default="olscs", choices=["olscs", "frame"])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-lanes",   action="store_true", help="차선 라인 그리기 생략")
    p.add_argument("--show",       action="store_true", help="저장하면서 화면에도 표시")
    a = p.parse_args()

    argv = ["--source", a.video, "--output", a.output, "--weights", a.weights,
            "--backend", a.backend, "--scheduler", a.scheduler,
            "--max-frames", str(a.max_frames)]
    if a.no_lanes:
        argv.append("--no-lanes")
    if not a.show:
        argv.append("--no-display")
    demo.main(argv)


if __name__ == "__main__":
    main()
