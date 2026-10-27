# PerceptDrive 🚗

**Codec-Inspired Real-Time Autonomous Driving Perception Pipeline**

> 캡스톤 디자인 프로젝트(codec-inspired-realtime-inference)에서 개발한
> 실시간 추론 스케줄링 기술을 자율주행 인식 파이프라인에 적용한 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-red.svg)](https://github.com/ultralytics/yolov5)
[![TensorRT](https://img.shields.io/badge/TensorRT-INT8-green.svg)](https://developer.nvidia.com/tensorrt)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 개요

PerceptDrive는 다음 두 가지를 결합합니다:

| 캡스톤에서 가져온 기술 | 새로 추가하는 기술 |
|---|---|
| I/P-frame 스케줄러 (5.4× FPS) | 차선 인식 (LaneNet / HSV 파이프라인) |
| TensorRT INT8 (2.08×) | 단안 거리 추정 (MiDaS / DepthAnything) |
| Proactive 지연 예측기 | 조향 제어 신호 생성 |
| 멀티스트림 4카메라 (3.73×) | 주행 경로 계획 (기본) |

## 주요 결과 (목표)

- 객체 탐지: 728.9 FPS (스케줄러 ON, RTX 4060)
- 차선 인식: 실시간 (≥30 FPS on 720p)
- 거리 추정: ±0.5m 오차 (10m 이내)
- 전체 파이프라인: ≥60 FPS

## 아키텍처

```
Camera Input
    │
    ├─ CodecInspiredScheduler ──────────────────────────┐
    │   (I/P-frame 결정, SCS = w₁·Δpixel + w₂·(1-conf)) │
    │                                                    │
    ├─ ObjectDetector (YOLOv5s TensorRT INT8) ──────────┤
    │                                                    │
    ├─ LaneDetector (HSV + Hough / LaneNet) ────────────┤
    │                                                    │
    ├─ DepthEstimator (MiDaS Tiny) ─────────────────────┤
    │                                                    │
    └─ SteeringController ◄─────────────────────────────┘
            │
            └─ CAN Bus / RC Car / Simulator
```

## 설치

```bash
git clone https://github.com/amgalantur/PerceptDrive.git
cd PerceptDrive
pip install -r requirements.txt
```

## 사용법

```bash
# 기본 데모 (웹캠)
python demo.py --source 0

# 영상 파일
python demo.py --source test_video.mp4 --show-lanes --show-depth

# TensorRT INT8 가속
python demo.py --source 0 --trt-int8
```

## 프로젝트 구조

```
PerceptDrive/
├── core/
│   ├── scheduler.py          # CodecInspiredScheduler (캡스톤)
│   ├── tail_latency.py       # ProactiveTailLatencyPredictor (캡스톤)
│   └── object_scheduler.py  # ObjectSelectiveScheduler (캡스톤)
├── perception/
│   ├── detector.py           # YOLOv5s / TensorRT INT8
│   ├── lane_detector.py      # 차선 인식
│   └── depth_estimator.py   # 단안 거리 추정
├── control/
│   └── steering.py           # 조향 제어 신호
├── utils/
│   ├── visualizer.py         # 시각화
│   └── profiler.py           # 성능 측정
├── demo.py                   # 메인 데모
├── benchmark.py              # 성능 벤치마크
├── requirements.txt
└── README.md
```

## 로드맵

- [x] 캡스톤 스케줄러 이식
- [x] YOLOv5s TensorRT INT8
- [ ] 차선 인식 모듈
- [ ] 단안 거리 추정
- [ ] 조향 제어 신호
- [ ] CARLA 시뮬레이터 연동
- [ ] RC카 실차 테스트
- [ ] 논문 작성

## 관련 프로젝트

- [codec-inspired-realtime-inference](https://github.com/amgalantur/codec-inspired-realtime-inference) — 기반 캡스톤 프로젝트

## 저자

- **암갈란투르** (21101198) — 서울과학기술대학교
- 지도교수: 김성곤 교수님

## 라이선스

MIT License
