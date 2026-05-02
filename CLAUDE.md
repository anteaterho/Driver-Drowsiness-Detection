# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate drowsiness
```

## Running the App

```bash
python app.py
```

Gradio가 로컬 서버를 열고 브라우저에서 웹캠을 사용할 수 있습니다 (기본 http://localhost:7860).

## Architecture

### Key Parameters (`detector.py`)
| 상수 | 기본값 | 설명 |
|------|--------|------|
| `EAR_THRESHOLD` | 0.22 | 이 값 미만이면 눈이 감긴 것으로 판단 |
| `CONSEC_FRAMES` | 20 | 연속 감힘 프레임 수 → 졸음 판정 |

### Data Flow

```
Gradio webcam (RGB numpy array)
    ↓
process_frame(frame, state)  [app.py]
    ↓
DrowsinessDetector.detect()  [detector.py]
    ├─ MediaPipe FaceMesh → 468-point landmarks
    ├─ EAR = (A + B) / (2 * C)  for each eye
    └─ counter 갱신 (EAR < threshold이면 증가, 아니면 0으로 리셋)
    ↓
OpenCV로 eye contour 및 상태 텍스트 오버레이 (RGB 색상)
    ↓
Gradio outputs: (annotated_frame, updated_state, status_text)
```

### Eye Landmark Indices (MediaPipe 468-point model)
- **Right eye**: `[33, 160, 158, 133, 153, 144]`
- **Left eye**: `[362, 385, 387, 263, 373, 380]`

### Color Convention
프레임이 Gradio에서 RGB로 전달되므로 **OpenCV 드로잉 색상도 RGB 순서**로 지정:
- 정상: `(0, 255, 0)` (초록)
- 주의: `(255, 165, 0)` (주황)
- 졸음: `(255, 0, 0)` (빨강)
