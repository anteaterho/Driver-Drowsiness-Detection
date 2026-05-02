# Driver Drowsiness Detection

웹캠 영상에서 얼굴과 눈 랜드마크를 추적해 EAR(Eye Aspect Ratio)를 계산하고, 일정 시간 이상 눈을 감고 있으면 졸음 상태로 판단하는 실시간 감지 프로젝트입니다. Gradio 기반 UI로 브라우저에서 바로 실행할 수 있고, 경고 문구와 사운드 알림을 함께 제공합니다.

## Features

- MediaPipe Face Landmarker 기반 눈 랜드마크 추적
- EAR 기반 실시간 눈 감김 감지
- 연속 프레임 누적 방식의 졸음 판정
- Gradio 웹 UI를 통한 실시간 결과 확인
- EAR 임계값 슬라이더 조절
- 브라우저 내장 경고음 재생
- 향후 아두이노/진동 모터 연동 확장 가능

## Project Structure

- `app.py`: Gradio UI, 웹캠 캡처 루프, 화면 렌더링, 경고음 제어
- `detector.py`: 얼굴 랜드마크 추출, EAR 계산, 졸음 판정 로직
- `environment.yml`: Conda 실행 환경 정의
- `face_landmarker.task`: MediaPipe 모델 파일

## Requirements

- Python 3.10
- Conda 또는 Miniconda
- 웹캠 사용 가능 환경

## Setup

```bash
conda env create -f environment.yml
conda activate drowsiness
```

## Run

```bash
python app.py
```

실행 후 Gradio 로컬 서버가 열리며, 일반적으로 브라우저에서 `http://localhost:7860`으로 접속할 수 있습니다.

## How It Works

1. 웹캠 프레임을 RGB 형식으로 읽습니다.
2. MediaPipe Face Landmarker로 얼굴 랜드마크를 검출합니다.
3. 양쪽 눈의 6개 포인트를 사용해 EAR를 계산합니다.
4. EAR가 기준값보다 작으면 눈 감김 프레임 카운트를 증가시킵니다.
5. 카운트가 지정된 프레임 수 이상 누적되면 졸음 상태로 판정합니다.
6. 결과를 화면 오버레이와 상태 메시지, 경고음으로 사용자에게 전달합니다.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | `0.22` | 이 값보다 작으면 눈을 감은 것으로 판단 |
| `CONSEC_FRAMES` | `20` | 연속 눈 감김 프레임 수가 이 값 이상이면 졸음 판정 |

## Notes

- `face_landmarker.task` 파일이 없으면 실행 시 자동으로 다운로드됩니다.
- Gradio와 브라우저 오디오 정책 때문에 경고음을 사용하려면 UI에서 한 번 오디오 초기화를 해야 할 수 있습니다.
- Windows 환경에서는 웹캠 접근 권한과 브라우저 권한이 필요할 수 있습니다.

## Future Extension

현재 프로젝트는 브라우저 경고음까지 구현되어 있으며, 다음 단계로는 Python 애플리케이션에서 시리얼 통신으로 상태를 보내고 아두이노가 진동 모터를 구동하도록 확장할 수 있습니다.

예시 흐름:

- Python: 졸음 상태 감지 후 시리얼로 상태 코드 전송
- Arduino: 상태 코드 수신 후 진동 모터 패턴 제어
- Motor Driver: 트랜지스터 또는 MOSFET으로 안전하게 모터 구동

## License

개인 학습 및 실험용 프로젝트입니다. 필요하면 여기에 원하는 라이선스를 추가해 사용할 수 있습니다.