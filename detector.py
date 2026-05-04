import urllib.request  # 인터넷에서 모델 파일을 다운로드할 때 사용
from pathlib import Path  # 운영체제와 관계없이 파일 경로를 다루기 쉽게 해줌

import mediapipe as mp  # 얼굴 랜드마크(눈, 코, 입 점들) 검출용 핵심 라이브러리
import numpy as np  # 거리 계산 등 수치 연산에 사용
from mediapipe.tasks import python as mp_python  # MediaPipe Tasks의 파이썬용 기본 옵션 클래스 제공
from mediapipe.tasks.python import vision as mp_vision  # 얼굴 랜드마커 같은 비전 작업 클래스 제공

_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

# MediaPipe 478-point face mesh — EAR 계산에 사용되는 6점 인덱스
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# EAR(Eye Aspect Ratio, 눈 개폐 비율) 기준값.
# 일반적으로 EAR가 크면 눈이 떠 있고, 작으면 눈이 감긴 상태에 가깝다.
# 이 프로젝트에서는 EAR가 0.22보다 작아지면 "눈 감김"으로 처리한다.
EAR_THRESHOLD = 0.22
# 눈 감김이 이 프레임 수 이상 이어지면 졸음으로 판정한다.
CONSEC_FRAMES = 20

# 고개 자세(좌우 회전·숙임) UI 슬라이더 기본값 — 환경에 맞게 앱에서 조정
HEAD_YAW_THRESHOLD_DEFAULT = 0.35
HEAD_PITCH_THRESHOLD_DEFAULT = 0.38

# MediaPipe 얼굴 랜드마크: 코 끝, 턱(아래쪽) — 고개 각도 근사에 사용
NOSE_TIP_IDX = 1
CHIN_IDX = 152


def _ensure_model() -> None:
    # 모델 파일이 없으면 처음 한 번만 자동 다운로드한다.
    if not _MODEL_PATH.exists():
        print("face_landmarker.task 모델 다운로드 중 (~29 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("다운로드 완료")


def _dist(p1, p2) -> float:
    # 두 점 사이의 직선 거리를 계산한다.
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _eye_aspect_ratio(pts: list) -> float:
    """EAR = (A + B) / (2 * C)  — A,B: 수직, C: 수평"""
    # EAR 해석:
    # - 분자(A+B): 눈의 세로(위아래) 열림 정도
    # - 분모(2*C): 눈의 가로 길이(사람마다 다른 크기 보정)
    # 즉, EAR는 "눈 모양의 비율"이라 얼굴이 커지거나 작아져도 비교적 안정적이다.
    # 눈이 감기면 세로 길이(A, B)가 줄어 EAR도 함께 작아진다.
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def _get_pts(landmarks, indices: list, w: int, h: int) -> list:
    # 정규화 좌표(0~1)를 실제 화면 픽셀 좌표로 변환한다.
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]


def _mean_norm_xy(landmarks, indices: list) -> tuple[float, float]:
    """랜드마크 인덱스들의 정규화 좌표(0~1) 평균."""
    xs = [float(landmarks[i].x) for i in indices]
    ys = [float(landmarks[i].y) for i in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def _head_pose_metrics(landmarks) -> tuple[float, float]:
    """
    정규화 좌표 기준 근사 지표.
    - yaw: 코가 양눈 중점에서 벗어난 정도 / 눈 간 거리 (좌우 돌림이 클수록 커짐)
    - pitch: (코_y - 눈중_y) / (턱_y - 눈중_y) — 고개를 숙이면 보통 값이 커짐
    """
    lx, ly = _mean_norm_xy(landmarks, LEFT_EYE)
    rx, ry = _mean_norm_xy(landmarks, RIGHT_EYE)
    nx = float(landmarks[NOSE_TIP_IDX].x)
    ny = float(landmarks[NOSE_TIP_IDX].y)
    cy = float(landmarks[CHIN_IDX].y)

    eye_mx = (lx + rx) * 0.5
    eye_my = (ly + ry) * 0.5
    ipd = float(np.hypot(rx - lx, ry - ly)) + 1e-6
    yaw_m = abs(nx - eye_mx) / ipd

    denom = (cy - eye_my) + 1e-6
    if denom <= 1e-5:
        pitch_m = 0.0
    else:
        pitch_m = (ny - eye_my) / denom
    return yaw_m, pitch_m


class DrowsinessDetector:
    def __init__(self):
        # 얼굴 랜드마크 모델 준비
        _ensure_model()
        # FaceLandmarker 옵션 설정 (한 번에 얼굴 1개만 추적)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(
        self,
        rgb_frame: np.ndarray,
        counter: int,
        ear_threshold: float = EAR_THRESHOLD,
        head_pose_enabled: bool = False,
        head_yaw_threshold: float = HEAD_YAW_THRESHOLD_DEFAULT,
        head_pitch_threshold: float = HEAD_PITCH_THRESHOLD_DEFAULT,
    ) -> dict:
        # 입력 프레임 크기 확인
        h, w = rgb_frame.shape[:2]
        # numpy 이미지를 MediaPipe 형식으로 변환
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(mp_img)

        # 얼굴이 없으면 카운터를 0으로 초기화해서 반환
        if not result.face_landmarks:
            return {
                "face_found": False,
                "avg_ear": None,
                "r_pts": None,
                "l_pts": None,
                "counter": 0,
                "drowsy": False,
                "head_pose_enabled": head_pose_enabled,
                "head_yaw_metric": None,
                "head_pitch_metric": None,
                "head_yaw_threshold": head_yaw_threshold,
                "head_pitch_threshold": head_pitch_threshold,
                "head_pose_alert": False,
            }

        # 첫 번째 얼굴의 랜드마크 사용
        lm = result.face_landmarks[0]
        # 오른쪽/왼쪽 눈 6개 점 좌표 추출
        r_pts = _get_pts(lm, RIGHT_EYE, w, h)
        l_pts = _get_pts(lm, LEFT_EYE, w, h)
        # 양쪽 눈 EAR 평균값
        # (한쪽 눈만 순간적으로 잘못 잡혀도 영향이 줄어들도록 평균 사용)
        avg_ear = (_eye_aspect_ratio(r_pts) + _eye_aspect_ratio(l_pts)) / 2.0

        # EAR가 기준값(ear_threshold)보다 작으면 눈 감김 프레임 누적, 아니면 0으로 리셋
        if avg_ear < ear_threshold:
            counter += 1
        else:
            counter = 0

        head_yaw_m, head_pitch_m = _head_pose_metrics(lm)
        head_pose_alert = False
        if head_pose_enabled:
            head_pose_alert = (head_yaw_m > head_yaw_threshold) or (head_pitch_m > head_pitch_threshold)

        # 이후 app.py에서 그리기/문구 표시할 수 있도록 정보 패키징
        return {
            "face_found": True,
            "avg_ear": avg_ear,
            "r_pts": r_pts,
            "l_pts": l_pts,
            "counter": counter,
            "threshold": ear_threshold,
            "drowsy": counter >= CONSEC_FRAMES,
            "head_pose_enabled": head_pose_enabled,
            "head_yaw_metric": head_yaw_m,
            "head_pitch_metric": head_pitch_m,
            "head_yaw_threshold": head_yaw_threshold,
            "head_pitch_threshold": head_pitch_threshold,
            "head_pose_alert": head_pose_alert,
        }
