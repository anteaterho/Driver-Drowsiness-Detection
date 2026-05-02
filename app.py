import threading  # 웹캠 캡처를 백그라운드 스레드에서 돌리기 위해 사용

import cv2  # 웹캠 영상 읽기, 색상 변환, 화면에 선/글자 그리기
import gradio as gr  # 웹에서 실행되는 간단한 UI(버튼, 이미지 출력) 구성
import numpy as np  # 점 좌표를 배열로 바꿔 눈 윤곽선 그릴 때 사용

from detector import DrowsinessDetector, CONSEC_FRAMES, EAR_THRESHOLD  # 감지 로직 클래스와 기준값들

# 졸음 감지기 객체를 한 번 만들어서 계속 재사용
detector = DrowsinessDetector()

# RGB 색상
_GREEN  = (0,   255, 0)
_ORANGE = (255, 165, 0)
_RED    = (255,   0, 0)
_YELLOW = (255, 255, 0)

# 백그라운드 캡처 스레드와 공유하는 상태값들
_lock    = threading.Lock()
_running = False
_counter = 0
_ear_threshold = float(EAR_THRESHOLD)
_latest_frame:  np.ndarray | None = None
_latest_status: str = "웹캠을 시작해주세요"


# ── 렌더링 ────────────────────────────────────────────────────────────────────

def _draw_eye(frame: np.ndarray, pts: list, color: tuple) -> None:
    # 눈 테두리를 선으로 그려서 상태를 눈에 띄게 표시
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True,
                  color=color, thickness=1)


def _render(rgb: np.ndarray, info: dict) -> tuple[np.ndarray, str]:
    # 원본 프레임을 복사해서 그 위에 안내 요소를 그린다.
    out = rgb.copy()
    h, w = out.shape[:2]

    # 얼굴이 없으면 경고 문구만 보여준다.
    if not info["face_found"]:
        return out, "😶 얼굴이 감지되지 않았습니다"

    # EAR 값은 "눈이 얼마나 떠 있는지"를 나타내는 비율 값이다.
    # 값이 내려갈수록 눈이 감긴 상태에 가깝다.
    ear     = info["avg_ear"]
    threshold = info.get("threshold", EAR_THRESHOLD)
    counter = info["counter"]

    # 졸음이 확정되면 빨간 오버레이 + 큰 경고 문구를 띄운다.
    if info["drowsy"]:
        eye_color = _RED
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), _RED, -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        cv2.putText(out, "! DROWSY !", (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, _RED, 3)
        status = f"🚨 졸음 감지!  EAR: {ear:.3f} (기준: {threshold:.3f})"
    # 아직 졸음 확정은 아니지만, 눈을 감은 프레임이 누적되는 중
    elif counter > 0:
        eye_color = _ORANGE
        status = f"😴 눈 감음 감지 중 ({counter}/{CONSEC_FRAMES})  EAR: {ear:.3f} (기준: {threshold:.3f})"
    # 정상 상태
    else:
        eye_color = _GREEN
        status = f"✅ 정상  EAR: {ear:.3f} (기준: {threshold:.3f})"

    # 눈 윤곽선과 수치(EAR, 카운트) 표시
    _draw_eye(out, info["r_pts"], eye_color)
    _draw_eye(out, info["l_pts"], eye_color)
    cv2.putText(out, f"EAR: {ear:.3f}",          (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2)
    cv2.putText(out, f"THR: {threshold:.3f}",    (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2)
    cv2.putText(out, f"Count: {counter}/{CONSEC_FRAMES}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2)

    return out, status


# ── 캡처 스레드 ───────────────────────────────────────────────────────────────

def _capture_loop() -> None:
    # 스레드 안에서 전역 상태를 갱신하므로 global 선언
    global _running, _counter, _ear_threshold, _latest_frame, _latest_status

    # 기본 웹캠(0번 장치) 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with _lock:
            _latest_status = "⚠️ 웹캠을 열 수 없습니다"
        _running = False
        return

    # _running이 True인 동안 계속 프레임 처리
    while _running:
        ret, bgr = cap.read()
        if not ret:
            continue

        # OpenCV 기본 BGR -> RGB 변환 (Gradio/MediaPipe와 맞추기)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with _lock:
            # detect()에 넘길 현재 카운터를 안전하게 읽는다.
            cnt = _counter
            thr = _ear_threshold

        # 얼굴/눈 상태 분석 -> 화면 렌더링
        info = detector.detect(rgb, cnt, ear_threshold=thr)
        rendered, status = _render(rgb, info)

        with _lock:
            # 최신 결과를 UI가 읽을 수 있게 저장
            _counter       = info["counter"]
            _latest_frame  = rendered
            _latest_status = status

    # 루프가 끝나면 웹캠 장치 해제
    cap.release()


# ── Gradio 콜백 ───────────────────────────────────────────────────────────────

def start_webcam() -> str:
    global _running, _counter
    # 이미 실행 중이면 중복 시작 방지
    if _running:
        return "이미 실행 중입니다"
    # 새로 시작할 때는 누적 카운터 초기화
    _counter = 0
    _running = True
    # 백그라운드 스레드로 캡처 루프 실행
    threading.Thread(target=_capture_loop, daemon=True).start()
    return "▶ 웹캠 시작됨"


def stop_webcam() -> str:
    global _running
    # 다음 루프부터 while이 종료되어 웹캠이 멈춘다.
    _running = False
    return "⏹ 웹캠 중지됨"


def update_ear_threshold(value: float) -> str:
    global _ear_threshold
    # 슬라이더 값(실수)을 공유 변수에 저장해 다음 프레임부터 반영한다.
    with _lock:
        _ear_threshold = float(value)
    return f"EAR 기준값이 {float(value):.3f}으로 변경되었습니다"


def get_latest() -> tuple:
    # UI 타이머가 주기적으로 호출해서 최신 프레임/문구를 받아간다.
    with _lock:
        return _latest_frame, _latest_status


# ── UI ───────────────────────────────────────────────────────────────────────

_ALARM_HEAD_SCRIPT = """
<script>
  (() => {
    const setup = () => {
      // 중복 초기화 방지
      if (window.__drowsyAlarmSetupDone) return true;

      const initBtn = document.getElementById("alarm_init_btn");
      const enabledEl = document.getElementById("alarm_enabled");
      const modeEl = document.getElementById("alarm_mode");
      const freqEl = document.getElementById("alarm_freq");
      const volEl = document.getElementById("alarm_volume");
      const beepRateEl = document.getElementById("alarm_beep_rate");
      const beepDutyEl = document.getElementById("alarm_beep_duty");
      const freqText = document.getElementById("alarm_freq_text");
      const volText = document.getElementById("alarm_volume_text");
      const beepRateText = document.getElementById("alarm_beep_rate_text");
      const beepDutyText = document.getElementById("alarm_beep_duty_text");

      // UI가 아직 렌더링되지 않았으면 다음 tick에 재시도
      if (!initBtn || !enabledEl || !modeEl || !freqEl || !volEl || !beepRateEl || !beepDutyEl) {
        return false;
      }
      window.__drowsyAlarmSetupDone = true;

      let ctx = null;
      let osc = null;
      let gainNode = null;
      let started = false;
      let beepTimerId = null;

      const updateTexts = () => {
        if (freqText) freqText.textContent = String(freqEl.value);
        if (volText) volText.textContent = Number(volEl.value).toFixed(2);
        if (beepRateText) beepRateText.textContent = String(beepRateEl.value);
        if (beepDutyText) beepDutyText.textContent = String(beepDutyEl.value);
      };
      updateTexts();
      freqEl.addEventListener("input", updateTexts);
      volEl.addEventListener("input", updateTexts);
      beepRateEl.addEventListener("input", updateTexts);
      beepDutyEl.addEventListener("input", updateTexts);

      const ensureAudioGraph = async () => {
        if (!ctx) {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          ctx = new AudioCtx();
        }
        if (ctx.state === "suspended") await ctx.resume();
        if (!osc) {
          osc = ctx.createOscillator();
          gainNode = ctx.createGain();
          osc.type = "sine";
          osc.frequency.value = Number(freqEl.value);
          gainNode.gain.value = 0.0;
          osc.connect(gainNode).connect(ctx.destination);
          osc.start();
        }
      };

      const startTone = async () => {
        await ensureAudioGraph();
        const now = ctx.currentTime;
        gainNode.gain.cancelScheduledValues(now);
        gainNode.gain.linearRampToValueAtTime(Number(volEl.value), now + 0.03);
        osc.frequency.setValueAtTime(Number(freqEl.value), now);
      };

      const stopTone = async () => {
        if (!ctx || !gainNode) return;
        const now = ctx.currentTime;
        gainNode.gain.cancelScheduledValues(now);
        gainNode.gain.linearRampToValueAtTime(0.0, now + 0.05);
      };

      const stopBeepTimer = () => {
        if (beepTimerId !== null) {
          window.clearInterval(beepTimerId);
          beepTimerId = null;
        }
      };

      const startBeepMode = async () => {
        await ensureAudioGraph();
        stopBeepTimer();

        const rate = Number(beepRateEl.value);
        const duty = Number(beepDutyEl.value) / 100.0;
        const periodMs = Math.max(80, Math.round(1000 / Math.max(1, rate)));
        const onMs = Math.max(20, Math.round(periodMs * duty));

        const now = ctx.currentTime;
        osc.frequency.setValueAtTime(Number(freqEl.value), now);
        gainNode.gain.cancelScheduledValues(now);
        gainNode.gain.setValueAtTime(Number(volEl.value), now);
        window.setTimeout(() => {
          if (!ctx || !gainNode) return;
          gainNode.gain.setValueAtTime(0.0, ctx.currentTime);
        }, onMs);

        beepTimerId = window.setInterval(() => {
          if (!ctx || !gainNode || !osc) return;
          const t = ctx.currentTime;
          osc.frequency.setValueAtTime(Number(freqEl.value), t);
          gainNode.gain.cancelScheduledValues(t);
          gainNode.gain.setValueAtTime(Number(volEl.value), t);
          window.setTimeout(() => {
            if (!ctx || !gainNode) return;
            gainNode.gain.setValueAtTime(0.0, ctx.currentTime);
          }, onMs);
        }, periodMs);
      };

      initBtn.addEventListener("click", async () => {
        await ensureAudioGraph();
        initBtn.textContent = "✅ 오디오 준비됨";
      });

      [enabledEl, modeEl, freqEl, volEl, beepRateEl, beepDutyEl].forEach((el) => {
        el.addEventListener("input", () => { ensureAudioGraph().catch(() => {}); });
        el.addEventListener("change", () => { ensureAudioGraph().catch(() => {}); });
      });

      freqEl.addEventListener("input", () => {
        if (osc && ctx) osc.frequency.setValueAtTime(Number(freqEl.value), ctx.currentTime);
      });
      volEl.addEventListener("input", () => {
        if (gainNode && ctx && started) gainNode.gain.setValueAtTime(Number(volEl.value), ctx.currentTime);
      });

      const resetBeepIfNeeded = async () => {
        if (!started || modeEl.value !== "beep") return;
        await startBeepMode();
      };
      beepRateEl.addEventListener("input", resetBeepIfNeeded);
      beepDutyEl.addEventListener("input", resetBeepIfNeeded);
      modeEl.addEventListener("change", async () => {
        if (!started) return;
        if (modeEl.value === "beep") {
          await startBeepMode();
        } else {
          stopBeepTimer();
          await startTone();
        }
      });

      window.setInterval(async () => {
        const statusRoot = document.getElementById("status_box");
        const textInput = statusRoot
          ? statusRoot.querySelector("input, textarea, [contenteditable='true']")
          : null;
        const statusText = textInput
          ? (textInput.value ?? textInput.textContent ?? "")
          : (statusRoot?.textContent ?? "");
        const isDrowsy = statusText.includes("졸음 감지");
        const enabled = enabledEl.checked;

        if (isDrowsy && enabled) {
          if (!started) {
            started = true;
            if (modeEl.value === "beep") await startBeepMode();
            else await startTone();
          } else if (osc && ctx) {
            osc.frequency.setValueAtTime(Number(freqEl.value), ctx.currentTime);
          }
        } else if (started) {
          started = false;
          stopBeepTimer();
          await stopTone();
        }
      }, 120);

      return true;
    };

    // Gradio 렌더 타이밍 이슈 대응: 준비될 때까지 짧게 재시도
    let retry = 0;
    const timer = window.setInterval(() => {
      retry += 1;
      if (setup() || retry > 80) window.clearInterval(timer);
    }, 100);
  })();
</script>
"""

with gr.Blocks(title="운전자 졸음 감지", head=_ALARM_HEAD_SCRIPT) as demo:
    gr.Markdown(
        f"# ✏️공부 졸음감지 시스템\n"
        f"눈 개폐도(EAR)를 실시간으로 측정합니다. "
        f"눈이 **{CONSEC_FRAMES}프레임 이상 연속**으로 감기면 졸음으로 판정합니다.\n"
        f"아래 슬라이더로 EAR 기준값을 조절할 수 있습니다."
    )

    with gr.Row():
        btn_start = gr.Button("▶ 웹캠 시작", variant="primary")
        btn_stop  = gr.Button("⏹ 웹캠 중지", variant="stop")

    output_img = gr.Image(label="실시간 감지 결과")
    status_box = gr.Textbox(label="상태", value="웹캠을 시작해주세요", interactive=False, elem_id="status_box")
    ear_slider = gr.Slider(
        minimum=0.10,
        maximum=0.40,
        value=float(EAR_THRESHOLD),
        step=0.005,
        label="EAR 졸음 판정 기준값 (낮을수록 덜 민감, 높을수록 더 민감)",
    )
    gr.HTML(
        """
        <div style="border:1px solid #ddd; border-radius:12px; padding:12px; margin-top:8px;">
          <h4 style="margin:0 0 8px 0;">경고음 설정 (사인 파형)</h4>
          <p style="margin:0 0 10px 0; font-size:13px;">
            브라우저 정책 때문에 처음 1회는 <b>오디오 초기화</b> 버튼을 눌러야 소리가 납니다.
          </p>
          <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
            <button id="alarm_init_btn" type="button">🔊 오디오 초기화</button>
            <label><input id="alarm_enabled" type="checkbox" checked /> 경고음 켜기</label>
            <label>알림 모드:
              <select id="alarm_mode">
                <option value="continuous" selected>지속음</option>
                <option value="beep">연속 비프음</option>
              </select>
            </label>
            <label>주파수(Hz):
              <input id="alarm_freq" type="range" min="220" max="1200" step="10" value="880" />
              <span id="alarm_freq_text">880</span>
            </label>
            <label>볼륨:
              <input id="alarm_volume" type="range" min="0" max="0.6" step="0.01" value="0.2" />
              <span id="alarm_volume_text">0.20</span>
            </label>
            <label>비프 속도(Hz):
              <input id="alarm_beep_rate" type="range" min="1" max="8" step="1" value="3" />
              <span id="alarm_beep_rate_text">3</span>
            </label>
            <label>비프 길이(%):
              <input id="alarm_beep_duty" type="range" min="10" max="90" step="5" value="40" />
              <span id="alarm_beep_duty_text">40</span>
            </label>
          </div>
        </div>
        <script>
          (() => {
            // 중복 초기화 방지
            if (window.__drowsyAlarmSetupDone) return;
            window.__drowsyAlarmSetupDone = true;

            let ctx = null;
            let osc = null;
            let gainNode = null;
            let started = false;
            let beepTimerId = null;

            const initBtn = document.getElementById("alarm_init_btn");
            const enabledEl = document.getElementById("alarm_enabled");
            const modeEl = document.getElementById("alarm_mode");
            const freqEl = document.getElementById("alarm_freq");
            const volEl = document.getElementById("alarm_volume");
            const beepRateEl = document.getElementById("alarm_beep_rate");
            const beepDutyEl = document.getElementById("alarm_beep_duty");
            const freqText = document.getElementById("alarm_freq_text");
            const volText = document.getElementById("alarm_volume_text");
            const beepRateText = document.getElementById("alarm_beep_rate_text");
            const beepDutyText = document.getElementById("alarm_beep_duty_text");

            const updateTexts = () => {
              freqText.textContent = String(freqEl.value);
              volText.textContent = Number(volEl.value).toFixed(2);
              beepRateText.textContent = String(beepRateEl.value);
              beepDutyText.textContent = String(beepDutyEl.value);
            };
            updateTexts();
            freqEl.addEventListener("input", updateTexts);
            volEl.addEventListener("input", updateTexts);
            beepRateEl.addEventListener("input", updateTexts);
            beepDutyEl.addEventListener("input", updateTexts);

            const ensureAudioGraph = async () => {
              if (!ctx) {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                ctx = new AudioCtx();
              }
              if (ctx.state === "suspended") {
                await ctx.resume();
              }
              if (!osc) {
                osc = ctx.createOscillator();
                gainNode = ctx.createGain();
                osc.type = "sine"; // 사인 파형 고정
                osc.frequency.value = Number(freqEl.value);
                gainNode.gain.value = 0.0; // 시작은 무음
                osc.connect(gainNode).connect(ctx.destination);
                osc.start();
              }
            };

            const startTone = async () => {
              await ensureAudioGraph();
              const now = ctx.currentTime;
              gainNode.gain.cancelScheduledValues(now);
              gainNode.gain.linearRampToValueAtTime(Number(volEl.value), now + 0.03);
              osc.frequency.setValueAtTime(Number(freqEl.value), now);
            };

            const stopTone = async () => {
              if (!ctx || !gainNode) return;
              const now = ctx.currentTime;
              gainNode.gain.cancelScheduledValues(now);
              gainNode.gain.linearRampToValueAtTime(0.0, now + 0.05);
            };

            const stopBeepTimer = () => {
              if (beepTimerId !== null) {
                window.clearInterval(beepTimerId);
                beepTimerId = null;
              }
            };

            const startBeepMode = async () => {
              await ensureAudioGraph();
              stopBeepTimer();

              // 비프 한 주기: period, 그 중 duty 비율만 소리가 난다.
              const applyBeepSchedule = () => {
                const rate = Number(beepRateEl.value); // 초당 비프 횟수
                const duty = Number(beepDutyEl.value) / 100.0; // 소리 나는 비율
                const periodMs = Math.max(80, Math.round(1000 / Math.max(1, rate)));
                const onMs = Math.max(20, Math.round(periodMs * duty));
                const offMs = Math.max(20, periodMs - onMs);

                // 즉시 1회 재생 시작
                const now = ctx.currentTime;
                osc.frequency.setValueAtTime(Number(freqEl.value), now);
                gainNode.gain.cancelScheduledValues(now);
                gainNode.gain.setValueAtTime(Number(volEl.value), now);

                beepTimerId = window.setInterval(() => {
                  if (!ctx || !gainNode || !osc) return;
                  const t = ctx.currentTime;
                  osc.frequency.setValueAtTime(Number(freqEl.value), t);
                  gainNode.gain.cancelScheduledValues(t);
                  gainNode.gain.setValueAtTime(Number(volEl.value), t);
                  window.setTimeout(() => {
                    if (!ctx || !gainNode) return;
                    gainNode.gain.setValueAtTime(0.0, ctx.currentTime);
                  }, onMs);
                }, periodMs);

                // 첫 펄스의 OFF도 적용
                window.setTimeout(() => {
                  if (!ctx || !gainNode) return;
                  gainNode.gain.setValueAtTime(0.0, ctx.currentTime);
                }, onMs);
              };

              applyBeepSchedule();
            };

            initBtn.addEventListener("click", async () => {
              await ensureAudioGraph();
              initBtn.textContent = "✅ 오디오 준비됨";
            });

            // 사용자가 다른 컨트롤을 만져도 오디오 컨텍스트를 깨워본다(브라우저 정책 대응).
            [enabledEl, modeEl, freqEl, volEl, beepRateEl, beepDutyEl].forEach((el) => {
              el.addEventListener("input", () => { ensureAudioGraph().catch(() => {}); });
              el.addEventListener("change", () => { ensureAudioGraph().catch(() => {}); });
            });

            freqEl.addEventListener("input", () => {
              if (osc && ctx) {
                osc.frequency.setValueAtTime(Number(freqEl.value), ctx.currentTime);
              }
            });

            volEl.addEventListener("input", () => {
              if (gainNode && ctx && started) {
                gainNode.gain.setValueAtTime(Number(volEl.value), ctx.currentTime);
              }
            });

            // 비프 모드에서 속도/길이를 바꾸면 타이머를 다시 구성해 즉시 반영
            const resetBeepIfNeeded = async () => {
              if (!started || modeEl.value !== "beep") return;
              await startBeepMode();
            };
            beepRateEl.addEventListener("input", resetBeepIfNeeded);
            beepDutyEl.addEventListener("input", resetBeepIfNeeded);
            modeEl.addEventListener("change", async () => {
              if (!started) return;
              if (modeEl.value === "beep") {
                await startBeepMode();
              } else {
                stopBeepTimer();
                await startTone();
              }
            });

            // 상태 박스 텍스트를 주기적으로 읽어 졸음 여부 판단
            window.setInterval(async () => {
              const statusRoot = document.getElementById("status_box");
              // Gradio 버전에 따라 textbox가 input 또는 textarea로 렌더링될 수 있다.
              const textInput = statusRoot
                ? statusRoot.querySelector("input, textarea, [contenteditable='true']")
                : null;
              const statusText = textInput
                ? (textInput.value ?? textInput.textContent ?? "")
                : (statusRoot?.textContent ?? "");
              const isDrowsy = statusText.includes("졸음 감지");
              const enabled = enabledEl.checked;

              if (isDrowsy && enabled) {
                if (!started) {
                  started = true;
                  if (modeEl.value === "beep") {
                    await startBeepMode();
                  } else {
                    await startTone();
                  }
                } else if (osc && ctx) {
                  osc.frequency.setValueAtTime(Number(freqEl.value), ctx.currentTime);
                }
              } else if (started) {
                started = false;
                stopBeepTimer();
                await stopTone();
              }
            }, 120);
          })();
        </script>
        """
    )

    # 100ms마다 최신 결과를 화면에 반영 (~10fps)
    gr.Timer(value=0.1).tick(fn=get_latest, outputs=[output_img, status_box])

    btn_start.click(fn=start_webcam, outputs=[status_box])
    btn_stop.click(fn=stop_webcam,  outputs=[status_box])
    ear_slider.change(fn=update_ear_threshold, inputs=[ear_slider], outputs=[status_box])

if __name__ == "__main__":
    demo.launch()
