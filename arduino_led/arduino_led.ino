// 아두이노 Uno 등: USB 시리얼 9600bps
// PC에서 '1' + 줄바꿈 = 내장 LED(핀 13) ON, '0' + 줄바꿈 = OFF
// 업로드 후 PC 앱(app.py)에서 COM 포트(예: COM9)로 연결 — 시리얼 모니터는 닫을 것

const int LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    int c = Serial.read();
    if (c == '1') {
      digitalWrite(LED_PIN, HIGH);
    } else if (c == '0') {
      digitalWrite(LED_PIN, LOW);
    }
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}
