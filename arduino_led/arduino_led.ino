// 아두이노 Uno 등: USB 시리얼 9600bps
// PC에서 '1' + 줄바꿈 = 내장 LED(핀 13) ON, '0' + 줄바꿈 = OFF
// 시리얼 모니터는 닫은 뒤 PC에서 app.py 실행 (COM 포트 중복 불가)

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
