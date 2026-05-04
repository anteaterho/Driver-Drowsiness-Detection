// 아두이노 Uno 등: USB 시리얼 9600bps, PC에서 '1' = LED ON, '0' = LED OFF
// 업로드 후 PC 앱에서 COM 포트(예: COM9)로 동일 보드레이트 연결

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
