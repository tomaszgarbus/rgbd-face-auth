int currentLED = 0;
int incomingByte = 0;

void setLEDs() {
  for (int i = 2; i <= 7; ++i) {
    digitalWrite(i, currentLED != 0 && (currentLED == i - 1 || currentLED == i - 4) ? HIGH : LOW);
  }
}

void setup() {
  Serial.begin(9600);
  for (int i = 2; i <= 7; ++i) {
    pinMode(i, OUTPUT);
  }
  setLEDs();
}

void loop() {
  if (Serial.available() > 0) {
    incomingByte = Serial.read();
    if (incomingByte == 'C') {
      Serial.write('K');
    } else if (incomingByte >= '0' && incomingByte <= '3') {
      currentLED = incomingByte - '0';
      setLEDs();
    }
  }
}
