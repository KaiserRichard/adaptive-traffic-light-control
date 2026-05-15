#include <Arduino.h>

/*
  Arduino Uno 2-Digit 7-Segment Smoke Test

  Purpose:
  - Verify wiring of two 7-segment digits before integrating with UART/FSM.
  - Count from 00 to 99 repeatedly.
  - Use direct GPIO, no 74HC595, no TM1637, no MAX7219.

  Segment mapping:
    A -> D8
    B -> D9
    C -> D10
    D -> D11
    E -> D12
    F -> D13
    G -> A0

  Digit common:
    Tens  -> A1
    Units -> A2
*/

// Change this depending on your 7-segment type.
const bool COMMON_CATHODE = true;

// Segment order: A, B, C, D, E, F, G
const int SEGMENT_PINS[7] = {
  8,
  9,
  10,
  11,
  12,
  13,
  A0
};

const int DIGIT_TENS_PIN = A1;
const int DIGIT_UNITS_PIN = A2;

const byte DIGIT_PATTERNS[10][7] = {
  {1, 1, 1, 1, 1, 1, 0}, // 0
  {0, 1, 1, 0, 0, 0, 0}, // 1
  {1, 1, 0, 1, 1, 0, 1}, // 2
  {1, 1, 1, 1, 0, 0, 1}, // 3
  {0, 1, 1, 0, 0, 1, 1}, // 4
  {1, 0, 1, 1, 0, 1, 1}, // 5
  {1, 0, 1, 1, 1, 1, 1}, // 6
  {1, 1, 1, 0, 0, 0, 0}, // 7
  {1, 1, 1, 1, 1, 1, 1}, // 8
  {1, 1, 1, 1, 0, 1, 1}  // 9
};

int counterValue = 0;

unsigned long lastCountMs = 0;
unsigned long lastMuxMs = 0;

bool showTens = true;

int segmentOnLevel() {
  return COMMON_CATHODE ? HIGH : LOW;
}

int segmentOffLevel() {
  return COMMON_CATHODE ? LOW : HIGH;
}

int digitOnLevel() {
  return COMMON_CATHODE ? LOW : HIGH;
}

int digitOffLevel() {
  return COMMON_CATHODE ? HIGH : LOW;
}

void disableDigits() {
  digitalWrite(DIGIT_TENS_PIN, digitOffLevel());
  digitalWrite(DIGIT_UNITS_PIN, digitOffLevel());
}

void writeDigit(int digit) {
  if (digit < 0 || digit > 9) {
    digit = 0;
  }

  for (int i = 0; i < 7; i++) {
    digitalWrite(
      SEGMENT_PINS[i],
      DIGIT_PATTERNS[digit][i] ? segmentOnLevel() : segmentOffLevel()
    );
  }
}

void multiplexDisplay() {
  unsigned long now = millis();

  if (now - lastMuxMs < 3) {
    return;
  }

  lastMuxMs = now;

  int tens = counterValue / 10;
  int units = counterValue % 10;

  disableDigits();

  if (showTens) {
    writeDigit(tens);
    digitalWrite(DIGIT_TENS_PIN, digitOnLevel());
  } else {
    writeDigit(units);
    digitalWrite(DIGIT_UNITS_PIN, digitOnLevel());
  }

  showTens = !showTens;
}

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 7; i++) {
    pinMode(SEGMENT_PINS[i], OUTPUT);
    digitalWrite(SEGMENT_PINS[i], segmentOffLevel());
  }

  pinMode(DIGIT_TENS_PIN, OUTPUT);
  pinMode(DIGIT_UNITS_PIN, OUTPUT);

  disableDigits();

  Serial.println("ARDUINO_7SEG_SMOKE_TEST_READY");
}

void loop() {
  multiplexDisplay();

  unsigned long now = millis();

  if (now - lastCountMs >= 1000) {
    lastCountMs = now;

    Serial.print("DISPLAY=");
    Serial.println(counterValue);

    counterValue++;

    if (counterValue > 99) {
      counterValue = 0;
    }
  }
}