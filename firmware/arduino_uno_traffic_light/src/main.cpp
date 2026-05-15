#include <Arduino.h>

#ifndef SERIAL_BAUD
#define SERIAL_BAUD 115200
#endif

/*
  Arduino Uno Traffic Light UART Controller with 2-Digit 7-Segment Countdown

  Purpose:
  - Receive adaptive traffic timing plan from Python through USB serial.
  - Execute two-direction traffic light FSM.
  - Display remaining countdown time on two 7-segment digits.
  - Return ACK when a valid PLAN message is received.

  UART protocol:
    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

  Example:
    PLAN,1,10,5,3,1

  Response:
    ACK,1

  Invalid:
    ERR

  Current MCU testbed:
    Arduino Uno

  Notes:
  - This version does NOT use 74HC595, TM1637, or MAX7219.
  - 7-segment display is driven directly by Arduino GPIO.
  - D0/D1 are not used because they are used by USB serial.
*/

// -----------------------------------------------------------------------------
// 7-segment type
// -----------------------------------------------------------------------------

// Set true for common cathode 7-segment.
// Set false for common anode 7-segment.
const bool COMMON_CATHODE = true;

// -----------------------------------------------------------------------------
// Traffic LED pin mapping
// -----------------------------------------------------------------------------

const int A_RED_PIN = 2;
const int A_YELLOW_PIN = 3;
const int A_GREEN_PIN = 4;

const int B_RED_PIN = 5;
const int B_YELLOW_PIN = 6;
const int B_GREEN_PIN = 7;

// -----------------------------------------------------------------------------
// 7-segment pin mapping
// -----------------------------------------------------------------------------

// Segment order: A, B, C, D, E, F, G
const int SEGMENT_PINS[7] = {
  8,   // A
  9,   // B
  10,  // C
  11,  // D
  12,  // E
  13,  // F
  A0   // G
};

const int DIGIT_TENS_PIN = A1;
const int DIGIT_UNITS_PIN = A2;

// Digit patterns for 0-9.
// Bit order: A B C D E F G
// 1 means segment should be ON logically.
// Actual HIGH/LOW depends on common cathode/anode.
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

// -----------------------------------------------------------------------------
// Timing configuration
// -----------------------------------------------------------------------------

unsigned long greenASeconds = 10;
unsigned long greenBSeconds = 10;
unsigned long yellowSeconds = 3;
unsigned long allRedSeconds = 1;

// If no valid PLAN is received for this duration,
// keep using safe fallback timing.
const unsigned long PLAN_TIMEOUT_MS = 15000;
unsigned long lastPlanReceivedMs = 0;

// Countdown shown on 7-segment.
int remainingSeconds = 10;

// -----------------------------------------------------------------------------
// Traffic state machine
// -----------------------------------------------------------------------------

enum TrafficState {
  STATE_A_GREEN,
  STATE_A_YELLOW,
  STATE_ALL_RED_AFTER_A,
  STATE_B_GREEN,
  STATE_B_YELLOW,
  STATE_ALL_RED_AFTER_B
};

TrafficState currentState = STATE_A_GREEN;

unsigned long lastTickMs = 0;
unsigned long lastMuxMs = 0;
bool showTensDigit = true;

// -----------------------------------------------------------------------------
// Serial parsing
// -----------------------------------------------------------------------------

String inputLine = "";
int currentPlanId = 0;

// -----------------------------------------------------------------------------
// GPIO helper functions
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Traffic LED output
// -----------------------------------------------------------------------------

void setAllTrafficLightsOff() {
  digitalWrite(A_RED_PIN, LOW);
  digitalWrite(A_YELLOW_PIN, LOW);
  digitalWrite(A_GREEN_PIN, LOW);

  digitalWrite(B_RED_PIN, LOW);
  digitalWrite(B_YELLOW_PIN, LOW);
  digitalWrite(B_GREEN_PIN, LOW);
}

void applyTrafficStateOutputs(TrafficState state) {
  setAllTrafficLightsOff();

  switch (state) {
    case STATE_A_GREEN:
      digitalWrite(A_GREEN_PIN, HIGH);
      digitalWrite(B_RED_PIN, HIGH);
      break;

    case STATE_A_YELLOW:
      digitalWrite(A_YELLOW_PIN, HIGH);
      digitalWrite(B_RED_PIN, HIGH);
      break;

    case STATE_ALL_RED_AFTER_A:
      digitalWrite(A_RED_PIN, HIGH);
      digitalWrite(B_RED_PIN, HIGH);
      break;

    case STATE_B_GREEN:
      digitalWrite(A_RED_PIN, HIGH);
      digitalWrite(B_GREEN_PIN, HIGH);
      break;

    case STATE_B_YELLOW:
      digitalWrite(A_RED_PIN, HIGH);
      digitalWrite(B_YELLOW_PIN, HIGH);
      break;

    case STATE_ALL_RED_AFTER_B:
      digitalWrite(A_RED_PIN, HIGH);
      digitalWrite(B_RED_PIN, HIGH);
      break;
  }
}

// -----------------------------------------------------------------------------
// 7-segment display
// -----------------------------------------------------------------------------

void disableBothDigits() {
  digitalWrite(DIGIT_TENS_PIN, digitOffLevel());
  digitalWrite(DIGIT_UNITS_PIN, digitOffLevel());
}

void writeDigitPattern(int digit) {
  if (digit < 0 || digit > 9) {
    digit = 0;
  }

  for (int i = 0; i < 7; i++) {
    int level = DIGIT_PATTERNS[digit][i] ? segmentOnLevel() : segmentOffLevel();
    digitalWrite(SEGMENT_PINS[i], level);
  }
}

void multiplexDisplay() {
  unsigned long now = millis();

  // Refresh every 3 ms.
  // Smaller value = less flicker.
  if (now - lastMuxMs < 3) {
    return;
  }

  lastMuxMs = now;

  int displayValue = remainingSeconds;

  if (displayValue < 0) {
    displayValue = 0;
  }

  if (displayValue > 99) {
    displayValue = 99;
  }

  int tens = displayValue / 10;
  int units = displayValue % 10;

  disableBothDigits();

  if (showTensDigit) {
    writeDigitPattern(tens);

    // Hide leading zero for 1-9.
    // Example: show " 5" instead of "05".
    if (tens > 0) {
      digitalWrite(DIGIT_TENS_PIN, digitOnLevel());
    }
  } else {
    writeDigitPattern(units);
    digitalWrite(DIGIT_UNITS_PIN, digitOnLevel());
  }

  showTensDigit = !showTensDigit;
}

// -----------------------------------------------------------------------------
// FSM helpers
// -----------------------------------------------------------------------------

unsigned long getStateDurationSeconds(TrafficState state) {
  switch (state) {
    case STATE_A_GREEN:
      return greenASeconds;

    case STATE_A_YELLOW:
      return yellowSeconds;

    case STATE_ALL_RED_AFTER_A:
      return allRedSeconds;

    case STATE_B_GREEN:
      return greenBSeconds;

    case STATE_B_YELLOW:
      return yellowSeconds;

    case STATE_ALL_RED_AFTER_B:
      return allRedSeconds;
  }

  return 1;
}

TrafficState getNextState(TrafficState state) {
  switch (state) {
    case STATE_A_GREEN:
      return STATE_A_YELLOW;

    case STATE_A_YELLOW:
      return STATE_ALL_RED_AFTER_A;

    case STATE_ALL_RED_AFTER_A:
      return STATE_B_GREEN;

    case STATE_B_GREEN:
      return STATE_B_YELLOW;

    case STATE_B_YELLOW:
      return STATE_ALL_RED_AFTER_B;

    case STATE_ALL_RED_AFTER_B:
      return STATE_A_GREEN;
  }

  return STATE_A_GREEN;
}

const char* stateToString(TrafficState state) {
  switch (state) {
    case STATE_A_GREEN:
      return "A_GREEN";

    case STATE_A_YELLOW:
      return "A_YELLOW";

    case STATE_ALL_RED_AFTER_A:
      return "ALL_RED_AFTER_A";

    case STATE_B_GREEN:
      return "B_GREEN";

    case STATE_B_YELLOW:
      return "B_YELLOW";

    case STATE_ALL_RED_AFTER_B:
      return "ALL_RED_AFTER_B";
  }

  return "UNKNOWN";
}

void transitionTo(TrafficState nextState) {
  currentState = nextState;
  remainingSeconds = (int)getStateDurationSeconds(currentState);

  applyTrafficStateOutputs(currentState);

  Serial.print("[MCU/STATE] ");
  Serial.print(stateToString(currentState));
  Serial.print(" remaining=");
  Serial.println(remainingSeconds);
}

void updateCountdownAndFSM() {
  unsigned long now = millis();

  if (now - lastTickMs < 1000) {
    return;
  }

  lastTickMs += 1000;

  if (remainingSeconds > 1) {
    remainingSeconds--;
    return;
  }

  TrafficState nextState = getNextState(currentState);
  transitionTo(nextState);
}

void applyFallbackTimingIfNeeded() {
  unsigned long now = millis();

  if (lastPlanReceivedMs == 0) {
    return;
  }

  if (now - lastPlanReceivedMs > PLAN_TIMEOUT_MS) {
    greenASeconds = 10;
    greenBSeconds = 10;
    yellowSeconds = 3;
    allRedSeconds = 1;

    lastPlanReceivedMs = now;

    Serial.println("[MCU/WARN] PLAN_TIMEOUT,FALLBACK_TIMING");
  }
}

// -----------------------------------------------------------------------------
// PLAN parser
// -----------------------------------------------------------------------------

bool parsePlanMessage(const String& line) {
  // Expected:
  // PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

  String parts[6];
  int partIndex = 0;
  int startIndex = 0;

  for (int i = 0; i <= line.length(); i++) {
    if (i == line.length() || line.charAt(i) == ',') {
      if (partIndex >= 6) {
        return false;
      }

      parts[partIndex] = line.substring(startIndex, i);
      parts[partIndex].trim();

      partIndex++;
      startIndex = i + 1;
    }
  }

  if (partIndex != 6) {
    return false;
  }

  if (parts[0] != "PLAN") {
    return false;
  }

  int planId = parts[1].toInt();
  int newGreenA = parts[2].toInt();
  int newGreenB = parts[3].toInt();
  int newYellow = parts[4].toInt();
  int newAllRed = parts[5].toInt();

  if (planId <= 0) {
    return false;
  }

  if (newGreenA <= 0 || newGreenB <= 0 || newYellow <= 0 || newAllRed <= 0) {
    return false;
  }

  if (newGreenA > 120 || newGreenB > 120 || newYellow > 20 || newAllRed > 20) {
    return false;
  }

  currentPlanId = planId;

  greenASeconds = (unsigned long)newGreenA;
  greenBSeconds = (unsigned long)newGreenB;
  yellowSeconds = (unsigned long)newYellow;
  allRedSeconds = (unsigned long)newAllRed;

  lastPlanReceivedMs = millis();

  // Apply new plan immediately from A_GREEN for easy testing.
  transitionTo(STATE_A_GREEN);

  Serial.print("[MCU/ACK] ACK,");
  Serial.println(currentPlanId);

  Serial.print("[MCU/APPLIED] plan_id=");
  Serial.print(currentPlanId);
  Serial.print(" greenA=");
  Serial.print(greenASeconds);
  Serial.print(" greenB=");
  Serial.print(greenBSeconds);
  Serial.print(" yellow=");
  Serial.print(yellowSeconds);
  Serial.print(" allRed=");
  Serial.println(allRedSeconds);

  return true;
}

void handleSerialLine(String line) {
  line.trim();

  if (line.length() == 0) {
    return;
  }

  Serial.print("[USER/RX] ");
  Serial.println(line);

  bool ok = parsePlanMessage(line);

  if (!ok) {
    Serial.print("[MCU/ERR] invalid message: ");
    Serial.println(line);
    Serial.println("ERR");
  }
}

void readSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      handleSerialLine(inputLine);
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
    }
  }
}

// -----------------------------------------------------------------------------
// Setup and loop
// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(SERIAL_BAUD);

  pinMode(A_RED_PIN, OUTPUT);
  pinMode(A_YELLOW_PIN, OUTPUT);
  pinMode(A_GREEN_PIN, OUTPUT);

  pinMode(B_RED_PIN, OUTPUT);
  pinMode(B_YELLOW_PIN, OUTPUT);
  pinMode(B_GREEN_PIN, OUTPUT);

  for (int i = 0; i < 7; i++) {
    pinMode(SEGMENT_PINS[i], OUTPUT);
    digitalWrite(SEGMENT_PINS[i], segmentOffLevel());
  }

  pinMode(DIGIT_TENS_PIN, OUTPUT);
  pinMode(DIGIT_UNITS_PIN, OUTPUT);
  disableBothDigits();

  lastTickMs = millis();

  transitionTo(STATE_A_GREEN);

  Serial.println("[MCU/READY] ARDUINO_TRAFFIC_LIGHT_READY");
  Serial.println("[MCU/DEFAULT] DEFAULT_PLAN,10,10,3,1");
}

void loop() {
  readSerialInput();
  applyFallbackTimingIfNeeded();

  updateCountdownAndFSM();
  multiplexDisplay();
}