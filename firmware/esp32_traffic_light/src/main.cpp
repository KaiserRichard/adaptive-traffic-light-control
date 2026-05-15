#include <Arduino.h>
#ifndef SERIAL_BAUD
#define SERIAL_BAUD 115200
#endif
/*
  ESP32 Traffic Light UART Controller

  Purpose:
  - Receive traffic signal timing plan from Python/Raspberry Pi through UART.
  - Reply ACK when a valid PLAN message is received.
  - Control two-direction traffic light LEDs using a finite-state machine.

  UART protocol:
    PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>

  Example:
    PLAN,1,10,5,3,1

  Response:
    ACK,1

  Invalid message:
    ERR

  LED layout:
    Direction A: red, yellow, green
    Direction B: red, yellow, green
*/

// -----------------------------------------------------------------------------
// Pin mapping
// -----------------------------------------------------------------------------

const int A_RED_PIN = 25;
const int A_YELLOW_PIN = 26;
const int A_GREEN_PIN = 27;

const int B_RED_PIN = 14;
const int B_YELLOW_PIN = 12;
const int B_GREEN_PIN = 13;

// -----------------------------------------------------------------------------
// Timing configuration
// -----------------------------------------------------------------------------

unsigned long greenASeconds = 10;
unsigned long greenBSeconds = 10;
unsigned long yellowSeconds = 3;
unsigned long allRedSeconds = 1;

// If no valid PLAN is received for this duration, keep using safe fallback timing.
const unsigned long PLAN_TIMEOUT_MS = 15000;

unsigned long lastPlanReceivedMs = 0;

// -----------------------------------------------------------------------------
// State machine
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
unsigned long stateStartMs = 0;

// -----------------------------------------------------------------------------
// Serial parsing
// -----------------------------------------------------------------------------

String inputLine = "";
int currentPlanId = 0;

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

unsigned long secondsToMs(unsigned long seconds) {
  return seconds * 1000UL;
}

void setAllOff() {
  digitalWrite(A_RED_PIN, LOW);
  digitalWrite(A_YELLOW_PIN, LOW);
  digitalWrite(A_GREEN_PIN, LOW);

  digitalWrite(B_RED_PIN, LOW);
  digitalWrite(B_YELLOW_PIN, LOW);
  digitalWrite(B_GREEN_PIN, LOW);
}

void applyStateOutputs(TrafficState state) {
  setAllOff();

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

unsigned long getStateDurationMs(TrafficState state) {
  switch (state) {
    case STATE_A_GREEN:
      return secondsToMs(greenASeconds);

    case STATE_A_YELLOW:
      return secondsToMs(yellowSeconds);

    case STATE_ALL_RED_AFTER_A:
      return secondsToMs(allRedSeconds);

    case STATE_B_GREEN:
      return secondsToMs(greenBSeconds);

    case STATE_B_YELLOW:
      return secondsToMs(yellowSeconds);

    case STATE_ALL_RED_AFTER_B:
      return secondsToMs(allRedSeconds);
  }

  return secondsToMs(1);
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
  stateStartMs = millis();
  applyStateOutputs(currentState);

  Serial.print("[MCU/STATE] ");
  Serial.println(stateToString(currentState));
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

  // Basic validation.
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

  // Apply new plan immediately from A_GREEN for easier testing.
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

  stateStartMs = millis();
  applyStateOutputs(currentState);


  Serial.println("[MCU/READY] ESP32_TRAFFIC_LIGHT_READY");
  Serial.println("[MCU/DEFAULT] DEFAULT_PLAN,10,10,3,1");
}

void loop() {
  readSerialInput();
  applyFallbackTimingIfNeeded();

  unsigned long now = millis();
  unsigned long elapsedMs = now - stateStartMs;
  unsigned long durationMs = getStateDurationMs(currentState);

  if (elapsedMs >= durationMs) {
    TrafficState nextState = getNextState(currentState);
    transitionTo(nextState);
  }
}