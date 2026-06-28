#include <Arduino.h>

#include "config/app_config.h"
#include "core/traffic_fsm.h"
#include "messages/messages.h"

SignalPlan getDefaultSignalPlan()
{
    return SignalPlan {
        .plan_id = 0,
        .green_a = 20,
        .green_b = 20,
        .yellow = 3,
        .all_red = 1,
    };
}

uint32_t secondsToMs (int seconds)
{
    return static_cast<uint32_t>(seconds) * 1000UL;
}

void setupTrafficLightPins()
{
    pinMode(A_RED_PIN, OUTPUT);
    pinMode(A_YELLOW_PIN, OUTPUT);
    pinMode(A_GREEN_PIN, OUTPUT);

    pinMode(B_RED_PIN, OUTPUT);
    pinMode(B_YELLOW_PIN, OUTPUT);
    pinMode(B_GREEN_PIN, OUTPUT);

    setAllLightsOff();
}

void initTrafficFsm()
{
    setupTrafficLightPins();
}

void setAllLightsOff()
{
    digitalWrite(A_RED_PIN, LOW);
    digitalWrite(A_YELLOW_PIN, LOW);
    digitalWrite(A_GREEN_PIN, LOW);

    digitalWrite(B_RED_PIN, LOW);
    digitalWrite(B_YELLOW_PIN, LOW);
    digitalWrite(B_GREEN_PIN, LOW);
}

const char *trafficStateToString(TrafficState state)
{
    switch (state)
    {
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

        default:
            return "UNKNOWN";
    }
}

void applyTrafficOutputs(TrafficState state)
{
    setAllLightsOff();

    switch (state)
    {
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

        default:
            setAllLightsOff();
            break;
    }
}

uint32_t getStateDurationMs(
    TrafficState state,
    const SignalPlan *activePlan
)
{
    if(activePlan == nullptr)
    {
        return secondsToMs(1);
    }

    switch(state)
    {
        case STATE_A_GREEN:
            return secondsToMs(activePlan->green_a);

        case STATE_A_YELLOW:
            return secondsToMs(activePlan->yellow);

        case STATE_ALL_RED_AFTER_A:
            return secondsToMs(activePlan->all_red);

        case STATE_B_GREEN:
            return secondsToMs(activePlan->green_b);

        case STATE_B_YELLOW:
            return secondsToMs(activePlan->yellow);

        case STATE_ALL_RED_AFTER_B:
            return secondsToMs(activePlan->all_red);

        default:
            return secondsToMs(1);
    }
}

TrafficState getNextTrafficState(TrafficState state)
{
    switch (state)
    {
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

        default:
            return STATE_A_GREEN;
    }
}
