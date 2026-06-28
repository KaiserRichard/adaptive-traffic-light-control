#pragma once

#include <Arduino.h>

#include "messages/messages.h"

SignalPlan getDefaultSignalPlan();

uint32_t secondsToMs(int seconds);

void initTrafficFsm();

void setupTrafficLightPins();

void setAllLightsOff();

const char *trafficStateToString(TrafficState state);

void applyTrafficOutputs(TrafficState state);

uint32_t getStateDurationMs(
    TrafficState state,
    const SignalPlan *activePlan
);

TrafficState getNextTrafficState(TrafficState state);
