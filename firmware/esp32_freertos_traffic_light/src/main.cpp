/*
 * ATLC Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade
 */

#include <Arduino.h>

// Must be the same speed as the Serial Monitor
static const uint32_t SERIAL_BAUD_RATE = 115200;

// Queue length: 5 RawMessage objects at the same time
static const UBaseType_t RAW_MESSAGE_QUEUE_LENGTH = 5;

// TaskSimulatedProducer will send one fake message every 3000 ms
static const TickType_t PRODUCER_PERIOD_TICKS = pdMS_TO_TICKS(3000);

// Stack size for each task
static const uint32_t PRODUCER_TASK_STACK_SIZE = 4096;
static const uint32_t PARSER_TASK_STACK_SIZE = 4096;

// Task priorities
static const UBaseType_t PRODUCER_TASK_PRIORITY = 1;
static const UBaseType_t PARSER_TASK_PRIORITY = 1;

// RawMessage represents one raw command line
// Example: PLAN,17,25,15,3,1
struct RawMessage
{
    char data[96]; // Fixed-size character buffer
};

// Temporary structure for parsed PLAN fields
struct ParsedPlanFields
{
    int plan_id;
    int green_a;
    int green_b;
    int yellow;
    int all_red;
};

/*
 * Global FreeRTOS Objects
 */

// Queue handle
static QueueHandle_t rawMessageQueue = nullptr;

static void printBootBanner()
{
    Serial.println();
    Serial.println("[BOOT] ATLC Phase 15 FreeRTOS Controller");
    Serial.println("[BOOT] Phase 15.2 -  Queue-Based Fake PLAN Parser");
}

// Copy text into a RawMessage object safely
static void setRawMessage(RawMessage *message, const char *text)
{
    if (message == nullptr || text == nullptr)
    {
        return;
    }

    snprintf(
        message->data,         // Where to write
        sizeof(message->data), // Maximum size: 96 bytes
        "%s",                  // Insert a string
        text                   // String to insert
    );
}

// Check whether a raw message starts with "PLAN, "
static bool isPlanCommand(const RawMessage *message)
{
    if (message == nullptr)
    {
        return false;
    }
    return strncmp(message->data, "PLAN,", 5) == 0;
}

// Parse: PLAN,<plan_id>,<green_a>,<green_b>,<yellow>,<all_red>
static bool parsePlanCommand(const RawMessage *message, ParsedPlanFields *fields)
{
    if(message == nullptr || fields == nullptr)
    {
        return false;
    }

    // ParsedPlanFields = extracted numbers
    // sscanf() returns: How many values were successfully parsed
    int parsedCount = sscanf(
        message->data,
        "PLAN,%d,%d,%d,%d,%d",
        &fields->plan_id,
        &fields->green_a,
        &fields->green_b,
        &fields->yellow,
        &fields->all_red
    );
    return parsedCount == 5;
}

// Print parsed PLAN fields for debugging
static void printParsedPlan  (const ParsedPlanFields *fields)
{
    if (fields == nullptr)
    {
        return;
    }

    Serial.print("[PARSER] plan_id=");
    Serial.print(fields->plan_id);

    Serial.print(" green_a=");
    Serial.print(fields->green_a);

    Serial.print(" green_b=");
    Serial.print(fields->green_b);

    Serial.print(" yellow=");
    Serial.print(fields->yellow);

    Serial.print(" all_red=");
    Serial.println(fields->all_red);
}


/*
 * TaskSimulatedProducer:
 * Simulates a future host computer sending a command over USB Serial.
 */
void TaskSimulatedProducer(void *pvParameters)
{
    // Prevent compiler warnings
    (void)pvParameters;

    // The producer sends multiple test messages.
    const char *testMessages[] = {
        "PLAN,17,25,15,3,1",
        "HELLO",
        "BAD,17,25",
        "PLAN,18,30,20,3,1"
    };

    size_t messageIndex = 0;
    const size_t messageCount = sizeof(testMessages) / sizeof(testMessages[0]);

    for (;;)
    {
        // Local RawMessage variable
        RawMessage message;

        // Fill message.data with fake PLAN text
        setRawMessage(&message, testMessages[messageIndex]);

        messageIndex++;
        if (messageIndex >= messageCount)
        {
            messageIndex = 0;
        }

        // Send the RawMessage into the queue
        BaseType_t sendResult = xQueueSendToBack(
            rawMessageQueue,
            &message,
            portMAX_DELAY
        );

        // Check if the send operation succeeded
        if (sendResult == pdPASS)
        {
            Serial.print("[PRODUCER] Sent raw message: ");
            Serial.println(message.data);
        }
        else
        {
            Serial.println("[PRODUCER] ERROR: Failed to send raw message.");
        }

        // Sleep for 3 seconds
        vTaskDelay(PRODUCER_PERIOD_TICKS);
    }
}

/*
 * TaskPlanParser:
 * Waits for RawMessage objects from rawMessageQueue.
 */
void TaskPlanParser(void *pvParameters)
{
    // Prevent compiler warnings
    (void)pvParameters;

    // Local variable to store the message received from the queue
    RawMessage receivedMessage;

    for (;;)
    {
        BaseType_t receiveResult = xQueueReceive(
            rawMessageQueue,
            &receivedMessage,
            portMAX_DELAY
        );

        // If receive succeeded, print the raw message
        if (receiveResult == pdPASS)
        {
            Serial.print("[PARSER] Received raw message: ");
            Serial.println(receivedMessage.data);

            if (isPlanCommand(&receivedMessage))
            {
                ParsedPlanFields fields;

                if (parsePlanCommand(&receivedMessage, &fields))
                {
                    Serial.println("[PARSER] PLAN command detected.");
                    printParsedPlan(&fields);
                }
                else
                {
                    Serial.println("[PARSER] ERROR: Malformed PLAN command.");
                }
            }
            else
            {
                Serial.println("[PARSER] Unknown command format.");
            }
        }
        else
        {
            Serial.println("[PARSER] ERROR: Failed to receive raw message.");
        }
    }
}

void setup()
{
    Serial.begin(SERIAL_BAUD_RATE);

    // Small delay to give Serial Monitor time to connect
    delay(1000);

    // Print boot information
    printBootBanner();

    // Create the queue before creating tasks
    rawMessageQueue = xQueueCreate(
        RAW_MESSAGE_QUEUE_LENGTH,
        sizeof(RawMessage)
    );

    // Check if queue creation failed
    if (rawMessageQueue == nullptr)
    {
        Serial.println("[BOOT] ERROR: Failed to create rawMessageQueue.");
        Serial.println("[BOOT] System halted.");

        // Stop here forever because the tasks need the queue
        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("[BOOT] rawMessageQueue created.");

    /*
     * Create the producer task
     */
    BaseType_t producerCreated = xTaskCreate(
        TaskSimulatedProducer,    // Function that implements the task
        "SimProducer",            // Human-readable task name
        PRODUCER_TASK_STACK_SIZE, // Stack size
        nullptr,                  // No parameter passed to the task
        PRODUCER_TASK_PRIORITY,   // Task priority
        nullptr                   // Do not store task handle for now
    );

    // Check if task creation failed
    if (producerCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskSimulatedProducer.");
        Serial.println("[BOOT] System halted.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("[BOOT] TaskSimulatedProducer created.");

    /*
     * Create the parser task
     */
    BaseType_t parserCreated = xTaskCreate(
        TaskPlanParser,          // Function that implements the task
        "PlanParser",            // Human-readable task name
        PARSER_TASK_STACK_SIZE,  // Stack size
        nullptr,                 // No parameter passed to the task
        PARSER_TASK_PRIORITY,    // Task priority
        nullptr                  // Do not store task handle for now
    );

    // Check if task creation failed
    if (parserCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskPlanParser.");
        Serial.println("[BOOT] System halted.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("[BOOT] TaskPlanParser created.");
    Serial.println("[BOOT] Phase 15.2 - Queue-Based Fake PLAN Parser");
}

// In Arduino projects, loop() usually contains the main logic.
// In this FreeRTOS design, loop() stays minimal.
void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}