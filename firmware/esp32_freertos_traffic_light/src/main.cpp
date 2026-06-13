/*
 * ATLC Phase 15 — FreeRTOS-Based Embedded Controller and Edge Deployment Upgrade
 */

#include <Arduino.h>

// Must be the same speed as the Serial Monitor
static const uint32_t SERIAL_BAUD_RATE = 115200;

// Queue length: 5 RawMessage objects at the same time
static const UBaseType_t RAW_MESSAGE_QUEUE_LENGTH = 5;

// Queue length: 3 validated SignalPlan objects at the same time
static const UBaseType_t PLAN_QUEUE_LENGTH = 3;

// Maximum serial command length, including null terminator
static const size_t SERIAL_LINE_BUFFER_SIZE = 96;

// TaskSimulatedProducer will send one fake message every 3000 ms
// static const TickType_t PRODUCER_PERIOD_TICKS = pdMS_TO_TICKS(3000);

// UART receive task checks Serial periodically
static const TickType_t UART_RECEIVE_POLL_TICK = pdMS_TO_TICKS(20); //Phase 15.4 


// Stack size for each task
// static const uint32_t PRODUCER_TASK_STACK_SIZE = 4096;
static const uint32_t UART_RECEIVE_TASK_STACK_SIZE = 4096; //Phase 15.4 
static const uint32_t PARSER_TASK_STACK_SIZE = 4096;
static const uint32_t FSM_TASK_STACK_SIZE = 4096;

// Task priorities
// static const UBaseType_t PRODUCER_TASK_PRIORITY = 1;
static const UBaseType_t UART_RECEIVE_TASK_PRIORITY = 1; //Phase 15.4 
static const UBaseType_t PARSER_TASK_PRIORITY = 1;
static const UBaseType_t FSM_TASK_PRIORITY = 1;

// Timing validation limits
// These values mirror the host-side scheduler configuration in pc_app/config.py.
// The MCU still validates them because it must not blindly trust host input.
static const int MIN_GREEN_SECONDS = 10;
static const int MAX_GREEN_SECONDS = 45;

static const int MIN_YELLOW_SECONDS = 3;
static const int MAX_YELLOW_SECONDS = 3;

static const int MIN_ALL_RED_SECONDS = 1;
static const int MAX_ALL_RED_SECONDS = 1;

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

// SignalPlan represents a validated traffic signal timing plan which will be sent into planQueue
// ParsedPlanFields = raw extracted fields
// SignalPlan       = validated plan for the controller
struct SignalPlan
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
static QueueHandle_t rawMessageQueue = nullptr;     // Stores RawMessage
static QueueHandle_t planQueue = nullptr;           // stores Signal Plan

static void printBootBanner()
{
    Serial.println();
    Serial.println("[BOOT] ATLC Phase 15 FreeRTOS Controller");
    Serial.println("[BOOT] Phase 15.4 - Real USB Serial Receive Task");
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

// Convert parsed fields into a SignalPlan object
static SignalPlan makeSignalPlan(const ParsedPlanFields *fields)
{
    SignalPlan plan;

    plan.plan_id = fields->plan_id;
    plan.green_a = fields->green_a;
    plan.green_b = fields->green_b;
    plan.yellow = fields->yellow;
    plan.all_red = fields->all_red;

    return plan;

}

// Validate timing values by checking timing ranges before sending the plan to planQueue
static bool validateSignalPlan(const SignalPlan *plan, const char **reason)
{
    if (reason != nullptr) 
    {
        *reason = "OK";
    }

    if (plan == nullptr)
    {
        if(reason != nullptr)
        {
            *reason = "NULL_PLAN";
        }
        return false;
    }

    if (plan->green_a < MIN_GREEN_SECONDS || plan->green_a > MAX_GREEN_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "GREEN_A_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->green_b < MIN_GREEN_SECONDS || plan->green_b > MAX_GREEN_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "GREEN_B_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->yellow < MIN_YELLOW_SECONDS || plan->yellow > MAX_YELLOW_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "YELLOW_OUT_OF_RANGE";
        }
        return false;
    }

    if (plan->all_red < MIN_ALL_RED_SECONDS || plan->all_red > MAX_ALL_RED_SECONDS)
    {
        if (reason != nullptr)
        {
            *reason = "ALL_RED_OUT_OF_RANGE";
        }
        return false;
    }

    return true;
}

// Print a validated SignalPlan for debugging
static void printSignalPlan(const SignalPlan *plan)
{
    if (plan == nullptr)
    {
        return;
    }

    Serial.print("[PLAN] plan_id=");
    Serial.print(plan->plan_id);
    Serial.print(" green_a=");
    Serial.print(plan->green_a);
    Serial.print(" green_b=");
    Serial.print(plan->green_b);
    Serial.print(" yellow=");
    Serial.print(plan->yellow);
    Serial.print(" all_red=");
    Serial.println(plan->all_red);
}

/*
 * TaskSimulatedProducer:
 * Simulates a future host computer sending a command over USB Serial.
 */
// In Phase 15.4 We dont neeedSimulated Producer anymore 
// void TaskSimulatedProducer(void *pvParameters)
// {
//     // Prevent compiler warnings
//     (void)pvParameters;

//     // The producer sends multiple test messages.
//     const char *testMessages[] = {
//         "PLAN,17,25,15,3,1",    // valid
//         "HELLO",                // unknown
//         "BAD,17,25",            // unknown
//         "PLAN,18,30,20,3,1",    // valid
//         "PLAN,19,2,15,3,1",     // invalid green_a
//         "PLAN,20,25,200,3,1",   // invalid green_b
//         "PLAN,21,25,15,0,1"     // invalid yellow
//     };

//     size_t messageIndex = 0;
//     const size_t messageCount = sizeof(testMessages) / sizeof(testMessages[0]);

//     for (;;)
//     {
//         // Local RawMessage variable
//         RawMessage message;

//         // Fill message.data with fake PLAN text
//         setRawMessage(&message, testMessages[messageIndex]);

//         messageIndex++;
//         if (messageIndex >= messageCount)
//         {
//             messageIndex = 0;
//         }

//         // Send the RawMessage into the queue
//         BaseType_t sendResult = xQueueSendToBack(
//             rawMessageQueue,
//             &message,
//             portMAX_DELAY
//         );

//         // Check if the send operation succeeded
//         if (sendResult == pdPASS)
//         {
//             Serial.print("[PRODUCER] Sent raw message: ");
//             Serial.println(message.data);
//         }
//         else
//         {
//             Serial.println("[PRODUCER] ERROR: Failed to send raw message.");
//         }

//         // Sleep for 3 seconds
//         vTaskDelay(PRODUCER_PERIOD_TICKS);
//     }
// }

/*
 * TaskUARTReceive: 
 * Reads command lines from USB Serial and sends them to rawMessageQueue.
 */
void TaskUARTReceive(void *pvParameters)
{
    (void)pvParameters;

    char lineBuffer[SERIAL_LINE_BUFFER_SIZE];
    size_t lineIndex = 0;
    for (;;)
    {
        while(Serial.available() > 0) // Check if characters exist
        {
            char receivedChar = static_cast<char>(Serial.read());

            if (receivedChar == '\r') 
            {
                continue;
            }

            if (receivedChar == '\n') // New line means: User pressed Enter or Command complete.
            {
                // Check Empty line
                if (lineIndex > 0)
                {
                    lineBuffer[lineIndex] = '\0'; // Without \0, C does know where the string ends
                    RawMessage message;
                    setRawMessage(&message, lineBuffer);

                    // Send RawMessage to parser task
                    BaseType_t sendResult = xQueueSendToBack(
                        rawMessageQueue,
                        &message,
                        portMAX_DELAY
                    );

                        if (sendResult == pdPASS)
                        {
                            Serial.print("[UART] Received line: ");
                            Serial.println(message.data);
                        }
                        else
                        {
                            Serial.println("[UART] ERROR: Failed to send raw message.");
                        }

                        lineIndex = 0;
                }
            }
            else 
            {
                if (lineIndex < SERIAL_LINE_BUFFER_SIZE -1) 
                {
                    lineBuffer[lineIndex] = receivedChar;
                    lineIndex++;
                }
                else
                {
                    Serial.println("[UART] ERROR: Serial line too long. Dropping line.");
                    lineIndex = 0;
                }
            }
        }
        vTaskDelay(UART_RECEIVE_POLL_TICK);
    }
}

/*
 * TaskPlanParser:
 * Waits for RawMessage objects from rawMessageQueue.
 * Parser behavior: 
 * Valid PLAN
        → parse
        → validate
        → send to planQueue

    Invalid PLAN
        → parse
        → fail validation
        → reject

    Unknown message
        → reject before parsing
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

                    SignalPlan plan = makeSignalPlan(&fields);

                    const char *validationReason = "OK";
                    if (validateSignalPlan(&plan, &validationReason))
                    {
                        BaseType_t sendResult = xQueueSendToBack(
                            planQueue,
                            &plan,
                            portMAX_DELAY);

                        if (sendResult == pdPASS)
                        {
                            Serial.println("[PARSER] Valid SignalPlan sent to planQueue.");
                        }
                        else
                        {
                            Serial.println("[PARSER] ERROR: Failed to send SignalPlan to planQueue.");
                        }
                    }
                    else
                    {
                        Serial.print("[PARSER] Rejected SignalPlan: ");
                        Serial.println(validationReason);
                    }
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

/* 
 * TaskTrafficFSMPlaceholder:
 * Receives validated SignalPlan object from planQueue
 * 
 */
void TaskTrafficFSMPlaceholder(void *pvParmeters)
{
    (void)pvParmeters;

    SignalPlan receivedPlan;

    for (;;)
    {
        BaseType_t receiveResult = xQueueReceive(
            planQueue,
            &receivedPlan,
            portMAX_DELAY
        );

        if (receiveResult == pdPASS)
        {
            Serial.println("[FSM] Received validated SignalPlan from planQueue.");
            printSignalPlan(&receivedPlan);
        }
        else
        {
            Serial.println("[FSM] ERROR: Failed to receive SignalPlan.");
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

    // Create the second queue forplan
    planQueue = xQueueCreate(
        PLAN_QUEUE_LENGTH,
        sizeof(SignalPlan)
    );

    if (planQueue == nullptr)
    {
        Serial.println("[BOOT] ERROR: Failed to create planQueue.");
        Serial.println("[BOOT] System halted. ");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("[BOOT] planQueue created.");

    // In Phase 15.4 We dont neeedSimulated Producer anymore 
    /*
     * Create the producer task
     */
    // BaseType_t producerCreated = xTaskCreate(
    //     TaskSimulatedProducer,    // Function that implements the task
    //     "SimProducer",            // Human-readable task name
    //     PRODUCER_TASK_STACK_SIZE, // Stack size
    //     nullptr,                  // No parameter passed to the task
    //     PRODUCER_TASK_PRIORITY,   // Task priority
    //     nullptr                   // Do not store task handle for now
    // );

    // Check if task creation failed
    // if (producerCreated != pdPASS)
    // {
    //     Serial.println("[BOOT] ERROR: Failed to create TaskSimulatedProducer.");
    //     Serial.println("[BOOT] System halted.");

    //     while (true)
    //     {
    //         delay(1000);
    //     }
    // }

    // Serial.println("[BOOT] TaskSimulatedProducer created.");

    /*
     * Create the UART receive task 
     */
    BaseType_t uartReceiveCreated = xTaskCreate(
        TaskUARTReceive,
        "UARTReceive",
        UART_RECEIVE_TASK_STACK_SIZE,
        nullptr,
        UART_RECEIVE_TASK_PRIORITY,
        nullptr
    );

    // CHeck if task creation failed
    if (uartReceiveCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskUARTReceive.");
        Serial.println("[BOOT] System halted.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("[BOOT] TaskUARTReceive created.");

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

    /*
     * Create the FSM Placeholder Task  
     */
    BaseType_t fsmCreated = xTaskCreate(
        TaskTrafficFSMPlaceholder,
        "FSMPlaceholder",
        FSM_TASK_STACK_SIZE,
        nullptr,
        FSM_TASK_PRIORITY,
        nullptr
    );
    
    if (fsmCreated != pdPASS)
    {
        Serial.println("[BOOT] ERROR: Failed to create TaskTrafficFSMPlaceholder.");
        Serial.println("[BOOT] System halted.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("[BOOT] TaskTrafficFSMPlaceholder created.");



    Serial.println("[BOOT] Phase 15.4 - Real USB Serial Receive Task");
}

// In Arduino projects, loop() usually contains the main logic.
// In this FreeRTOS design, loop() stays minimal.
void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}