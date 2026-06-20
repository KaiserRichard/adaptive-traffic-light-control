# Placeholder Image Prompts

Save each generated image as a PNG with the exact filename shown here, then upload it into `reports/thesis_latex/figures/` or the Overleaf `figures/` folder. The LaTeX report will automatically replace the placeholder box when the file exists.

## bounding_box.png

Prompt:

```text
Create a clean technical computer vision illustration for an adaptive traffic light control report. Show a real-looking urban traffic camera frame with cars, motorbikes, a bus, and a truck. Overlay several YOLO-style bounding boxes with class labels and confidence scores such as "car 0.91", "motorbike 0.86", "bus 0.88". Use professional colors, thin rectangular boxes, readable labels, and no decorative elements. The image should look like an annotated detection output, not a presentation slide.
```

## roi_example.png

Prompt:

```text
Create a technical traffic-camera illustration showing a road intersection or multi-lane road from an elevated fixed camera. Draw two semi-transparent polygon regions of interest labeled "Direction A ROI" and "Direction B ROI". Include a few vehicles inside and outside the polygons. The style should be realistic and suitable for a university engineering report. Use thin green and blue polygon outlines, clear labels, and no cartoon style.
```

## esp32_gpio_traffic_light_mapping.png

Prompt:

```text
Create a clean embedded-systems wiring diagram for an ESP32 traffic light prototype. Show an ESP32 development board connected to six LEDs: A_RED, A_YELLOW, A_GREEN, B_RED, B_YELLOW, B_GREEN. Label GPIO pins 25, 26, 27, 14, 12, and 13. Include resistors for each LED and a shared ground. Use a professional engineering schematic style with simple lines, readable pin labels, and no decorative background.
```

## freertos_task_queue_architecture.png

Prompt:

```text
Create a professional FreeRTOS architecture diagram for an ESP32 adaptive traffic light controller. Show the flow: UART RX callback -> TaskUARTReceive -> rawMessageQueue -> TaskPlanParser -> planQueue -> TaskTrafficFSM -> GPIO traffic lights. Also show StatusTimer notifying TaskStatusReporter, DiagnosticsTimer, and a Serial mutex protecting ACK, NACK, STATUS, DIAG, and debug logs. Use clear blocks, arrows, and concise labels. The style should be suitable for an embedded systems report.
```

## uart_rx_deferred_processing.png

Prompt:

```text
Create a technical sequence diagram showing deferred UART receive processing on ESP32 FreeRTOS. Participants: Host, UART RX callback, TaskUARTReceive, rawMessageQueue, TaskPlanParser. Show that the callback only sends a task notification, TaskUARTReceive drains serial bytes and builds a RawMessage, then sends it to rawMessageQueue, and TaskPlanParser parses the PLAN command later. Use a clean black-and-white engineering style with small blue highlights.
```

## safe_plan_apply_timeline.png

Prompt:

```text
Create a timeline diagram for safe traffic light plan application. Show an active plan running through A_GREEN, A_YELLOW, ALL_RED, B_GREEN, B_YELLOW, ALL_RED. Show a new PLAN message arriving in the middle of a cycle and being stored as pending_plan. Then show it applied only when the FSM reaches the next A_GREEN boundary. Use a clear engineering timeline style with labels "raw plan", "pending plan", and "active plan".
```

## status_notification_flow.png

Prompt:

```text
Create a FreeRTOS status reporting flow diagram. Show TaskTrafficFSM updating a shared ControllerStatus snapshot, StatusTimer callback sending a task notification, TaskStatusReporter waking up, copying the snapshot, formatting "STATUS,<plan_id>,<state>,<remaining_seconds>,<health>", and printing through a Serial mutex. Emphasize that the timer callback does not print Serial directly. Use clean blocks and arrows for a university embedded systems report.
```
