# Placeholder Image Prompts

Save each generated image as a PNG with the exact filename shown here, then upload it into `reports/thesis_latex/figures/` or the Overleaf `figures/` folder. The LaTeX report will automatically replace the placeholder box when the file exists.

## General Style Rules

Use the same visual style for all system diagrams:

- flat vector block diagram, clean engineering report style
- light pastel background, soft colored blocks, dark thin arrows
- large readable English text, at least 28-34 pt equivalent in a 16:9 image
- no Vietnamese text
- no tiny bullet text; use short phrases only
- no hand-drawn sketch effect
- no decorative blobs, stickers, shadows, or cartoon elements
- keep generous spacing between blocks
- export as a sharp PNG, 16:9 landscape, at least 2400 x 1350 px

## atlc_block_diagram.png

Prompt:

```text
Create a clean 16:9 flat vector block diagram for a university engineering report about an Adaptive Traffic Light Control (ATLC) system. Use a consistent pastel technical diagram style: light mint background, soft colored rectangular blocks, dark thin arrows, no hand-drawn effect, no decorative stickers, no Vietnamese text.

Make all text large and readable, at least 28-34 pt equivalent. Avoid tiny bullet lists. Use short English labels only.

Diagram title: "ATLC System Block Diagram"

Show the system as a true block diagram with five main left-to-right blocks:
1. "Camera / Video Input"
2. "Edge Host (PC / Raspberry Pi)"
3. "UART Link"
4. "MCU Controller (ESP32)"
5. "Traffic Light Output"

Inside the Edge Host block, include four smaller internal sub-blocks connected top-to-bottom:
"YOLO Vehicle Detection" -> "ROI & Direction Counting" -> "PCE Density Estimation" -> "Signal Scheduler"

Inside the MCU Controller block, include three smaller internal sub-blocks:
"PLAN Validation" -> "Local FSM" -> "GPIO Driver"

Show a small monitoring/logging block connected to the Edge Host labeled "Dashboard & Runtime Logs".
Show an arrow from the Scheduler to UART labeled "PLAN".
Show a return arrow from MCU to Host labeled "ACK / STATUS".
Show an arrow from MCU GPIO Driver to Traffic Light Output.

At the bottom, add one large readable note in English:
"AI estimates traffic demand; the MCU performs real-time signal execution."

The diagram should look similar to a textbook computer organization block diagram: clean boxes, clear arrows, large readable labels, and consistent colors.
```

## host_mcu_split.png

Prompt:

```text
Create a clean 16:9 flat vector architecture diagram for a university embedded systems report. The image must be entirely in English. Use the same visual style as the ATLC block diagram: light pastel background, soft colored blocks, dark thin arrows, large readable text, no hand-drawn effect, no Vietnamese text, no decorative icons.

Make all text large and readable, at least 28-34 pt equivalent. Use short labels, not long bullet paragraphs.

Diagram title: "Host-MCU Responsibility Split"

Use a two-column layout:

Left column title: "Host Layer (PC / Raspberry Pi)"
Inside it, show stacked sub-blocks:
"Video Capture"
"YOLO Detection"
"Class Normalization"
"ROI Assignment"
"PCE Density + EMA"
"Signal Timing Plan"
"UART Sender"

Right column title: "MCU Layer (ESP32)"
Inside it, show stacked sub-blocks:
"UART Receiver"
"PLAN Format Check"
"Timing Range Validation"
"ACK / NACK Response"
"Local Traffic FSM"
"GPIO Output"
"Watchdog Fallback"

Between columns, show two thick arrows:
Host to MCU: "PLAN over UART"
MCU to Host: "ACK / STATUS over UART"

At the bottom of the left column, add a small label: "High-level perception and planning".
At the bottom of the right column, add a small label: "Low-level real-time execution".

Use blue tones for the Host column and orange tones for the MCU column. Keep spacing generous and text very legible.
```

## bounding_box.png

Prompt:

```text
Create a clean technical computer vision illustration for an adaptive traffic light control report. Show a real-looking urban traffic camera frame with cars, motorbikes, a bus, and a truck. Overlay several YOLO-style bounding boxes with large readable class labels and confidence scores such as "car 0.91", "motorbike 0.86", "bus 0.88". Use professional colors, thin rectangular boxes, readable labels, and no decorative elements. The image should look like an annotated detection output, not a presentation slide. Make label text large enough to read in a printed A4 report.
```

## roi_example.png

Prompt:

```text
Create a technical traffic-camera illustration showing a road intersection or multi-lane road from an elevated fixed camera. Draw two semi-transparent polygon regions of interest labeled "Direction A ROI" and "Direction B ROI". Include a few vehicles inside and outside the polygons. The style should be realistic and suitable for a university engineering report. Use thin green and blue polygon outlines, large readable labels, and no cartoon style. Make the ROI labels large enough to read in a printed A4 report.
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
