# YOLO Hyperparameter Tuning Notes

## Purpose

This document records the current detector tuning observations for the local YOLO pipeline.

The goal is to understand why the pretrained YOLO model performs poorly on motorbike detection in the current traffic-camera scene, and to document the trade-off between detection quality and runtime performance.

---

## Test Context

The current test scene is a traffic-camera-style video frame with:

- high-angle camera view
- dense traffic
- many small motorbikes
- partial occlusion between vehicles
- fixed road-side perspective
- multiple lanes and road regions
- cars, buses, trucks, and motorbikes appearing at different scales

This scene is close to the expected final deployment scenario because the real system will likely use a fixed camera angle mounted above or beside the road.

---

## Observed Problem

The pretrained YOLO model detects cars, buses, and trucks reasonably well, but performs poorly on motorbikes.

Main observed issue:

```text
Motorbike recall is low.
Many visible motorbikes are missed by the detector.

This is a critical issue for this project because motorbikes are an important traffic participant in the target scenario. If motorbikes are missed, the density estimation and adaptive signal timing will become less accurate.

Current Hypothesis

The poor motorbike detection is likely caused by domain mismatch.

The pretrained model is a general-purpose object detector, while the project requires reliable detection in a specific traffic-camera domain.

Possible reasons:

motorbikes are small in the image
motorbikes are often partially occluded
the high-angle camera view changes the object appearance
dense traffic makes individual motorbikes harder to separate
the model is not specifically optimized for Vietnamese-style road traffic
the pretrained model is trained for general object detection, not only vehicle detection
Hyperparameter Tuning Attempt

Two main parameters were tested:

YOLO_IMGSZ=960
CONFIDENCE_THRESHOLD=0.2

Previous baseline:

YOLO_IMGSZ=640
CONFIDENCE_THRESHOLD=0.3
YOLO_IMGSZ Trade-off

Increasing image size from:

640 → 960

improved detection of small vehicles to some extent.

Reason:

Larger input size preserves more visual detail for small objects such as motorbikes.

However, this creates a clear runtime trade-off:

Higher YOLO_IMGSZ → better small-object detection
Higher YOLO_IMGSZ → slower inference
Higher YOLO_IMGSZ → lower FPS

Therefore, increasing YOLO_IMGSZ can partially improve detection quality, but it is not a complete solution because the final system must also run efficiently on Raspberry Pi.

Confidence Threshold Trade-off

The confidence threshold was reduced from:

0.3 → 0.2

This did not significantly improve motorbike detection in the current test.

Observation:

When motorbikes are detected, their confidence is usually already above 0.3.

Therefore, the issue is probably not mainly caused by confidence filtering.

Instead, the detector often fails to localize or classify motorbikes in the first place.

This supports the hypothesis that the main problem is model-domain mismatch, not only threshold tuning.

Current Benchmark Interpretation

The tuning result suggests:

YOLO_IMGSZ affects motorbike detection more than CONFIDENCE_THRESHOLD.

But increasing image size reduces FPS noticeably.

This creates a project-level trade-off:

Better motorbike recall
vs
Real-time inference performance

For the current adaptive traffic light system, this trade-off matters because the detector must support both:

accurate traffic density estimation
acceptable runtime speed on edge hardware
Engineering Conclusion

The pretrained YOLO model is not sufficient for reliable motorbike detection in the current traffic-camera scenario.

Hyperparameter tuning can partially help, but it does not fully solve the issue.

The stronger solution is to fine-tune or train a custom vehicle detector using a dataset that matches the final deployment scenario.

Next Direction

The next recommended step is to prepare a custom traffic dataset focused on the target camera perspective.

Recommended custom classes:

car
bus
truck
motorbike
bicycle

The dataset should include:

high-angle road camera views
dense traffic
many motorbikes
partially occluded vehicles
different lighting conditions
vehicles at different distances
frames similar to the final Raspberry Pi camera placement

The model should be fine-tuned from a pretrained YOLO model instead of trained from scratch.

Recommended candidates:

YOLO26n pretrained → custom traffic YOLO26n
YOLOv8n pretrained → custom traffic YOLOv8n baseline
Expected Benefit of Custom Training

Custom training is expected to improve:

motorbike recall
detection consistency
domain-specific vehicle detection
density estimation quality
final traffic signal timing reliability

However, custom training is not expected to significantly improve FPS if the same model architecture is used.

For example:

YOLO26n pretrained
vs
custom-trained YOLO26n

Both have similar architecture and similar inference cost.

Therefore:

Custom training mainly improves accuracy.
Deployment optimization mainly improves FPS.

FPS should later be improved using:

lower input size if acceptable
frame skipping
ONNX export
NCNN export
Raspberry Pi benchmarking
possible ROI-based cropping
Final Note

The current tuning result justifies the need for a custom-trained detector.

The project should continue with local YOLO deployment, but the final detection model should be fine-tuned on traffic-camera data that matches the real deployment environment.
