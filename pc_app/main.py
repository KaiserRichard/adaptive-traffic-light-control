'''
main.py

Phase 1 full pipeline:
1. Open Video
2. Initialize detector
3. Read frame by frame
4. Run detection
5. Count classes
6. Compute FPS
7. Draw boxes and status panel
8. Show on screen 
9. Optinally save annotated video

'''

import time
from pathlib import Path
import cv2

from pc_app.config import (
    VIDEO_SOURCE,
    SHOW_WINDOW, 
    SAVE_OUTPUT_VIDEO,
    OUTPUT_VIDEO_PATH
)

from pc_app.vision.detector import RoboflowDetector

from pc_app.vision.visualize import (
    draw_detections,
    count_by_class,
    draw_counts_panel,
)


def main():
    '''
    Main loop for Phase 1
    '''
    # Initialize detector
    detector = RoboflowDetector()

    # Open the video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    # Video writer will be created only after the first frame
    # because we need frame width/height first
    writer = None

    if SAVE_OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

        # Store previous frame timestamps for FPS estimation

    prev_time = time.time()

    while True:
        # Read one frame
        ret, frame = cap.read()

        if not ret:
            # End of video or failed read
            break

        # 1) Detect objects
        detections = detector.detect(frame=frame)

        # 2) Count objects by class
        counts = count_by_class(detections=detections)

        # 3) Compute FPS
        # dt = time spent since previous frame
        current_time = time.time()
        dt = current_time - prev_time
        
        # FPS = 1 / time per frame
        # The check dt > 0 avoids division by zero
        fps = 1.0 / dt if dt > 0 else 0.0

        prev_time = current_time

        # 4) Draw annotations
        frame = draw_detections(frame=frame, detections=detections)
        frame = draw_counts_panel(frame=frame, counts=counts,fps=fps)

        # 5) Create writer after frame size is known
        if SAVE_OUTPUT_VIDEO and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                filename=OUTPUT_VIDEO_PATH,
                fourcc=fourcc,
                fps=20,
                frameSize=(w,h)
            )

        # 6) Save frame to video file
        if writer is not None:
            writer.write(frame)

        # 7) Show live window
        if SHOW_WINDOW:
            cv2.imshow("ATLC Phase 1 - Detection", frame)

            # Press 'q' to quit:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # 8) Print structured info to console
        print({
            "counts":counts,
            "num_detections": len(detections),
            "fps": round(fps, 2),
        })

    # Cleanup
    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()