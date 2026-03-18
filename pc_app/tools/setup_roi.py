'''
setup_tool.py
Purpose:
    - Interactive ROI setup tool for Phase 2.
    - Allows the user to click polygon points for Direction A and B
    - Save the ROI configuration to JSON

Controls:
    - Left mouse click: add a point
    - n: finish the current ROI and move to the next one
    - s: Save ROI JSON and exit
    - q: quit without saving

Workflow:
    1) Draw ROI A (>= 3 points), press 'n'
    2) Draw ROI B (>= 3 points), press 'n'
    3) Press 's' to save configuration

Output: Saves a JSON file at ROI_CONFIG_PATH with format:
{
    "direction_a_name": "A",
    "direction_b_name": "B",
    "roi_a": [[x1, y1], ...],
    "roi_b": [[x1, y1], ...]
}

'''

import json
import cv2 
import numpy as np
from pc_app.config import VIDEO_SOURCE, ROI_CONFIG_PATH

current_points = []
roi_a = []
roi_b = []
stage = "A" # Controls which ROI the user is defining

def mouse_callback(event, x, y, flags, param):
    global current_points 
    if event == cv2.EVENT_LBUTTONDOWN:
        # Points are stored in order of clicking -> Polygon shape
        print("CLICK: ", x, y)
        current_points.append((x, y))

def main():
    global current_points, roi_a, roi_b, stage

    # Load first frame (ROI is defined relative to static scene)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")
    
    ret, frame = cap.read()
    cap.release()

    if not ret: 
        raise RuntimeError("Failed to read first frame.")
    
    # Base frame remains unchanegd, we draw overlays on copies
    base_frame = frame.copy()
    
    # Create UI window and bind mouse interaction
    cv2.namedWindow("ROI Setup")
    cv2.setMouseCallback("ROI Setup", mouse_callback)

    # ------ MAIN UI Loop ------
    while True:
        # Reset drawing canvas each frame to avoid accumulation artifacts
        display = base_frame.copy()

        # Draw currently selected points (in-progress polygon)
        for pt in current_points:
            cv2.circle(
                img=display, 
                center=pt, 
                radius=4,
                color=(0,255,255), # Yello point
                thickness=-1
            )

        # Draw open polygon while user is still selecting points
        if len(current_points) > 1:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(
                img=display,
                pts= [pts],
                isClosed=False,
                color=(0, 255, 255),
                thickness=1
            )

        # Draw completed ROI A and ROI B
        if roi_a:
            pts_a = np.array(roi_a, dtype=np.int32)
            cv2.polylines(display, [pts_a], True, (0, 255, 0), 2)
            # Label anchored at first vertex
            cv2.putText(
                img=display,
                text="ROI A",
                org=roi_a[0],
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=2
            )
        if roi_b:
            pts_b = np.array(roi_b, dtype=np.int32)
            cv2.polylines(display, [pts_b], True, (255, 0, 0), 2) # Different color for ROI B
            # Label anchored at first vertex
            cv2.putText(
                img=display,
                text="ROI B",
                org=roi_b[0],
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 0, 0),
                thickness=2
            )

        # UI Instructions Overlay
        message = f"Drawing ROI {stage} | click: add point | n: next | s: save | q: quit."
        cv2.putText(display, message, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2 )

        # Show frame
        cv2.imshow(
            winname="ROI Setup",
            mat=display
        )    

        # Capture keyboard input
        key = cv2.waitKey(1) & 0xFF

        # If "DONE" user cannot modify anything anymore
        if stage == "DONE":
            # Allow saving
            if key == ord('s'):
                pass 
            # Quit
            elif key == ord('q'):
                break
            else:  
                continue

        # Key: 'n' -> finalize current ROI
        if key == ord('n'):

            # ROI A completion
            if stage == "A" and len(current_points) >= 3:
                roi_a = current_points.copy() # Store finalized polygon
                current_points = [] # Reset buffer
                stage = "B"

            elif stage == "B" and len(current_points) >= 3:
                roi_b = current_points.copy()
                current_points = []
                stage = "DONE"

        # Key 's' -> save configuration
        elif key == ord('s'):

            # Ensure both ROIs are defined before saving
            if roi_a and roi_b:
                data = {
                    "direction_a_name": "A",
                    "direction_b_name": "B",
                    "roi_a": roi_a,
                    "roi_b": roi_b,
                }
                
                with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
                    # json.dump: writes to a file != json.dumps() returns a string
                    json.dump(
                        obj=data,
                        fp=f, # Write JSON data into file f
                        indent=2, # JSON file format: 2 spaces per level
                    )

                print(f"Save ROI config to: {ROI_CONFIG_PATH}.")
                break

            else:
                print("Both ROI A and ROI B must be defined before saving.")

        # Key: 'q' -> quit withoug saving
        elif key == ord("q"):
            break

        
    # Cleanup OpenCV resources
    cv2.destroyAllWindows()

# Entry point

if __name__ == "__main__":
    main()

