'''
test_counter.py

Purpose:
    - Verify counting logic independently from detection and ROI logic.
'''

from pc_app.vision.counter import count_by_class, build_direction_counts

def main():
    detections_a = [
        {"class_name": "car"},
        {"class_name": "car"},
        {"class_name": "motorbike"},
    ]

    detections_b = [
        {"class_name": "bus"},
        {"class_name": "truck"},
        {"class_name": "car"},
    ]

    print("A counts:", count_by_class(detections_a))
    print("B counts:", count_by_class(detections_b))
    print("Direction counts:", build_direction_counts(detections_a, detections_b))

if __name__ == "__main__":
    main()