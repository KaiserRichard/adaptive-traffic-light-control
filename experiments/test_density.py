''''
test_density.py

Purpose: 
    - Verify raw density and PCE-weighted density independently from live detection and ROI logic.
    - Verify EMA smoothing behaviour
'''

from pc_app.config import PCE_WEIGHTS, DENSITY_SMOOTHING_ALPHA
from pc_app.vision.density import (
    compute_raw_density,
    compute_pce_density,
    build_density_summary,
    EMASmoother,
)

def main():
    detections_a = [
        {"class_name": "car"},
        {"class_name": "car"},
        {"class_name": "motorbike"},
        {"class_name": "bus"},
    ]

    detections_b = [
        {"class_name": "motorbike"},
        {"class_name": "motorbike"},
        {"class_name": "truck"},
        {"class_name": "rickshaw"},
    ]

    raw_a = compute_raw_density(detections_a)
    raw_b = compute_raw_density(detections_b)

    pce_a = compute_pce_density(detections_a, PCE_WEIGHTS)
    pce_b = compute_pce_density(detections_b, PCE_WEIGHTS)

    print("Raw density A:", raw_a)
    print("Raw density B:", raw_b)
    print("PCE density A:", pce_a)
    print("PCE density B:", pce_b)

    summary = build_density_summary(detections_a, detections_b, PCE_WEIGHTS)
    print("Density summary:", summary)
    
    smoother = EMASmoother(alpha=DENSITY_SMOOTHING_ALPHA)
    sample_series = [5.0, 8.0, 6.0, 10.0]

    print("EMA smoothing demo: ")
    for value in sample_series: 
        smoothed = smoother.update(value)        
        print(f"input={value:.2f}, smoothed={smoothed:.2f}")

    
if __name__ == "__main__":
    main()