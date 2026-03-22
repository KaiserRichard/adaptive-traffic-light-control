'''
test_scheduler.py

Purpose:
    - Verify scheduler logic independently from live detection, ROI logic and density computation.
'''

from pc_app.config import (
    BASE_GREEN_TIME,
    MAX_GREEN_TIME,
    MIN_GREEN_TIME,
    YELLO_TIME,
    ALL_RED_TIME,
    DENSITY_EPSILON,
)

from pc_app.control.scheduler import(
    compute_density_ratio,
    build_signal_plan
)

def main():
    density_a = 12.0
    density_b = 5.0

    ratio = compute_density_ratio(density_a, density_b, epsilon=DENSITY_EPSILON)
    print("Ratio: ", ratio )

    signal_plan = build_signal_plan(density_a, density_b, DENSITY_EPSILON, BASE_GREEN_TIME, MIN_GREEN_TIME, MAX_GREEN_TIME,YELLO_TIME, ALL_RED_TIME)
    print("Signal: ", signal_plan)


if __name__ == "__main__":
    main()