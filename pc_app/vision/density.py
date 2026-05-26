'''
density.py

Purpose: 
    - Compute per-direction traffic density.
    - Support both raw count density and PCE-weighted density.
    - Smooth density values over time using EMA. 
'''

from typing import List, Dict, Tuple, Any, Optional, Sequence

# Raw density baseline: Simple the number of detctions in the direction.
def compute_raw_density(detections: List[Dict[str, Any]]) -> float :
    return float(len(detections))

# Compute PCE-weighted density.
def compute_pce_density(
        detections: List[Dict[str, Any]],
        pce_weights: Dict[str, float]
) -> float :

    # Reuse count_by_class()
    from pc_app.vision.counter import count_by_class

    counts = count_by_class(detections)
    total = 0.0

    for class_name, count in counts.items():
        weight = pce_weights.get(class_name, 0.0)
        total += count * weight
    
    return total

class EMASmoother:
    '''
    Exponential Moving Average smoother for numeric values.

    Fomula: 
        smoothed = alpha * current + (1 - alpha) * previous

    Purpose: 
        - Detection counts fluctuate frame to frame
        - Smoother gives more stable density signals
        - Later improves timing stability in Phase 4
    '''
    
    def __init__(self, alpha: float) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Alpha must be in the range (0, 1).")
        
        self.alpha = alpha
        self._value: Optional[float] = None

    # Update the EMA state and return the smoothed value.
    def update(self, current_value: float) -> float:
        if self._value is None:
            self._value = float(current_value)

        else:
            self._value = (
                self.alpha * float(current_value) + (1.0 - self.alpha) * self._value
            )

        return self._value
        
# Build unsmoothed density summary for both directions.
def build_density_summary(
        detections_a: List[Dict[str, Any]],
        detections_b: List[Dict[str, Any]],
        pce_weights: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Output format:
    {
        "A": {
            "raw_density": 8.0,
            "pce_density": 6.5
        },
        "B": {
            "raw_density": 5.0,
            "pce_density": 7.0
        }
    }
    """
    return {
        "A":{
            "raw_density": compute_raw_density(detections_a),
            "pce_density": compute_pce_density(detections_a, pce_weights),
        },
        "B":{
            "raw_density": compute_raw_density(detections_b),
            "pce_density": compute_pce_density(detections_b, pce_weights),
        },
    }

# Phase 12: Density-based timing adjustment logic will consume the density summary and smoothed density values to make decisions.

# Optional occupancy and traffic-level helpers.

def calculate_occupancy_from_boxes(
    boxes: List[Sequence[float]],
    roi_area: float,
) -> float:
    """
    Estimate ROI occupancy Oi from bounding boxes that have already been
    filtered by ROI.

    Oi = total bounding-box area / ROI area

    Note:
        This is an approximation. It does not compute exact polygon
        intersection between each bounding box and the ROI polygon.
    """
    if roi_area <= 0:
        return 0.0

    total_box_area = 0.0

    for box in boxes:
        x1, y1, x2, y2 = box

        width = max(0.0, float(x2) - float(x1))
        height = max(0.0, float(y2) - float(y1))

        total_box_area += width * height

    occupancy = total_box_area / roi_area
    return max(0.0, min(occupancy, 1.0))


def calculate_occupancy_from_count(
    count: int,
    saturation_count: int = 20,
) -> float:
    """
    Estimate a normalized occupancy proxy from vehicle count.

    This is not geometric ROI occupancy. It is a fallback signal used when
    box-level ROI area is not available in the current runtime pipeline.

    occupancy_proxy = count / saturation_count
    """
    if saturation_count <= 0:
        return 0.0

    occupancy = float(count) / float(saturation_count)
    return max(0.0, min(occupancy, 1.0))


def classify_traffic_level(
    occupancy: float,
    low_threshold: float = 0.20,
    high_threshold: float = 0.50,
    queue_increasing: bool = False,
) -> str:
    """
    Convert occupancy or occupancy proxy into LOW / MEDIUM / HIGH.
    """
    occupancy = max(0.0, min(float(occupancy), 1.0))

    if occupancy >= high_threshold or queue_increasing:
        return "HIGH"

    if occupancy >= low_threshold:
        return "MEDIUM"

    return "LOW"