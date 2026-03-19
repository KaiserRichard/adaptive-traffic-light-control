'''
density.py

Purpose: 
    - Compute per-direction traffic density.
    - Support both raw count density and PCE-weighted density.
    - Smooth density values over time using EMA. 
'''

from typing import List, Dict, Tuple, Any, Optional

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