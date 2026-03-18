'''
counter.py : decide how many detections of each class exits

Purpose:
    - Count detections by class.
    - Build compact summaries for each road direction.
'''

from collections import Counter
from typing import List, Dict, Any

# Count how many detections belong to each class. 
def count_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for det in detections:
        counts[det["class_name"]] += 1

    '''
    Convert Counter object to a normal dictionary 

    Example Ouput:
    {
        "car": 4,
        "motorbike": 7
    }
    '''
    return dict(counts)

# Return the total number of detections in a list
def count_total_objects(detections: List[Dict[str, Any]]) -> int :
    return len(detections)

# Build a per-direction count summary
def build_direction_counts(
        detection_a: List[Dict[str, Any]],
        detection_b: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    return{
        "A": count_by_class(detection_a),
        "B": count_by_class(detection_b),
    }