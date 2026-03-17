''' 
classes.py

Purpose:
- Normalize raw labels coming from the detector.
- Decide whether a detected class is allowed in this project.
'''

from typing import Optional

''' Convert a raw class name from the detector into the project's canonical class name.
    Input:
        raw_name: original label from model
        mapping: dictionary that defines how to normalize labels

    Output:
        normalized class name or None if the label is empty or unsupported
'''
def normalize_class_name(raw_name: str, mapping: dict) -> Optional[str]:
    # If raw_name is empty string, None or invalid
    if not raw_name: 
        return None
    
    '''
    Return mapped class if known, else None.
    We only want classes explicitly supported by our project.
    '''
    # strip(): remove extra spaces like " car " -> "car"
    # lower() : make matching case-insensitive
    key = raw_name.strip().lower()

    # We return None instead of key itself
    return mapping.get(key, None)


def is_allowed_class(raw_name: str, mapping: dict) -> bool:
    if not raw_name: 
        return False
    key = raw_name.strip().lower()
    return key in mapping