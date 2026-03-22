'''
scheduler.py

Purpose:
    - Convert direction-specific traffic density into traffic signal
    - Keep timing logic seperate 
'''

from typing import Dict

# Clamp a numeric value into the range [min_value, max_value] and return it as an integer
def clamp(value: float, min_value: int, max_value: int):
    return max(min_value, min(int(round(value)), max_value))

# Compute normalized density ratios for Direction A and B
def compute_density_ratio(
        density_a: float,
        density_b: float,
        epsilon: float
) -> Dict[str, float]:
    total = density_a + density_b + epsilon
    return{
        "ratio_a": density_a / total,
        "ratio_b": density_b / total,
    }

'''
Allocate green times using direction density and then build a full signal plan

Strategy:
    - Compute normalized density ratios
    - Use 2 * base_green as the total budget
    - Split that budget proportionally
    - Clamp each result into [min_green, max_green]

Output:
{
        "green_a": 24,
        "green_b": 16,
        "yellow": 3,
        "all_red": 1,
        "ratio_a": 0.60,
        "ratio_b": 0.40
    }
'''

def build_signal_plan(
        density_a: float,
        density_b: float,
        epsilon: float,
        base_green_time: int,
        min_green_time: int,
        max_green_time: int,
        yellow_time: int,
        all_red_time: int,
)-> Dict[str, int|float] :
    ratios = compute_density_ratio(density_a, density_b, epsilon)
    ratio_a = ratios["ratio_a"]
    ratio_b = ratios["ratio_b"]

    total_gren_budget = 2 * base_green_time
    green_a = clamp(total_gren_budget * ratio_a, min_green_time, max_green_time)
    green_b = clamp(total_gren_budget * ratio_b, min_green_time, max_green_time)

    return{
        "green_a": green_a,
        "green_b": green_b,
        "yellow": yellow_time,
        "all_red": all_red_time,
        "ratio_a": ratio_a,
        "ratio_b": ratio_b,
    }
