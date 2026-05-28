"""
plot_runtime_metrics.py

ATLC Phase 13 — Runtime Metrics Plot Generation

Purpose:
    - Read ATLC runtime CSV logs.
    - Generate report-ready PNG plots.
    - Summarize runtime behavior into markdown.

Input:
    outputs/pipeline_runs/yolo26n_runtime_baseline/logs/traffic_runtime_log.csv

Output:
    outputs/figures/phase13_runtime_metrics/
    outputs/reports/phase13_runtime_metrics_summary.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline_runs"
    / "yolo26n_runtime_baseline"
    / "logs"
    / "traffic_runtime_log.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "phase13_runtime_metrics"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

SUMMARY_MD = REPORT_DIR / "phase13_runtime_metrics_summary.md"


LEVEL_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

PHASE_MAP = {
    "A_GREEN": 0,
    "A_YELLOW": 1,
    "ALL_RED_AFTER_A": 2,
    "B_GREEN": 3,
    "B_YELLOW": 4,
    "ALL_RED_AFTER_B": 5,
}


def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_runtime_log(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Runtime log not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "frame_index" not in df.columns:
        df["frame_index"] = range(1, len(df) + 1)

    return df


def save_line_plot(
    df: pd.DataFrame,
    y_columns: list[str],
    title: str,
    ylabel: str,
    output_name: str,
) -> None:

    available_columns = [
        col for col in y_columns
        if col in df.columns
    ]

    if not available_columns:
        print(f"[SKIP] Missing columns for {output_name}")
        return

    plt.figure(figsize=(10, 5))

    for col in available_columns:
        plt.plot(
            df["frame_index"],
            df[col],
            label=col,
        )

    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURE_DIR / output_name

    plt.savefig(output_path, dpi=160)
    plt.close()

    print("Saved:", output_path)


def plot_processing_fps(df: pd.DataFrame) -> None:
    save_line_plot(
        df=df,
        y_columns=["processing_fps"],
        title="ATLC Runtime FPS Over Time",
        ylabel="FPS",
        output_name="runtime_fps.png",
    )


def plot_direction_counts(df: pd.DataFrame) -> None:
    save_line_plot(
        df=df,
        y_columns=[
            "direction_a_count",
            "direction_b_count",
        ],
        title="Vehicle Count per Direction",
        ylabel="Vehicle count",
        output_name="direction_counts.png",
    )


def plot_occupancies(df: pd.DataFrame) -> None:
    save_line_plot(
        df=df,
        y_columns=[
            "direction_a_occupancy",
            "direction_b_occupancy",
        ],
        title="Occupancy Proxy per Direction",
        ylabel="Occupancy",
        output_name="occupancy_proxy.png",
    )


def plot_green_times(df: pd.DataFrame) -> None:
    save_line_plot(
        df=df,
        y_columns=["green_a", "green_b"],
        title="Adaptive Green Times",
        ylabel="Seconds",
        output_name="green_times.png",
    )


def plot_traffic_levels(df: pd.DataFrame) -> None:

    required = [
        "direction_a_level",
        "direction_b_level",
    ]

    if not all(col in df.columns for col in required):
        print("[SKIP] Missing traffic level columns.")
        return

    plot_df = df.copy()

    plot_df["direction_a_level_code"] = (
        plot_df["direction_a_level"].map(LEVEL_MAP)
    )

    plot_df["direction_b_level_code"] = (
        plot_df["direction_b_level"].map(LEVEL_MAP)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        plot_df["frame_index"],
        plot_df["direction_a_level_code"],
        label="direction_a_level",
    )

    plt.plot(
        plot_df["frame_index"],
        plot_df["direction_b_level_code"],
        label="direction_b_level",
    )

    plt.yticks(
        [0, 1, 2],
        ["LOW", "MEDIUM", "HIGH"],
    )

    plt.title("Traffic Level Timeline")
    plt.xlabel("Frame index")
    plt.ylabel("Traffic level")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURE_DIR / "traffic_levels.png"

    plt.savefig(output_path, dpi=160)
    plt.close()

    print("Saved:", output_path)


def plot_runtime_phase_timeline(df: pd.DataFrame) -> None:

    if "phase" not in df.columns:
        print("[SKIP] Missing phase column.")
        return

    plot_df = df.copy()

    plot_df["phase_code"] = (
        plot_df["phase"].map(PHASE_MAP)
    )

    plt.figure(figsize=(12, 5))

    plt.step(
        plot_df["frame_index"],
        plot_df["phase_code"],
        where="post",
    )

    plt.yticks(
        list(PHASE_MAP.values()),
        list(PHASE_MAP.keys()),
    )

    plt.title("Runtime Phase Timeline")
    plt.xlabel("Frame index")
    plt.ylabel("Phase")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURE_DIR / "runtime_phase_timeline.png"

    plt.savefig(output_path, dpi=160)
    plt.close()

    print("Saved:", output_path)

def plot_raw_vs_active_green_times(df: pd.DataFrame) -> None:

    required = [
        "raw_green_a",
        "raw_green_b",
        "active_green_a",
        "active_green_b",
    ]

    if not all(col in df.columns for col in required):
        print("[SKIP] Missing raw/active green-time columns.")
        return

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["frame_index"],
        df["raw_green_a"],
        label="raw_green_a",
        linestyle="--",
    )

    plt.plot(
        df["frame_index"],
        df["active_green_a"],
        label="active_green_a",
    )

    plt.plot(
        df["frame_index"],
        df["raw_green_b"],
        label="raw_green_b",
        linestyle="--",
    )

    plt.plot(
        df["frame_index"],
        df["active_green_b"],
        label="active_green_b",
    )

    plt.title("Raw Signal Plan vs Active Runtime Plan")
    plt.xlabel("Frame index")
    plt.ylabel("Green time (seconds)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = (
        FIGURE_DIR /
        "raw_signal_plan_vs_active_plan.png"
    )

    plt.savefig(output_path, dpi=160)
    plt.close()

    print("Saved:", output_path)



def write_summary(df: pd.DataFrame) -> None:

    rows = len(df)

    avg_fps = (
        df["processing_fps"].mean()
        if "processing_fps" in df.columns
        else 0.0
    )

    summary = f"""
# ATLC Phase 13 Runtime Metrics Summary

## Runtime Samples

Total runtime samples:
{rows}

## Average Runtime FPS

{avg_fps:.2f}

## Generated Figures

- runtime_fps.png
- direction_counts.png
- occupancy_proxy.png
- green_times.png
- traffic_levels.png
- runtime_phase_timeline.png
"""

    SUMMARY_MD.write_text(
        summary,
        encoding="utf-8",
    )

    print("Saved summary:", SUMMARY_MD)


def main() -> None:

    ensure_output_dirs()

    df = load_runtime_log(DEFAULT_LOG_CSV)

    plot_processing_fps(df)
    plot_direction_counts(df)
    plot_occupancies(df)
    plot_green_times(df)
    plot_traffic_levels(df)
    plot_runtime_phase_timeline(df)
    plot_raw_vs_active_green_times(df)

    write_summary(df)

    print("Phase 13 plotting complete.")


if __name__ == "__main__":
    main()