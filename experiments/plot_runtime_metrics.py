"""
plot_runtime_metrics.py

ATLC Phase 13 - Runtime Metrics Plot Generation

Purpose:
    - Read ATLC runtime CSV logs.
    - Generate single-run report-ready PNG plots.
    - Generate baseline-vs-custom comparison plots.

Single-run example:
    python -m experiments.plot_runtime_metrics \
      --log outputs/pipeline_runs/yolo26n_custom_runtime/logs/traffic_runtime_log.csv \
      --figures-dir outputs/figures/yolo26n_custom_runtime \
      --summary outputs/reports/yolo26n_custom_runtime_summary.md

Compare example:
    python -m experiments.plot_runtime_metrics \
      --baseline outputs/pipeline_runs/yolo26n_runtime_baseline/logs/traffic_runtime_log.csv \
      --custom outputs/pipeline_runs/yolo26n_custom_runtime/logs/traffic_runtime_log.csv \
      --compare-dir outputs/figures/yolo26n_baseline_vs_custom
"""

import argparse
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

DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "phase13_runtime_metrics"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

DEFAULT_SUMMARY_MD = (
    DEFAULT_REPORT_DIR
    / "phase13_runtime_metrics_summary.md"
)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ATLC runtime metric plots."
    )

    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Single runtime CSV log path.",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline runtime CSV log path.",
    )

    parser.add_argument(
        "--custom",
        type=str,
        default=None,
        help="Custom runtime CSV log path.",
    )

    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Output directory for single-run figures.",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Output markdown summary path.",
    )

    parser.add_argument(
        "--compare-dir",
        type=str,
        default=None,
        help="Output directory for comparison figures.",
    )

    return parser.parse_args()


def ensure_output_dirs(
    figure_dir: Path,
    summary_md: Path,
) -> None:
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_md.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_runtime_log(
    csv_path: Path,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Runtime log not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "frame_index" not in df.columns:
        df["frame_index"] = range(1, len(df) + 1)

    df = df.copy()
    df["sample_index"] = range(1, len(df) + 1)

    return df


def save_line_plot(
    df: pd.DataFrame,
    y_columns: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    x_column: str = "frame_index",
) -> None:
    available_columns = [
        col for col in y_columns
        if col in df.columns
    ]

    if not available_columns:
        print(f"[SKIP] Missing columns for {output_path.name}")
        return

    plt.figure(figsize=(10, 5))

    for col in available_columns:
        plt.plot(
            df[x_column],
            df[col],
            label=col,
            linewidth=2,
        )

    plt.title(title)
    plt.xlabel(
        "Frame index"
        if x_column == "frame_index"
        else "Runtime sample index"
    )
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print("Saved:", output_path)


def plot_processing_fps(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    save_line_plot(
        df=df,
        y_columns=["processing_fps"],
        title="ATLC Runtime FPS Over Time",
        ylabel="FPS",
        output_path=figure_dir / "runtime_fps.png",
    )


def plot_direction_counts(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    save_line_plot(
        df=df,
        y_columns=[
            "direction_a_count",
            "direction_b_count",
        ],
        title="Vehicle Count per Direction",
        ylabel="Vehicle count",
        output_path=figure_dir / "direction_counts.png",
    )


def plot_occupancies(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    save_line_plot(
        df=df,
        y_columns=[
            "direction_a_occupancy",
            "direction_b_occupancy",
        ],
        title="Occupancy Proxy per Direction",
        ylabel="Occupancy",
        output_path=figure_dir / "occupancy_proxy.png",
    )


def plot_green_times(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    save_line_plot(
        df=df,
        y_columns=[
            "green_a",
            "green_b",
        ],
        title="Adaptive Green Times",
        ylabel="Seconds",
        output_path=figure_dir / "green_times.png",
    )


def plot_traffic_levels(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
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
        linewidth=2,
    )

    plt.plot(
        plot_df["frame_index"],
        plot_df["direction_b_level_code"],
        label="direction_b_level",
        linewidth=2,
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

    output_path = figure_dir / "traffic_levels.png"

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print("Saved:", output_path)


def plot_runtime_phase_timeline(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
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
        linewidth=2,
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

    output_path = figure_dir / "runtime_phase_timeline.png"

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print("Saved:", output_path)


def plot_raw_vs_active_green_times(
    df: pd.DataFrame,
    figure_dir: Path,
) -> None:
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

    plt.step(
        df["frame_index"],
        df["raw_green_a"],
        where="post",
        label="raw_green_a",
        linestyle="--",
        linewidth=2,
    )

    plt.step(
        df["frame_index"],
        df["active_green_a"],
        where="post",
        label="active_green_a",
        linewidth=2,
    )

    plt.step(
        df["frame_index"],
        df["raw_green_b"],
        where="post",
        label="raw_green_b",
        linestyle="--",
        linewidth=2,
    )

    plt.step(
        df["frame_index"],
        df["active_green_b"],
        where="post",
        label="active_green_b",
        linewidth=2,
    )

    plt.title("Raw Signal Plan vs Active Runtime Plan")
    plt.xlabel("Frame index")
    plt.ylabel("Green time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = figure_dir / "raw_signal_plan_vs_active_plan.png"

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print("Saved:", output_path)


def write_summary(
    df: pd.DataFrame,
    summary_md: Path,
) -> None:
    rows = len(df)

    avg_fps = (
        df["processing_fps"].mean()
        if "processing_fps" in df.columns
        else 0.0
    )

    avg_count_a = (
        df["direction_a_count"].mean()
        if "direction_a_count" in df.columns
        else 0.0
    )

    avg_count_b = (
        df["direction_b_count"].mean()
        if "direction_b_count" in df.columns
        else 0.0
    )

    summary = f"""# ATLC Phase 13 Runtime Metrics Summary

## Runtime Samples

Total runtime samples: {rows}

## Average Runtime FPS

{avg_fps:.2f}

## Average Vehicle Counts

Direction A average count: {avg_count_a:.2f}

Direction B average count: {avg_count_b:.2f}

## Generated Figures

- runtime_fps.png
- direction_counts.png
- occupancy_proxy.png
- green_times.png
- traffic_levels.png
- runtime_phase_timeline.png
- raw_signal_plan_vs_active_plan.png
"""

    summary_md.write_text(
        summary,
        encoding="utf-8",
    )

    print("Saved summary:", summary_md)


def run_single_mode(
    log_csv: Path,
    figure_dir: Path,
    summary_md: Path,
) -> None:
    ensure_output_dirs(
        figure_dir=figure_dir,
        summary_md=summary_md,
    )

    df = load_runtime_log(log_csv)

    plot_processing_fps(
        df=df,
        figure_dir=figure_dir,
    )

    plot_direction_counts(
        df=df,
        figure_dir=figure_dir,
    )

    plot_occupancies(
        df=df,
        figure_dir=figure_dir,
    )

    plot_green_times(
        df=df,
        figure_dir=figure_dir,
    )

    plot_traffic_levels(
        df=df,
        figure_dir=figure_dir,
    )

    plot_runtime_phase_timeline(
        df=df,
        figure_dir=figure_dir,
    )

    plot_raw_vs_active_green_times(
        df=df,
        figure_dir=figure_dir,
    )

    write_summary(
        df=df,
        summary_md=summary_md,
    )

    print("Single runtime plotting complete.")


def plot_compare_metric(
    baseline_df: pd.DataFrame,
    custom_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    use_step: bool = False,
) -> None:
    if column not in baseline_df.columns:
        print(f"[SKIP] Missing baseline column: {column}")
        return

    if column not in custom_df.columns:
        print(f"[SKIP] Missing custom column: {column}")
        return

    plt.figure(figsize=(12, 6))

    plot_func = plt.step if use_step else plt.plot

    if use_step:
        plot_func(
            baseline_df["sample_index"],
            baseline_df[column],
            where="post",
            label=f"baseline_{column}",
            linewidth=2,
        )

        plot_func(
            custom_df["sample_index"],
            custom_df[column],
            where="post",
            label=f"custom_{column}",
            linewidth=2,
        )
    else:
        plot_func(
            baseline_df["sample_index"],
            baseline_df[column],
            label=f"baseline_{column}",
            linewidth=2,
        )

        plot_func(
            custom_df["sample_index"],
            custom_df[column],
            label=f"custom_{column}",
            linewidth=2,
        )

    plt.title(title)
    plt.xlabel("Runtime sample index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print("Saved:", output_path)


def write_compare_summary(
    baseline_df: pd.DataFrame,
    custom_df: pd.DataFrame,
    compare_dir: Path,
) -> None:
    def mean_or_zero(
        df: pd.DataFrame,
        column: str,
    ) -> float:
        if column not in df.columns:
            return 0.0

        return float(df[column].mean())

    summary_path = compare_dir / "comparison_summary.md"

    summary = f"""# ATLC YOLO26n Baseline vs Custom Runtime Comparison

## Runtime Samples

Baseline samples: {len(baseline_df)}

Custom samples: {len(custom_df)}

## Average FPS

Baseline average FPS: {mean_or_zero(baseline_df, "processing_fps"):.2f}

Custom average FPS: {mean_or_zero(custom_df, "processing_fps"):.2f}

## Average Direction Counts

Baseline Direction A average count: {mean_or_zero(baseline_df, "direction_a_count"):.2f}

Custom Direction A average count: {mean_or_zero(custom_df, "direction_a_count"):.2f}

Baseline Direction B average count: {mean_or_zero(baseline_df, "direction_b_count"):.2f}

Custom Direction B average count: {mean_or_zero(custom_df, "direction_b_count"):.2f}

## Generated Comparison Figures

- compare_fps.png
- compare_direction_a_count.png
- compare_direction_b_count.png
- compare_raw_green_a.png
- compare_raw_green_b.png
- compare_active_green_a.png
- compare_active_green_b.png
"""

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print("Saved summary:", summary_path)


def run_compare_mode(
    baseline_csv: Path,
    custom_csv: Path,
    compare_dir: Path,
) -> None:
    baseline_df = load_runtime_log(baseline_csv)
    custom_df = load_runtime_log(custom_csv)

    compare_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="processing_fps",
        title="Baseline vs Custom Runtime FPS",
        ylabel="FPS",
        output_path=compare_dir / "compare_fps.png",
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="direction_a_count",
        title="Baseline vs Custom Direction A Count",
        ylabel="Vehicle count",
        output_path=compare_dir / "compare_direction_a_count.png",
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="direction_b_count",
        title="Baseline vs Custom Direction B Count",
        ylabel="Vehicle count",
        output_path=compare_dir / "compare_direction_b_count.png",
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="raw_green_a",
        title="Baseline vs Custom Raw Green A",
        ylabel="Green time (seconds)",
        output_path=compare_dir / "compare_raw_green_a.png",
        use_step=True,
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="raw_green_b",
        title="Baseline vs Custom Raw Green B",
        ylabel="Green time (seconds)",
        output_path=compare_dir / "compare_raw_green_b.png",
        use_step=True,
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="active_green_a",
        title="Baseline vs Custom Active Green A",
        ylabel="Green time (seconds)",
        output_path=compare_dir / "compare_active_green_a.png",
        use_step=True,
    )

    plot_compare_metric(
        baseline_df=baseline_df,
        custom_df=custom_df,
        column="active_green_b",
        title="Baseline vs Custom Active Green B",
        ylabel="Green time (seconds)",
        output_path=compare_dir / "compare_active_green_b.png",
        use_step=True,
    )

    write_compare_summary(
        baseline_df=baseline_df,
        custom_df=custom_df,
        compare_dir=compare_dir,
    )

    print("Runtime comparison plotting complete.")
    print("Comparison figures:", compare_dir)


def main() -> None:
    args = parse_args()

    if args.baseline and args.custom:
        compare_dir = (
            Path(args.compare_dir)
            if args.compare_dir
            else PROJECT_ROOT
            / "outputs"
            / "figures"
            / "runtime_comparisons"
        )

        run_compare_mode(
            baseline_csv=Path(args.baseline),
            custom_csv=Path(args.custom),
            compare_dir=compare_dir,
        )

        return

    log_csv = (
        Path(args.log)
        if args.log
        else DEFAULT_LOG_CSV
    )

    figure_dir = (
        Path(args.figures_dir)
        if args.figures_dir
        else DEFAULT_FIGURE_DIR
    )

    summary_md = (
        Path(args.summary)
        if args.summary
        else DEFAULT_SUMMARY_MD
    )

    run_single_mode(
        log_csv=log_csv,
        figure_dir=figure_dir,
        summary_md=summary_md,
    )


if __name__ == "__main__":
    main()
