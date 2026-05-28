"""
Generate report-ready PNG evidence for ATLC Phase 11.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_research.src.yolo_utils.plotting import (
    copy_ultralytics_figures,
    plot_bbox_size_distribution,
    plot_class_distribution,
    plot_density_over_time,
    plot_green_time_over_time,
    plot_images_per_split,
    plot_metrics_summary_bar,
    plot_pipeline_runtime_breakdown,
    plot_prediction_samples_grid,
    plot_sample_labels_grid,
    plot_signal_state_timeline,
    plot_train_loss_curve,
    plot_uart_ack_latency,
    plot_uart_success_rate,
    write_plot_metadata_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready PNG plots for ATLC YOLO research.")
    parser.add_argument("--data", default="yolo_research/configs/data.yaml", help="Path to YOLO data.yaml.")
    parser.add_argument("--run-dir", default="yolo_research/outputs/runs/atlc_yolo26n_custom", help="Ultralytics training run directory.")
    parser.add_argument("--predictions-dir", default="yolo_research/outputs/predictions/atlc_yolo26n_custom_predictions", help="Prediction output directory containing predicted images.")
    parser.add_argument("--outdir", default="yolo_research/outputs/figures", help="Output directory for PNG figures.")
    parser.add_argument("--pipeline-log", default="outputs/logs/pipeline_log.csv", help="Optional ATLC pipeline log CSV.")
    parser.add_argument("--profile-log", default="outputs/logs/profile_log.csv", help="Optional ATLC profiling log CSV.")
    parser.add_argument("--uart-log", default="outputs/logs/uart_log.csv", help="Optional UART log CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    run_dir = Path(args.run_dir)
    predictions_dir = Path(args.predictions_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLC PHASE 11 PLOT GENERATION")
    print("=" * 80)
    print(f"data: {data_yaml}")
    print(f"run_dir: {run_dir}")
    print(f"predictions_dir: {predictions_dir}")
    print(f"outdir: {outdir}")
    print("=" * 80)

    print("\nA. Dataset and model evidence")
    plot_images_per_split(data_yaml, outdir)
    plot_class_distribution(data_yaml, outdir)
    plot_bbox_size_distribution(data_yaml, outdir)
    plot_sample_labels_grid(data_yaml, outdir, split="train", max_images=9)
    plot_train_loss_curve(run_dir, outdir)
    plot_metrics_summary_bar(run_dir, outdir)
    copy_ultralytics_figures(run_dir, outdir)
    plot_prediction_samples_grid(predictions_dir, outdir, max_images=9)

    print("\nB. Embedded system / control evidence")
    plot_density_over_time(args.pipeline_log, outdir)
    plot_green_time_over_time(args.pipeline_log, outdir)
    plot_signal_state_timeline(args.pipeline_log, outdir)
    plot_pipeline_runtime_breakdown(args.profile_log, outdir)
    plot_uart_ack_latency(args.uart_log, outdir)
    plot_uart_success_rate(args.uart_log, outdir)

    print("\nC. Plot metadata table")
    write_plot_metadata_table(outdir)

    print("=" * 80)
    print("PLOT GENERATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()
