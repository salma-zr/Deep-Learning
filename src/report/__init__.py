"""Report generation module."""

from src.report.tables import generate_results_table, generate_ablation_table
from src.report.figures import (
    plot_metrics_comparison,
    plot_ablation_curve,
    plot_latency_distribution,
)
from src.report.latex import build_report

__all__ = [
    "generate_results_table",
    "generate_ablation_table",
    "plot_metrics_comparison",
    "plot_ablation_curve",
    "plot_latency_distribution",
    "build_report",
]
