"""Generate figures for the report."""

import re
from pathlib import Path
from typing import Optional
import pandas as pd

from src.utils.io_utils import ensure_dir, get_project_root, load_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def _load_scores(scores_dir: Path) -> pd.DataFrame:
    """Load all score CSVs into a DataFrame."""
    rows = []
    for csv_file in scores_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = csv_file.stem
                rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


def plot_metrics_comparison(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    metrics: list[str] = ["rougeL", "bleu", "judge_mean"],
) -> str:
    """
    Create bar chart comparing metrics across experiments.
    
    Args:
        scores_dir: Directory containing score CSV files
        output_file: Output image path
        metrics: List of metrics to compare
        
    Returns:
        Path to saved figure
    """
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    
    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / "metrics_comparison.png"
    
    scores_dir = Path(scores_dir)
    output_file = Path(output_file)
    ensure_dir(output_file)
    
    results_df = _load_scores(scores_dir)
    if results_df.empty:
        logger.warning("No results to plot")
        _create_placeholder_figure(output_file, "No results available")
        return str(output_file)
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        _create_placeholder_figure(output_file, "Metrics not available")
        return str(output_file)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(results_df))
    width = 0.8 / len(available_metrics)
    max_metric_value = 0.0
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        values = pd.to_numeric(results_df[metric], errors="coerce").fillna(0)
        max_metric_value = max(max_metric_value, float(values.max()))
        ax.bar([xi + offset for xi in x], values, width, label=metric)
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["experiment"], rotation=45, ha="right")
    ax.legend()
    y_upper = 1.1 if max_metric_value <= 1.0 else max(1.1, max_metric_value * 1.15)
    ax.set_ylim(0, y_upper)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Metrics comparison plot saved to {output_file}")
    return str(output_file)


def plot_ablation_curve(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    prefix: str = "finetune",
) -> str:
    """
    Plot ablation curve (e.g., performance vs training data size).
    
    Args:
        scores_dir: Directory containing score CSV files
        output_file: Output image path
        prefix: Prefix to filter experiments
        
    Returns:
        Path to saved figure
    """
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    
    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / "ablation_curve.png"
    
    scores_dir = Path(scores_dir)
    output_file = Path(output_file)
    ensure_dir(output_file)
    
    all_scores = _load_scores(scores_dir)
    if all_scores.empty:
        _create_placeholder_figure(output_file, "No ablation data available")
        return str(output_file)

    # 1) Try fine-tuning ablation first
    ft_rows = []
    for _, row in all_scores.iterrows():
        name = str(row.get("experiment", ""))
        if not name.startswith(prefix):
            continue
        match = re.search(r"(\d+)k?", name.lower())
        if not match:
            continue
        n = int(match.group(1))
        size = n * 1000 if "k" in name.lower() else n
        item = dict(row)
        item["x"] = size
        ft_rows.append(item)

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["rougeL", "judge_mean"]
    colors = ["#2ecc71", "#3498db"]
    max_metric_value = 0.0

    if ft_rows:
        results_df = pd.DataFrame(ft_rows).sort_values("x")
        for metric, color in zip(metrics, colors):
            if metric in results_df.columns:
                values = pd.to_numeric(results_df[metric], errors="coerce")
                max_metric_value = max(max_metric_value, float(values.max()))
                ax.plot(
                    results_df["x"],
                    values,
                    marker="o",
                    label=metric,
                    color=color,
                    linewidth=2,
                    markersize=8,
                )
        ax.set_xlabel("Training Examples")
        ax.set_title("Effect of Training Data Size on Performance")
        if results_df["x"].max() / max(results_df["x"].min(), 1) > 10:
            ax.set_xscale("log")
    else:
        # 2) Fallback to RAG top-k ablation
        rag_rows = []
        for _, row in all_scores.iterrows():
            name = str(row.get("experiment", "")).lower()
            if "rag" not in name:
                continue
            top_k = None
            m = re.search(r"topk[_-]?(\d+)", name)
            if m:
                top_k = int(m.group(1))
            elif "wikipedia" in name or "web" in name:
                top_k = 3
            if top_k is None:
                continue
            item = dict(row)
            item["x"] = top_k
            rag_rows.append(item)

        if not rag_rows:
            plt.close(fig)
            _create_placeholder_figure(output_file, "No ablation data available")
            return str(output_file)

        results_df = pd.DataFrame(rag_rows).sort_values("x")
        for metric, color in zip(metrics, colors):
            if metric in results_df.columns:
                values = pd.to_numeric(results_df[metric], errors="coerce")
                max_metric_value = max(max_metric_value, float(values.max()))
                ax.plot(
                    results_df["x"],
                    values,
                    marker="o",
                    label=metric,
                    color=color,
                    linewidth=2,
                    markersize=8,
                )
        ax.set_xlabel("Retrieved documents (top-k)")
        ax.set_title("Effect of Retrieval Top-k on RAG Performance")
        ax.set_xticks(sorted(results_df["x"].unique()))

    ax.set_ylabel("Score")
    ax.legend()
    y_upper = 1.05 if max_metric_value <= 1.0 else max(1.05, max_metric_value * 1.15)
    ax.set_ylim(0, y_upper)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Ablation curve saved to {output_file}")
    return str(output_file)


def plot_latency_distribution(
    predictions_file: str | Path,
    output_file: Optional[str | Path] = None,
) -> str:
    """
    Plot latency distribution for an experiment.
    
    Args:
        predictions_file: Path to predictions JSONL
        output_file: Output image path
        
    Returns:
        Path to saved figure
    """
    predictions_file = Path(predictions_file)
    
    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / f"{predictions_file.stem}_latency.png"
    
    output_file = Path(output_file)
    ensure_dir(output_file)
    
    # Load predictions
    predictions = load_jsonl(predictions_file)
    
    # Extract latencies
    latencies = [
        p.get("meta", {}).get("latency_ms")
        for p in predictions
        if p.get("meta", {}).get("latency_ms") is not None
    ]
    
    if not latencies:
        _create_placeholder_figure(output_file, "No latency data available")
        return str(output_file)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(latencies, bins=30, edgecolor="white", alpha=0.7)
    axes[0].axvline(sum(latencies)/len(latencies), color="red", linestyle="--", label=f"Mean: {sum(latencies)/len(latencies):.0f}ms")
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Latency Distribution")
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(latencies, vert=True)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency Box Plot")
    
    plt.suptitle(f"Latency Analysis: {predictions_file.stem}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Latency distribution saved to {output_file}")
    return str(output_file)


def plot_judge_distribution(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
) -> str:
    """
    Plot judge score distribution across experiments.
    
    Args:
        scores_dir: Directory containing score CSV files
        output_file: Output image path
        
    Returns:
        Path to saved figure
    """
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    
    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / "judge_distribution.png"
    
    scores_dir = Path(scores_dir)
    output_file = Path(output_file)
    ensure_dir(output_file)
    
    results_df = _load_scores(scores_dir)
    if results_df.empty:
        _create_placeholder_figure(output_file, "No judge data available")
        return str(output_file)
    
    # Check for judge columns
    judge_cols = ["judge_correct_pct", "judge_partial_pct", "judge_wrong_pct"]
    if not all(c in results_df.columns for c in judge_cols):
        _create_placeholder_figure(output_file, "Judge scores not available")
        return str(output_file)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(results_df))
    
    bottom = [0] * len(results_df)
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    labels = ["Correct (2)", "Partial (1)", "Wrong (0)"]
    
    for col, color, label in zip(judge_cols, colors, labels):
        values = results_df[col].fillna(0).values
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Percentage")
    ax.set_title("Judge Score Distribution by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["experiment"], rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Judge distribution saved to {output_file}")
    return str(output_file)


def plot_latency_quality_tradeoff(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
) -> str:
    """Scatter plot of latency vs ROUGE-L with judge color."""
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / "latency_quality_tradeoff.png"

    scores_dir = Path(scores_dir)
    output_file = Path(output_file)
    ensure_dir(output_file)

    df = _load_scores(scores_dir)
    required = {"latency_mean_ms", "rougeL"}
    if df.empty or not required.issubset(set(df.columns)):
        _create_placeholder_figure(output_file, "Latency/quality data not available")
        return str(output_file)

    fig, ax = plt.subplots(figsize=(10, 6))
    color_vals = df["judge_mean"] if "judge_mean" in df.columns else None
    sc = ax.scatter(
        df["latency_mean_ms"],
        df["rougeL"],
        c=color_vals,
        cmap="viridis",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["experiment"].replace("exp_", ""),
            (row["latency_mean_ms"], row["rougeL"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    if color_vals is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Judge mean")

    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("Latency vs ROUGE-L trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Latency-quality tradeoff saved to {output_file}")
    return str(output_file)


def plot_format_statistics(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
) -> str:
    """Bar plot for output-format behavior across experiments."""
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / "format_statistics.png"

    scores_dir = Path(scores_dir)
    output_file = Path(output_file)
    ensure_dir(output_file)

    df = _load_scores(scores_dir)
    cols = ["multi_sentence_pct", "insufficient_info_pct", "empty_pct"]
    if df.empty or not all(c in df.columns for c in cols):
        _create_placeholder_figure(output_file, "Format statistics not available")
        return str(output_file)

    plot_df = df[["experiment"] + cols].copy()
    melted = plot_df.melt(id_vars="experiment", var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melted, x="experiment", y="value", hue="metric", ax=ax)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Percentage")
    ax.set_title("Output format statistics by experiment")
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Format statistics figure saved to {output_file}")
    return str(output_file)


def generate_all_figures(
    scores_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> list[str]:
    """Generate all figures for the report."""
    if output_dir is None:
        output_dir = get_project_root() / "results" / "figures"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    figures = []
    
    # Metrics comparison
    try:
        fig = plot_metrics_comparison(scores_dir, output_dir / "metrics_comparison.png")
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Failed to create metrics comparison: {e}")
    
    # Judge distribution
    try:
        fig = plot_judge_distribution(scores_dir, output_dir / "judge_distribution.png")
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Failed to create judge distribution: {e}")
    
    # Ablation curve
    try:
        fig = plot_ablation_curve(scores_dir, output_dir / "ablation_curve.png")
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Failed to create ablation curve: {e}")

    # Latency/quality trade-off
    try:
        fig = plot_latency_quality_tradeoff(scores_dir, output_dir / "latency_quality_tradeoff.png")
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Failed to create latency-quality tradeoff: {e}")

    # Output format statistics
    try:
        fig = plot_format_statistics(scores_dir, output_dir / "format_statistics.png")
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Failed to create format statistics: {e}")
    
    return figures


def _create_placeholder_figure(output_file: Path, message: str):
    """Create a placeholder figure with a message."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(output_file, dpi=100)
    plt.close()
