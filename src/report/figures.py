"""Generate figures for the report."""

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
    
    # Load all scores
    all_results = []
    for csv_file in scores_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = csv_file.stem
                all_results.append(row)
        except Exception:
            pass
    
    if not all_results:
        logger.warning("No results to plot")
        _create_placeholder_figure(output_file, "No results available")
        return str(output_file)
    
    results_df = pd.DataFrame(all_results)
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        _create_placeholder_figure(output_file, "Metrics not available")
        return str(output_file)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(results_df))
    width = 0.8 / len(available_metrics)
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        values = results_df[metric].fillna(0)
        bars = ax.bar([xi + offset for xi in x], values, width, label=metric)
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["experiment"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Metrics comparison plot saved to {output_file}")
    return str(output_file)


def plot_latency_comparison(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
) -> str:
    """
    Create a bar chart comparing average latency across experiments.
    """
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"

    if output_file is None:
        output_file = get_project_root() / "results" / "figures" / "latency_comparison.png"

    scores_dir = Path(scores_dir)
    output_file = Path(output_file)
    ensure_dir(output_file)

    all_results = []
    for csv_file in scores_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = csv_file.stem
                all_results.append(row)
        except Exception:
            pass

    if not all_results:
        _create_placeholder_figure(output_file, "No latency data available")
        return str(output_file)

    results_df = pd.DataFrame(all_results)
    if "latency_mean_ms" not in results_df.columns:
        _create_placeholder_figure(output_file, "Latency metric not available")
        return str(output_file)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(results_df["experiment"], results_df["latency_mean_ms"].fillna(0))
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Average Latency per Experiment")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Latency comparison plot saved to {output_file}")
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
    
    # Load relevant experiments
    all_results = []
    for csv_file in scores_dir.glob(f"{prefix}*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = csv_file.stem
                
                # Extract size
                import re
                match = re.search(r'(\d+)k?', csv_file.stem)
                if match:
                    num = int(match.group(1))
                    row["size"] = num * 1000 if 'k' in csv_file.stem.lower() else num
                else:
                    row["size"] = 0
                
                all_results.append(row)
        except Exception:
            pass
    
    if not all_results:
        _create_placeholder_figure(output_file, "No ablation data available")
        return str(output_file)
    
    results_df = pd.DataFrame(all_results).sort_values("size")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ["rougeL", "judge_mean"]
    colors = ["#2ecc71", "#3498db"]
    
    for metric, color in zip(metrics, colors):
        if metric in results_df.columns:
            ax.plot(
                results_df["size"],
                results_df[metric],
                marker="o",
                label=metric,
                color=color,
                linewidth=2,
                markersize=8,
            )
    
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.set_title("Effect of Training Data Size on Performance")
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Log scale for x-axis if range is large
    if results_df["size"].max() / max(results_df["size"].min(), 1) > 10:
        ax.set_xscale("log")
    
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
    
    # Load scores
    all_results = []
    for csv_file in scores_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = csv_file.stem
                all_results.append(row)
        except Exception:
            pass
    
    if not all_results:
        _create_placeholder_figure(output_file, "No judge data available")
        return str(output_file)
    
    results_df = pd.DataFrame(all_results)
    
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

    # Latency comparison
    try:
        fig = plot_latency_comparison(scores_dir, output_dir / "latency_comparison.png")
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Failed to create latency comparison: {e}")
    
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
