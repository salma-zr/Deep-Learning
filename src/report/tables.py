"""Generate LaTeX tables from results."""

import re
from pathlib import Path
from typing import Optional
import pandas as pd

from src.utils.io_utils import ensure_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_score_rows(scores_dir: Path, pattern: str = "*.csv") -> list[dict]:
    """Load score rows from CSV files matching a pattern."""
    rows = []
    for csv_file in scores_dir.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = csv_file.stem
                rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
    return rows


def generate_results_table(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    experiments: Optional[list[str]] = None,
) -> str:
    """
    Generate main results table in LaTeX format.
    
    Args:
        scores_dir: Directory containing score CSV files
        output_file: Output .tex file path
        experiments: List of experiment names to include (default: all)
        
    Returns:
        LaTeX table string
    """
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    
    scores_dir = Path(scores_dir)
    
    # Load all score files
    all_results = []
    
    csv_files = list(scores_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No score files found in {scores_dir}")
        return _empty_table()
    
    for csv_file in csv_files:
        exp_name = csv_file.stem
        
        # Filter if experiments specified
        if experiments and exp_name not in experiments:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["experiment"] = exp_name
                all_results.append(row)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
    
    if not all_results:
        return _empty_table()
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Select and order columns
    columns = [
        ("experiment", "Experiment"),
        ("rougeL", "ROUGE-L"),
        ("bleu", "BLEU"),
        ("judge_mean", "Judge"),
        ("judge_correct_pct", "Correct \\%"),
        ("latency_mean_ms", "Latency (ms)"),
        ("n_examples", "N"),
    ]
    
    available_cols = [(col, name) for col, name in columns if col in results_df.columns]
    
    # Build LaTeX
    latex = _build_latex_table(
        df=results_df,
        columns=available_cols,
        caption="Main Results: Comparison of Medical QA Approaches",
        label="tab:main_results",
    )
    
    # Save if output file specified
    if output_file:
        output_file = Path(output_file)
        ensure_dir(output_file)
        output_file.write_text(latex, encoding="utf-8")
        logger.info(f"Results table saved to {output_file}")
    
    return latex


def generate_ablation_table(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    prefix: str = "finetune",
) -> str:
    """
    Generate ablation study table (e.g., for fine-tuning with different data sizes).
    
    Args:
        scores_dir: Directory containing score CSV files
        output_file: Output .tex file path
        prefix: Prefix to filter experiments (e.g., "finetune")
        
    Returns:
        LaTeX table string
    """
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    
    scores_dir = Path(scores_dir)
    
    # First attempt: fine-tuning ablation (by train size)
    all_results = _load_score_rows(scores_dir, f"{prefix}*.csv")
    latex = None

    if all_results:
        results_df = pd.DataFrame(all_results)

        def extract_size(name: str) -> int:
            match = re.search(r'(\d+)k?', name)
            if match:
                num = int(match.group(1))
                return num * 1000 if 'k' in name.lower() else num
            return 0

        results_df["train_size"] = results_df["experiment"].apply(extract_size)
        results_df = results_df.sort_values("train_size")

        columns = [
            ("experiment", "Configuration"),
            ("train_size", "Train Size"),
            ("rougeL", "ROUGE-L"),
            ("bleu", "BLEU"),
            ("judge_mean", "Judge"),
        ]
        available_cols = [(col, name) for col, name in columns if col in results_df.columns]

        latex = _build_latex_table(
            df=results_df,
            columns=available_cols,
            caption="Ablation Study: Effect of Training Data Size",
            label="tab:ablation",
        )
    else:
        # Fallback: RAG top-k ablation when fine-tuning data is unavailable
        rag_rows = []
        for row in _load_score_rows(scores_dir, "*.csv"):
            name = row["experiment"].lower()
            if "rag" not in name:
                continue

            top_k = None
            # Explicit top-k in experiment name
            m = re.search(r"topk[_-]?(\d+)", name)
            if m:
                top_k = int(m.group(1))
            # Default RAG configs usually use top_k=3
            elif "wikipedia" in name or "web" in name:
                top_k = 3

            if top_k is not None:
                row["top_k"] = top_k
                rag_rows.append(row)

        if rag_rows:
            rag_df = pd.DataFrame(rag_rows).sort_values(["top_k", "experiment"])
            columns = [
                ("experiment", "Configuration"),
                ("top_k", "Top-k"),
                ("rougeL", "ROUGE-L"),
                ("bleu", "BLEU"),
                ("judge_mean", "Judge"),
                ("latency_mean_ms", "Latency (ms)"),
            ]
            available_cols = [(col, name) for col, name in columns if col in rag_df.columns]

            latex = _build_latex_table(
                df=rag_df,
                columns=available_cols,
                caption="Ablation Study: Effect of Retrieval Top-k in RAG",
                label="tab:ablation",
            )
        else:
            latex = _empty_table()

    if output_file:
        output_file = Path(output_file)
        ensure_dir(output_file)
        output_file.write_text(latex, encoding="utf-8")
        logger.info(f"Ablation table saved to {output_file}")

    return latex


def generate_prompt_comparison_table(
    scores_dir: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
) -> str:
    """Generate table comparing different prompts."""
    if scores_dir is None:
        scores_dir = get_project_root() / "results" / "scores"
    
    scores_dir = Path(scores_dir)
    
    # Load prompt-related experiments
    prompt_patterns = ["prompt", "one_sentence", "flashcard", "uncertainty"]
    
    all_results = []
    for csv_file in scores_dir.glob("*.csv"):
        if any(p in csv_file.stem.lower() for p in prompt_patterns):
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    row = df.iloc[0].to_dict()
                    row["experiment"] = csv_file.stem
                    all_results.append(row)
            except Exception:
                pass
    
    if not all_results:
        return _empty_table()
    
    results_df = pd.DataFrame(all_results)
    
    columns = [
        ("experiment", "Prompt Style"),
        ("rougeL", "ROUGE-L"),
        ("bleu", "BLEU"),
        ("judge_mean", "Judge"),
        ("multi_sentence_pct", "Multi-sent \\%"),
    ]
    
    available_cols = [(col, name) for col, name in columns if col in results_df.columns]
    
    latex = _build_latex_table(
        df=results_df,
        columns=available_cols,
        caption="Prompt Engineering: Effect of Different Prompting Strategies",
        label="tab:prompts",
    )
    
    if output_file:
        output_file = Path(output_file)
        ensure_dir(output_file)
        output_file.write_text(latex, encoding="utf-8")
    
    return latex


def _build_latex_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
) -> str:
    """Build a LaTeX table from DataFrame."""
    col_names = [col for col, _ in columns]
    header_names = [name for _, name in columns]
    
    # Filter to available columns
    df_filtered = df[[c for c in col_names if c in df.columns]].copy()
    
    # Format numeric columns
    for col in df_filtered.columns:
        if df_filtered[col].dtype in ['float64', 'float32']:
            df_filtered[col] = df_filtered[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    
    # Build table
    n_cols = len(df_filtered.columns)
    col_spec = "l" + "c" * (n_cols - 1)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]
    
    # Header
    available_headers = [name for col, name in columns if col in df_filtered.columns]
    latex_lines.append(" & ".join(available_headers) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows
    for _, row in df_filtered.iterrows():
        values = [str(row[col]) if col in row else "-" for col in df_filtered.columns]
        latex_lines.append(" & ".join(values) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(latex_lines)


def _empty_table() -> str:
    """Return an empty placeholder table."""
    return """\\begin{table}[htbp]
\\centering
\\caption{Results (no data available yet)}
\\label{tab:empty}
\\begin{tabular}{lc}
\\toprule
Experiment & Metrics \\\\
\\midrule
\\textit{Run experiments to populate this table} & - \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""
