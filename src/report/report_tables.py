import argparse
from pathlib import Path
from typing import List

import polars as pl

from src.utils import load_config, log_config, log_versions, setup_logger, write_text


def table_to_latex(df: pl.DataFrame, caption: str, label: str) -> str:
    columns = df.columns
    header = " & ".join(columns) + " \\\\"
    rows = []
    for row in df.iter_rows(named=True):
        values = [str(row.get(col, "")) for col in columns]
        rows.append(" & ".join(values) + " \\\\")
    tabular = "\n".join(
        [
            "\\begin{tabular}{" + "l" * len(columns) + "}",
            "\\toprule",
            header,
            "\\midrule",
            "\n".join(rows),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            tabular,
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("report_tables")
    log_versions(logger)
    log_config(logger, cfg)

    rows: List[dict] = []
    for exp in cfg["experiments"]:
        score_path = exp["score_path"]
        if not Path(score_path).exists():
            logger.warning("Missing score file %s", score_path)
            continue
        df = pl.read_csv(score_path)
        rows.append({**exp, **df.to_dicts()[0]})

    if not rows:
        logger.warning("No results found to build tables.")
        return

    df = pl.DataFrame(rows)
    summary_cols = ["exp_name", "rougeL", "bleu", "judge_mean", "latency_mean_ms", "cost_mean"]
    summary = df.select([c for c in summary_cols if c in df.columns])
    summary_tex = table_to_latex(summary, "Résumé des métriques principales.", "tab:summary")

    assets_dir = Path(cfg.get("output_assets_dir", "results/report_assets"))
    assets_dir.mkdir(parents=True, exist_ok=True)
    write_text(summary_tex, assets_dir / "summary_table.tex")
    logger.info("Saved summary table to %s", assets_dir / "summary_table.tex")


if __name__ == "__main__":
    main()
