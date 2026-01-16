import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import polars as pl

from src.utils import load_config, log_config, log_versions, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for report.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("make_figures")
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
        logger.warning("No results found to build figures.")
        return

    df = pl.DataFrame(rows)
    output_dir = Path(cfg["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "rougeL" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.bar(df["exp_name"], df["rougeL"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("ROUGE-L")
        plt.tight_layout()
        plt.savefig(output_dir / "rougeL.png")
        plt.close()

    if "judge_mean" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.bar(df["exp_name"], df["judge_mean"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Judge mean (0-2)")
        plt.tight_layout()
        plt.savefig(output_dir / "judge.png")
        plt.close()

    logger.info("Saved figures to %s", output_dir)


if __name__ == "__main__":
    main()
