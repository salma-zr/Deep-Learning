import argparse
import html
from pathlib import Path
from typing import Dict, List

import evaluate

from src.utils import load_config, log_config, log_versions, read_jsonl, setup_logger, write_text


def build_html(rows: List[dict], title: str) -> str:
    header = f"<h2>{html.escape(title)}</h2>"
    table = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>id</th><th>question</th><th>reference</th><th>prediction</th><th>rougeL</th><th>judge</th></tr>",
    ]
    for row in rows:
        table.append(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td>{html.escape(row['question'])}</td>"
            f"<td>{html.escape(row['reference'])}</td>"
            f"<td>{html.escape(row['prediction'])}</td>"
            f"<td>{row.get('rougeL', '')}</td>"
            f"<td>{row.get('judge_score', '')}</td>"
            "</tr>"
        )
    table.append("</table>")
    return header + "\n" + "\n".join(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative selection and HTML export.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["exp_name"]
    logger = setup_logger(f"{exp_name}_qualitative", cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)

    preds = read_jsonl(cfg["output"]["pred_path"])
    judge_path = cfg["output"].get("judge_path")
    judge_scores: Dict[int, int] = {}
    if judge_path and Path(judge_path).exists():
        judge_rows = read_jsonl(judge_path)
        judge_scores = {row["id"]: row.get("score") for row in judge_rows}

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=[row["prediction"] for row in preds],
        references=[row["reference"] for row in preds],
        rouge_types=["rougeL"],
        use_aggregator=False,
    )["rougeL"]

    enriched = []
    for row, rouge_l in zip(preds, rouge_scores):
        enriched.append(
            {
                **row,
                "rougeL": float(rouge_l),
                "judge_score": judge_scores.get(row["id"]),
            }
        )

    enriched.sort(key=lambda x: (x.get("judge_score") or 0, x.get("rougeL") or 0), reverse=True)
    best = enriched[:10]

    enriched.sort(key=lambda x: (x.get("judge_score") or 0, x.get("rougeL") or 0))
    worst = enriched[:10]

    controversial = []
    for row in enriched:
        judge = row.get("judge_score")
        if judge is None:
            continue
        diff = abs((judge / 2.0) - row.get("rougeL", 0.0))
        controversial.append((diff, row))
    controversial.sort(key=lambda x: x[0], reverse=True)
    controversial_rows = [row for _, row in controversial[:10]]

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Qualitative Analysis</title></head><body>",
        "<h1>Qualitative Analysis (no medical advice)</h1>",
        build_html(best, "Best cases"),
        build_html(worst, "Worst cases"),
        build_html(controversial_rows, "Controversial cases (judge vs ROUGE)"),
        "</body></html>",
    ]

    output_path = cfg["output"]["qualitative_path"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_text("\n".join(html_parts), output_path)
    logger.info("Saved qualitative HTML to %s", output_path)


if __name__ == "__main__":
    main()
