import argparse
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import numpy as np
import polars as pl
import sacrebleu

from src.utils import count_sentences, load_config, log_config, log_versions, read_jsonl, setup_logger, write_json


def load_judgements(path: Optional[str]) -> Dict[int, dict]:
    if not path or not Path(path).exists():
        return {}
    rows = read_jsonl(path)
    return {row["id"]: row for row in rows}


def compute_latency(preds: List[dict]) -> Dict[str, float]:
    latencies = [row.get("meta", {}).get("latency_ms") for row in preds]
    latencies = [x for x in latencies if isinstance(x, int)]
    if not latencies:
        return {"latency_mean_ms": None, "latency_p95_ms": None}
    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def compute_cost(preds: List[dict]) -> Dict[str, float]:
    costs = [row.get("meta", {}).get("cost_estimate") for row in preds]
    costs = [c for c in costs if isinstance(c, (int, float))]
    if not costs:
        return {"cost_mean": None, "cost_sum": None}
    return {"cost_mean": float(np.mean(costs)), "cost_sum": float(np.sum(costs))}


def compute_stats(preds: List[dict]) -> Dict[str, float]:
    lengths = [len(str(row.get("prediction", "")).split()) for row in preds]
    empty_rate = float(np.mean([1 if not str(row.get("prediction", "")).strip() else 0 for row in preds]))
    multi_sent_rate = float(
        np.mean([1 if count_sentences(str(row.get("prediction", ""))) > 1 else 0 for row in preds])
    )
    insuf_rate = float(
        np.mean(
            [
                1
                if "insufficient information" in str(row.get("prediction", "")).lower()
                else 0
                for row in preds
            ]
        )
    )
    return {
        "pred_len_mean": float(np.mean(lengths)) if lengths else None,
        "pred_len_p95": float(np.percentile(lengths, 95)) if lengths else None,
        "empty_rate": empty_rate,
        "multi_sentence_rate": multi_sent_rate,
        "insufficient_rate": insuf_rate,
    }


def compute_metrics(preds: List[dict]) -> Dict[str, float]:
    predictions = [row.get("prediction", "") for row in preds]
    references = [row.get("reference", "") for row in preds]

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    return {
        "rougeL": float(rouge_scores.get("rougeL", 0.0)),
        "bleu": float(bleu_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROUGE/BLEU and stats.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["exp_name"]
    logger = setup_logger(f"{exp_name}_metrics", cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)

    preds = read_jsonl(cfg["output"]["pred_path"])
    metrics = compute_metrics(preds)
    metrics.update(compute_latency(preds))
    metrics.update(compute_cost(preds))
    metrics.update(compute_stats(preds))

    judge_path = cfg["output"].get("judge_path")
    judge_path_2 = cfg["output"].get("judge_path_2")
    judge = load_judgements(judge_path)
    judge2 = load_judgements(judge_path_2)
    if judge:
        scores = [row.get("score") for row in judge.values() if row.get("score") is not None]
        metrics["judge_mean"] = float(np.mean(scores)) if scores else None
    if judge and judge2:
        disagreements = 0
        total = 0
        for jid, row in judge.items():
            other = judge2.get(jid)
            if not other:
                continue
            total += 1
            if row.get("score") != other.get("score"):
                disagreements += 1
        metrics["judge_disagreement_rate"] = float(disagreements / total) if total else None

    metrics["n_samples"] = len(preds)
    metrics["exp_name"] = exp_name

    score_path = cfg["output"]["score_path"]
    df = pl.DataFrame([metrics])
    Path(score_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(score_path)
    logger.info("Saved metrics to %s", score_path)

    if cfg["output"].get("metrics_json_path"):
        write_json(metrics, cfg["output"]["metrics_json_path"])


if __name__ == "__main__":
    main()
