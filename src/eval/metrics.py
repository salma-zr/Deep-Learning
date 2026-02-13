"""Evaluation metrics for Medical QA."""

from typing import Optional
import statistics
import random

import numpy as np

from src.utils.logger import get_logger
from src.utils.normalize import get_answer_stats

logger = get_logger(__name__)


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    means, _ = compute_rouge_with_distributions(predictions, references)
    return means


def compute_rouge_with_distributions(
    predictions: list[str],
    references: list[str],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """
    Compute ROUGE means and keep per-example distributions.

    Returns:
        (mean_scores, per_example_scores)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.error("rouge-score not installed. Install with: pip install rouge-score")
        zero = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        return zero, {"rouge1": [], "rouge2": [], "rougeL": []}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }

    for pred, ref in zip(predictions, references):
        if not pred or not pred.strip():
            pred = "no answer"
        if not ref or not ref.strip():
            ref = "no answer"

        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    means = {
        "rouge1": statistics.mean(scores["rouge1"]) if scores["rouge1"] else 0.0,
        "rouge2": statistics.mean(scores["rouge2"]) if scores["rouge2"] else 0.0,
        "rougeL": statistics.mean(scores["rougeL"]) if scores["rougeL"] else 0.0,
    }
    return means, scores


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute BLEU score using sacrebleu.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers (each is a single reference)
        
    Returns:
        Dictionary with BLEU score
    """
    try:
        import sacrebleu
    except ImportError:
        logger.error("sacrebleu not installed. Install with: pip install sacrebleu")
        return {"bleu": 0.0}
    
    # sacrebleu expects references as list of lists
    refs = [[ref if ref and ref.strip() else "no answer"] for ref in references]
    preds = [pred if pred and pred.strip() else "no answer" for pred in predictions]
    
    # Compute corpus-level BLEU
    bleu = sacrebleu.corpus_bleu(preds, list(zip(*refs)))
    
    return {"bleu": bleu.score / 100}  # Normalize to 0-1 range


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Compute BERTScore (precision/recall/F1) as a semantic similarity metric.

    Note:
        This metric is optional and requires installing `bert-score`.
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.warning(
            "bert-score not installed. Install optional research extras "
            "with: pip install -e .[research]"
        )
        return {}

    preds = [pred if pred and pred.strip() else "no answer" for pred in predictions]
    refs = [ref if ref and ref.strip() else "no answer" for ref in references]

    try:
        precision, recall, f1 = bert_score(
            preds,
            refs,
            lang="en",
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
        )
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return {}

    return {
        "bertscore_p": float(precision.mean().item()),
        "bertscore_r": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def compute_exact_match(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute exact match score.
    
    Note: Exact match is generally NOT recommended for open-ended QA
    as it's too strict. Included for completeness.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        normalize: Whether to normalize text before comparison
        
    Returns:
        Dictionary with exact match score
    """
    matches = 0
    
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = pred.lower().strip() if pred else ""
            ref = ref.lower().strip() if ref else ""
        
        if pred == ref:
            matches += 1
    
    return {"exact_match": matches / len(predictions) if predictions else 0.0}


def compute_format_stats(
    predictions: list[str],
) -> dict[str, float]:
    """
    Compute format statistics for predictions.
    
    Args:
        predictions: List of predicted answers
        
    Returns:
        Dictionary with format statistics
    """
    stats = get_answer_stats(predictions)
    
    return {
        "avg_length_chars": stats["avg_length"],
        "avg_length_words": stats["avg_words"],
        "multi_sentence_pct": stats["multi_sentence_pct"],
        "empty_pct": stats["empty_pct"],
        "insufficient_info_pct": stats["insufficient_pct"],
    }


def compute_latency_stats(
    latencies_ms: list[float],
) -> dict[str, float]:
    """
    Compute latency statistics.
    
    Args:
        latencies_ms: List of latencies in milliseconds
        
    Returns:
        Dictionary with latency statistics
    """
    if not latencies_ms:
        return {
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    
    sorted_latencies = sorted(latencies_ms)
    p95_idx = int(len(sorted_latencies) * 0.95)
    
    return {
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)],
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    latencies_ms: Optional[list[float]] = None,
    judge_scores: Optional[list[int]] = None,
    include_bertscore: bool = False,
    bertscore_model: str = "distilbert-base-uncased",
    bertscore_batch_size: int = 16,
    bootstrap_samples: int = 0,
    bootstrap_ci: float = 95.0,
    bootstrap_seed: int = 42,
) -> dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        latencies_ms: Optional list of latencies
        judge_scores: Optional list of judge scores (0, 1, or 2)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # ROUGE (with per-example distribution for optional bootstrap CI)
    rouge_scores, rouge_distributions = compute_rouge_with_distributions(predictions, references)
    metrics.update(rouge_scores)

    if bootstrap_samples > 0:
        for metric_name, values in rouge_distributions.items():
            if values:
                low, high = _bootstrap_mean_ci(
                    values=values,
                    n_samples=bootstrap_samples,
                    ci=bootstrap_ci,
                    seed=bootstrap_seed,
                )
                metrics[f"{metric_name}_ci_low"] = low
                metrics[f"{metric_name}_ci_high"] = high
    
    # BLEU
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    # Optional semantic metric (research-level)
    if include_bertscore:
        bert_scores = compute_bertscore(
            predictions,
            references,
            model_type=bertscore_model,
            batch_size=bertscore_batch_size,
        )
        metrics.update(bert_scores)
    
    # Format stats
    format_stats = compute_format_stats(predictions)
    metrics.update(format_stats)
    
    # Latency stats
    if latencies_ms:
        latency_stats = compute_latency_stats(latencies_ms)
        metrics.update(latency_stats)
    
    # Judge scores
    if judge_scores:
        valid_scores = [s for s in judge_scores if s is not None]
        if valid_scores:
            metrics["judge_mean"] = statistics.mean(valid_scores)
            metrics["judge_correct_pct"] = (valid_scores.count(2) / len(valid_scores)) * 100
            # Backward-compatible alias used in older artifacts.
            metrics["judge_perfect_pct"] = metrics["judge_correct_pct"]
            metrics["judge_partial_pct"] = (valid_scores.count(1) / len(valid_scores)) * 100
            metrics["judge_wrong_pct"] = (valid_scores.count(0) / len(valid_scores)) * 100
    
    # Add count
    metrics["n_examples"] = len(predictions)
    
    return metrics


def _bootstrap_mean_ci(
    values: list[float],
    n_samples: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return 0.0, 0.0
    if n_samples <= 0:
        m = statistics.mean(values)
        return m, m
    if not (0 < ci < 100):
        raise ValueError("ci must be in (0, 100)")

    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))

    alpha = 100 - ci
    low = float(np.percentile(boot_means, alpha / 2))
    high = float(np.percentile(boot_means, 100 - alpha / 2))
    return low, high


def metrics_to_csv_row(
    experiment_name: str,
    metrics: dict[str, float],
) -> dict[str, any]:
    """
    Convert metrics to a CSV row format.
    
    Args:
        experiment_name: Name of the experiment
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary suitable for CSV writing
    """
    row = {"experiment": experiment_name}
    
    # Round floats for readability
    for key, value in metrics.items():
        if isinstance(value, float):
            row[key] = round(value, 4)
        else:
            row[key] = value
    
    return row
