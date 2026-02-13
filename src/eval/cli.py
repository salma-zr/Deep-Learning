"""CLI for evaluation."""

from pathlib import Path
from typing import Optional
import random

import typer
from rich.console import Console
from rich.table import Table
import pandas as pd

from src.eval.metrics import compute_all_metrics, metrics_to_csv_row
from src.eval.judge import LLMJudge, judge_predictions, verify_judge_consistency
from src.eval.qualitative import generate_qualitative_report
from src.utils.io_utils import load_jsonl, save_jsonl, ensure_dir, get_project_root
from src.utils.logger import setup_logger

app = typer.Typer(help="Evaluation commands")
console = Console()


@app.command()
def metrics(
    predictions_file: str = typer.Argument(..., help="Path to predictions JSONL"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV file"),
    bertscore: bool = typer.Option(False, "--bertscore", help="Include BERTScore metric"),
    bertscore_model: str = typer.Option(
        "distilbert-base-uncased",
        "--bertscore-model",
        help="Model used by BERTScore",
    ),
    bootstrap_samples: int = typer.Option(
        0,
        "--bootstrap-samples",
        help="Bootstrap samples for ROUGE confidence intervals (0 disables)",
    ),
    bootstrap_ci: float = typer.Option(
        95.0,
        "--bootstrap-ci",
        help="Confidence level percentage for bootstrap interval",
    ),
):
    """Compute ROUGE and BLEU metrics for predictions."""
    console.print(f"[bold blue]Computing metrics for {predictions_file}...[/bold blue]")
    
    # Load predictions
    predictions = load_jsonl(predictions_file)
    
    # Extract data
    preds = [p["prediction"] for p in predictions]
    refs = [p["reference"] for p in predictions]
    latencies = [p.get("meta", {}).get("latency_ms") for p in predictions]
    latencies = [l for l in latencies if l is not None]
    
    # Check for judge scores
    judge_scores = [p.get("judge_score") for p in predictions]
    judge_scores = [s for s in judge_scores if s is not None] or None
    
    # Compute metrics
    metrics = compute_all_metrics(
        predictions=preds,
        references=refs,
        latencies_ms=latencies if latencies else None,
        judge_scores=judge_scores,
        include_bertscore=bertscore,
        bertscore_model=bertscore_model,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )
    
    # Display
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)
    
    # Save if output specified
    if output_file:
        exp_name = Path(predictions_file).stem
        row = metrics_to_csv_row(exp_name, metrics)
        df = pd.DataFrame([row])
        df.to_csv(output_file, index=False)
        console.print(f"[green]Metrics saved to {output_file}[/green]")


@app.command()
def judge(
    predictions_file: str = typer.Argument(..., help="Path to predictions JSONL"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Judge model"),
    backend: str = typer.Option("openai", "--backend", "-b", help="Backend"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run LLM judge evaluation on predictions."""
    console.print(f"[bold blue]Running LLM judge ({backend}/{model})...[/bold blue]")
    
    # Determine output file
    if output_file is None:
        output_dir = get_project_root() / "results" / "preds"
        ensure_dir(output_dir)
        output_file = output_dir / f"{Path(predictions_file).stem}_judged.jsonl"
    
    # Run judging
    results, stats = judge_predictions(
        predictions_file=predictions_file,
        model=model,
        backend=backend,
        output_file=output_file,
    )
    
    # Display results
    table = Table(title="Judge Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total examples", str(stats["total"]))
    table.add_row("Judged", str(stats["judged"]))
    table.add_row("Errors", str(stats["errors"]))
    table.add_row("Mean score", f"{stats['mean_score']:.3f}")
    table.add_row("Correct (2)", f"{stats['correct_pct']:.1f}%")
    table.add_row("Partial (1)", f"{stats['partial_pct']:.1f}%")
    table.add_row("Wrong (0)", f"{stats['wrong_pct']:.1f}%")
    
    console.print(table)
    console.print(f"[green]Results saved to {output_file}[/green]")


@app.command()
def verify_consistency(
    predictions_file: str = typer.Argument(..., help="Path to judged predictions"),
    n_samples: int = typer.Option(50, "--samples", "-n", help="Number of samples to verify"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Judge model"),
):
    """Verify judge consistency by re-judging samples."""
    console.print(f"[bold blue]Verifying judge consistency...[/bold blue]")
    console.print(f"Samples: {n_samples}")
    
    # Load predictions
    predictions = load_jsonl(predictions_file)
    
    # Create judge
    judge = LLMJudge(model=model)
    
    # Verify consistency
    stats = verify_judge_consistency(
        examples=predictions,
        judge=judge,
        n_samples=n_samples,
        n_repeats=2,
    )
    
    table = Table(title="Consistency Check")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Samples checked", str(stats["samples_checked"]))
    table.add_row("Disagreements", str(stats["disagreements"]))
    table.add_row("Consistency rate", f"{stats['consistency_rate']:.1%}")
    
    console.print(table)


@app.command()
def qualitative(
    predictions_file: str = typer.Argument(..., help="Path to judged predictions"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output HTML file"),
    n_examples: int = typer.Option(30, "--n", help="Number of examples to include"),
):
    """Generate qualitative analysis HTML report."""
    console.print(f"[bold blue]Generating qualitative report...[/bold blue]")
    
    output_path = generate_qualitative_report(
        predictions_file=predictions_file,
        output_file=output_file,
        n_examples=n_examples,
    )
    
    console.print(f"[green]Report saved to {output_path}[/green]")


@app.command("prepare-human-annotation")
def prepare_human_annotation(
    predictions_file: str = typer.Argument(..., help="Path to predictions JSONL"),
    n_samples: int = typer.Option(40, "--n", "-n", help="Number of samples (30-50 recommended)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    strategy: str = typer.Option(
        "balanced",
        "--strategy",
        help="Sampling strategy: balanced (by judge score if available) or random",
    ),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV path"),
):
    """Create a human-annotation pack from model predictions."""
    if strategy not in {"balanced", "random"}:
        raise typer.BadParameter("strategy must be 'balanced' or 'random'")
    if n_samples <= 0:
        raise typer.BadParameter("n must be > 0")

    examples = load_jsonl(predictions_file)
    if not examples:
        console.print("[red]No predictions found in file.[/red]")
        raise typer.Exit(code=1)

    target_n = min(n_samples, len(examples))
    rng = random.Random(seed)

    # Try balanced sampling by judge score when available.
    sampled = []
    if strategy == "balanced":
        by_score = {0: [], 1: [], 2: []}
        for ex in examples:
            score = ex.get("judge_score")
            if score in by_score:
                by_score[score].append(ex)

        non_empty_groups = [g for g in by_score.values() if g]
        if non_empty_groups:
            per_group = target_n // len(non_empty_groups)
            remainder = target_n % len(non_empty_groups)
            for idx, group in enumerate(non_empty_groups):
                take = per_group + (1 if idx < remainder else 0)
                if len(group) <= take:
                    sampled.extend(group)
                else:
                    sampled.extend(rng.sample(group, take))

            # Fill shortfall if one group is too small.
            if len(sampled) < target_n:
                used_ids = {s.get("id") for s in sampled}
                pool = [ex for ex in examples if ex.get("id") not in used_ids]
                fill_n = min(target_n - len(sampled), len(pool))
                sampled.extend(rng.sample(pool, fill_n))
        else:
            sampled = rng.sample(examples, target_n)
    else:
        sampled = rng.sample(examples, target_n)

    # Build annotation template.
    rows = []
    for i, ex in enumerate(sampled, 1):
        rows.append(
            {
                "sample_id": i,
                "id": ex.get("id"),
                "question": ex.get("question"),
                "reference": ex.get("reference"),
                "prediction": ex.get("prediction"),
                "model": ex.get("meta", {}).get("model", ""),
                "prompt_id": ex.get("meta", {}).get("prompt_id", ""),
                "auto_judge_score": ex.get("judge_score"),
                "annotator_a_score_0_1_2": "",
                "annotator_b_score_0_1_2": "",
                "final_consensus_score_0_1_2": "",
                "notes": "",
            }
        )

    if output_file is None:
        out_dir = get_project_root() / "results" / "human_eval"
        ensure_dir(out_dir)
        output_path = out_dir / f"{Path(predictions_file).stem}_human_eval_{len(rows)}.csv"
    else:
        output_path = Path(output_file)
        ensure_dir(output_path)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    guide_path = output_path.with_suffix(".md")
    guide_path.write_text(
        "# Human Annotation Guide\n\n"
        "Scoring rubric:\n"
        "- 2: Correct (semantically equivalent and medically correct)\n"
        "- 1: Partial (contains useful correct content but incomplete or imprecise)\n"
        "- 0: Wrong (incorrect, contradictory, generic, or irrelevant)\n\n"
        "Protocol:\n"
        "1. Two annotators score independently.\n"
        "2. Resolve disagreements and fill final_consensus_score_0_1_2.\n"
        "3. Add notes for difficult/ambiguous cases.\n",
        encoding="utf-8",
    )

    table = Table(title="Human Annotation Pack")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Source file", predictions_file)
    table.add_row("Sampling strategy", strategy)
    table.add_row("Samples", str(len(rows)))
    table.add_row("CSV", str(output_path))
    table.add_row("Guide", str(guide_path))
    console.print(table)


@app.command()
def full(
    predictions_file: str = typer.Argument(..., help="Path to predictions JSONL"),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model", help="Judge model"),
    backend: str = typer.Option("openai", "--backend", help="Judge backend"),
    skip_judge: bool = typer.Option(False, "--skip-judge", help="Skip LLM judge"),
    bertscore: bool = typer.Option(False, "--bertscore", help="Include BERTScore metric"),
    bertscore_model: str = typer.Option(
        "distilbert-base-uncased",
        "--bertscore-model",
        help="Model used by BERTScore",
    ),
    bootstrap_samples: int = typer.Option(
        0,
        "--bootstrap-samples",
        help="Bootstrap samples for ROUGE confidence intervals (0 disables)",
    ),
    bootstrap_ci: float = typer.Option(
        95.0,
        "--bootstrap-ci",
        help="Confidence level percentage for bootstrap interval",
    ),
):
    """Run full evaluation pipeline (metrics + judge + qualitative)."""
    exp_name = Path(predictions_file).stem
    output_dir = get_project_root() / "results"
    
    console.print(f"[bold blue]Running full evaluation for {exp_name}...[/bold blue]")
    
    # 1. Compute metrics
    console.print("\n[bold]Step 1: Computing metrics...[/bold]")
    predictions = load_jsonl(predictions_file)
    
    preds = [p["prediction"] for p in predictions]
    refs = [p["reference"] for p in predictions]
    latencies = [p.get("meta", {}).get("latency_ms") for p in predictions if p.get("meta", {}).get("latency_ms")]
    
    all_metrics = compute_all_metrics(
        predictions=preds,
        references=refs,
        latencies_ms=latencies if latencies else None,
        include_bertscore=bertscore,
        bertscore_model=bertscore_model,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )
    
    # 2. Run judge (unless skipped)
    judged_file = predictions_file
    if not skip_judge:
        console.print("\n[bold]Step 2: Running LLM judge...[/bold]")
        judged_file = output_dir / "preds" / f"{exp_name}_judged.jsonl"
        ensure_dir(judged_file)
        
        results, judge_stats = judge_predictions(
            predictions_file=predictions_file,
            model=judge_model,
            backend=backend,
            output_file=judged_file,
        )
        
        # Add judge metrics
        all_metrics["judge_mean"] = judge_stats["mean_score"]
        all_metrics["judge_correct_pct"] = judge_stats["correct_pct"]
        all_metrics["judge_partial_pct"] = judge_stats["partial_pct"]
        all_metrics["judge_wrong_pct"] = judge_stats["wrong_pct"]
    
    # 3. Save metrics CSV
    console.print("\n[bold]Step 3: Saving metrics...[/bold]")
    scores_dir = output_dir / "scores"
    ensure_dir(scores_dir)
    
    row = metrics_to_csv_row(exp_name, all_metrics)
    df = pd.DataFrame([row])
    scores_file = scores_dir / f"{exp_name}.csv"
    df.to_csv(scores_file, index=False)
    console.print(f"Metrics saved to {scores_file}")
    
    # 4. Generate qualitative report
    if not skip_judge:
        console.print("\n[bold]Step 4: Generating qualitative report...[/bold]")
        qual_file = generate_qualitative_report(
            predictions_file=str(judged_file),
            experiment_name=exp_name,
        )
        console.print(f"Qualitative report saved to {qual_file}")
    
    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold green]Evaluation complete![/bold green]")
    
    table = Table(title=f"Summary: {exp_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    key_metrics = ["rougeL", "bleu", "judge_mean", "judge_correct_pct", "n_examples"]
    for key in key_metrics:
        if key in all_metrics:
            value = all_metrics[key]
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
    
    console.print(table)


if __name__ == "__main__":
    app()
