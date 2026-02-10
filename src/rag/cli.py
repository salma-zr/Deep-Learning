"""CLI for RAG experiments."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.rag.rag_generator import create_rag_generator
from src.data.split import load_split
from src.utils.io_utils import save_jsonl, load_yaml, ensure_dir, get_project_root
from src.utils.logger import setup_logger, log_experiment_start, log_experiment_end
from src.utils.timer import Timer

app = typer.Typer(help="RAG experiment commands")
console = Console()


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to experiment config YAML"),
    split: str = typer.Option("tiny_test", "--split", "-s", help="Split to use"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit examples"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run a RAG experiment from config."""
    # Load config
    config = load_yaml(config_path)
    exp_name = config.get("name", Path(config_path).stem)
    
    # Setup logging
    logger = setup_logger(
        f"rag_{exp_name}",
        log_file=get_project_root() / "logs" / f"rag_{exp_name}.log",
        verbose=verbose,
    )
    
    log_experiment_start(logger, exp_name, config)
    
    # Load data
    console.print(f"[bold blue]Loading {split} split...[/bold blue]")
    data = load_split(split, limit=limit)
    console.print(f"Loaded {len(data)} examples")
    
    # Create RAG generator
    console.print(f"[bold blue]Initializing RAG generator...[/bold blue]")
    console.print(f"  Retriever: {config.get('retriever_type', 'wikipedia')}")
    console.print(f"  Generator: {config.get('backend')}/{config.get('model')}")
    console.print(f"  top_k: {config.get('top_k', 3)}")
    
    rag_generator = create_rag_generator(config)
    
    # Run generation
    results = []
    total_latency = 0
    total_retrieval_latency = 0
    total_generation_latency = 0
    total_cost = 0
    errors = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"RAG Generation ({len(data)} examples)...", total=len(data))
        
        timer = Timer().start()
        
        for i, example in enumerate(data):
            # Generate with RAG
            result = rag_generator.generate(example["question"])
            
            # Build output record
            record = {
                "id": example["id"],
                "question": example["question"],
                "reference": example["answer"],
                "prediction": result.prediction,
                "meta": {
                    "model": result.model,
                    "prompt_id": result.prompt_id,
                    "latency_ms": result.latency_ms,
                    "retrieval_latency_ms": result.retrieval_latency_ms,
                    "generation_latency_ms": result.generation_latency_ms,
                    "token_usage": result.token_usage,
                    "cost_estimate": result.cost_estimate,
                    "retrieval_info": result.retrieval_info,
                },
            }
            
            if result.error:
                record["meta"]["error"] = result.error
                errors += 1
            
            results.append(record)
            total_latency += result.latency_ms
            total_retrieval_latency += result.retrieval_latency_ms
            total_generation_latency += result.generation_latency_ms
            total_cost += result.cost_estimate or 0
            
            progress.update(task, advance=1)
        
        total_time = timer.stop()
    
    # Save results
    if output_dir is None:
        output_dir = get_project_root() / "results" / "preds"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    output_file = output_dir / f"{exp_name}.jsonl"
    save_jsonl(results, output_file)
    
    # Summary
    n = len(data)
    metrics = {
        "total_examples": n,
        "errors": errors,
        "avg_latency_ms": total_latency / n if n else 0,
        "avg_retrieval_ms": total_retrieval_latency / n if n else 0,
        "avg_generation_ms": total_generation_latency / n if n else 0,
        "total_cost_usd": total_cost,
        "total_time_s": total_time / 1000,
    }
    
    log_experiment_end(logger, exp_name, metrics, total_time / 1000)
    
    # Print summary
    table = Table(title=f"RAG Experiment: {exp_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Examples", str(n))
    table.add_row("Errors", str(errors))
    table.add_row("Avg latency (ms)", f"{metrics['avg_latency_ms']:.1f}")
    table.add_row("  - Retrieval", f"{metrics['avg_retrieval_ms']:.1f}")
    table.add_row("  - Generation", f"{metrics['avg_generation_ms']:.1f}")
    table.add_row("Total cost (USD)", f"${total_cost:.4f}")
    table.add_row("Total time (s)", f"{total_time/1000:.1f}")
    table.add_row("Output file", str(output_file))
    
    console.print(table)


@app.command()
def test_retrieval(
    query: str = typer.Argument(..., help="Query to test"),
    retriever_type: str = typer.Option("wikipedia", "--type", "-t", help="Retriever type"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of documents"),
):
    """Test retrieval with a query."""
    from src.rag.retrievers import get_retriever
    
    console.print(f"[bold blue]Testing {retriever_type} retriever...[/bold blue]")
    console.print(f"Query: {query}")
    
    retriever = get_retriever(retriever_type)
    result = retriever.retrieve(query, top_k=top_k)
    
    console.print(f"\n[bold]Retrieved {len(result.documents)} documents in {result.latency_ms:.1f}ms[/bold]")
    
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
    
    for i, doc in enumerate(result.documents, 1):
        console.print(f"\n[bold cyan]Document {i}[/bold cyan]")
        console.print(f"[dim]Source: {doc.get('source', 'N/A')}[/dim]")
        console.print(doc.get('content', '')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get('content', ''))


@app.command()
def test_rag(
    question: str = typer.Argument(..., help="Question to answer"),
    backend: str = typer.Option("openai", "--backend", "-b", help="Generator backend"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model name"),
    retriever_type: str = typer.Option("wikipedia", "--retriever", "-r", help="Retriever type"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of documents"),
):
    """Test RAG with a single question."""
    config = {
        "backend": backend,
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "retriever_type": retriever_type,
        "top_k": top_k,
    }
    
    console.print(f"[bold blue]Testing RAG...[/bold blue]")
    console.print(f"Question: {question}")
    
    rag_generator = create_rag_generator(config)
    result = rag_generator.generate(question)
    
    console.print(f"\n[bold]Answer:[/bold] {result.prediction}")
    console.print(f"\n[dim]Latency: {result.latency_ms:.1f}ms (retrieval: {result.retrieval_latency_ms:.1f}ms, generation: {result.generation_latency_ms:.1f}ms)[/dim]")
    console.print(f"[dim]Sources: {', '.join(result.retrieval_info.get('sources', []))[:100]}[/dim]")
    
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")


if __name__ == "__main__":
    app()
