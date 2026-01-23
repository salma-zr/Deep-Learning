"""CLI for generation experiments."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.generation.prompts import get_prompt_manager
from src.data.split import load_split
from src.utils.io_utils import save_jsonl, load_yaml, ensure_dir, get_project_root
from src.utils.logger import setup_logger, log_experiment_start, log_experiment_end
from src.utils.timer import Timer
from src.utils.normalize import normalize_answer

app = typer.Typer(help="Generation experiment commands")
console = Console()


def get_generator(config: dict):
    """Create a generator from config."""
    backend = config.get("backend", "openai")
    model = config.get("model")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 256)
    prompt_id = config.get("prompt_id", "one_sentence_strict")
    system_prompt = config.get("system_prompt")
    
    if backend == "openai":
        from src.generation.openai_gen import OpenAIGenerator
        return OpenAIGenerator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_id=prompt_id,
            system_prompt=system_prompt,
        )
    elif backend == "ollama":
        from src.generation.ollama_gen import OllamaGenerator
        return OllamaGenerator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_id=prompt_id,
            system_prompt=system_prompt,
        )
    elif backend == "openrouter":
        from src.generation.openrouter_gen import OpenRouterGenerator
        return OpenRouterGenerator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_id=prompt_id,
            system_prompt=system_prompt,
        )
    elif backend == "huggingface":
        from src.generation.hf_gen import HuggingFaceGenerator
        return HuggingFaceGenerator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_id=prompt_id,
            device=config.get("device", "auto"),
            load_in_4bit=config.get("load_in_4bit", False),
            load_in_8bit=config.get("load_in_8bit", False),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to experiment config YAML"),
    split: Optional[str] = typer.Option(None, "--split", "-s", help="Split to use"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit examples"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run a generation experiment from config."""
    # Load config
    config = load_yaml(config_path)
    exp_name = config.get("name", Path(config_path).stem)
    split = split or config.get("split", "tiny_test")
    limit = limit if limit is not None else config.get("limit")
    postprocess_single_sentence = bool(config.get("postprocess_single_sentence", False))
    
    # Setup logging
    logger = setup_logger(
        f"gen_{exp_name}",
        log_file=get_project_root() / "logs" / f"gen_{exp_name}.log",
        verbose=verbose,
    )
    
    log_experiment_start(logger, exp_name, config)
    
    # Load data
    console.print(f"[bold blue]Loading {split} split...[/bold blue]")
    data = load_split(split, limit=limit)
    console.print(f"Loaded {len(data)} examples")
    
    # Get prompt manager
    pm = get_prompt_manager()
    prompt_id = config.get("prompt_id", "one_sentence_strict")
    pm.get_prompt_template(prompt_id)
    
    # Create generator
    console.print(f"[bold blue]Initializing generator ({config.get('backend')}/{config.get('model')})...[/bold blue]")
    generator = get_generator(config)
    
    # Run generation
    results = []
    total_latency = 0
    total_cost = 0
    errors = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Generating ({len(data)} examples)...", total=len(data))
        
        timer = Timer().start()
        
        for i, example in enumerate(data):
            # Format prompt
            prompt = pm.format_prompt(prompt_id, example["question"])
            
            # Generate
            result = generator.generate(prompt)
            prediction = result.prediction
            raw_prediction = prediction

            if postprocess_single_sentence:
                prediction = normalize_answer(
                    prediction,
                    lowercase=False,
                    remove_punct=False,
                    single_sentence=True,
                    remove_md=True,
                )
            
            # Build output record
            record = {
                "id": example["id"],
                "question": example["question"],
                "reference": example["answer"],
                "prediction": prediction,
                "meta": {
                    "model": result.model,
                    "prompt_id": result.prompt_id,
                    "latency_ms": result.latency_ms,
                    "token_usage": result.token_usage,
                    "cost_estimate": result.cost_estimate,
                    "retrieval_info": None,
                },
            }
            
            if result.error:
                record["meta"]["error"] = result.error
                errors += 1
            if postprocess_single_sentence:
                record["meta"]["postprocess_single_sentence"] = True
                record["meta"]["raw_prediction"] = raw_prediction
            
            results.append(record)
            total_latency += result.latency_ms
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
    avg_latency = total_latency / len(data) if data else 0
    
    metrics = {
        "total_examples": len(data),
        "errors": errors,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_cost,
        "total_time_s": total_time / 1000,
    }
    
    log_experiment_end(logger, exp_name, metrics, total_time / 1000)
    
    # Print summary
    table = Table(title=f"Experiment: {exp_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Examples", str(len(data)))
    table.add_row("Errors", str(errors))
    table.add_row("Avg latency (ms)", f"{avg_latency:.1f}")
    table.add_row("Total cost (USD)", f"${total_cost:.4f}")
    table.add_row("Total time (s)", f"{total_time/1000:.1f}")
    table.add_row("Output file", str(output_file))
    
    console.print(table)


@app.command()
def list_prompts():
    """List available prompt templates."""
    pm = get_prompt_manager()
    prompts = pm.load_answerer_prompts()
    
    table = Table(title="Available Prompts")
    table.add_column("ID", style="cyan")
    table.add_column("Description", style="green")
    
    for pid, info in prompts.items():
        table.add_row(pid, info.get("description", "N/A"))
    
    console.print(table)


@app.command()
def test_generator(
    backend: str = typer.Argument(..., help="Backend: openai, ollama, openrouter, huggingface"),
    model: str = typer.Argument(..., help="Model name"),
    question: str = typer.Option(
        "What is the function of insulin in the body?",
        "--question", "-q",
        help="Test question",
    ),
):
    """Test a generator with a single question."""
    config = {
        "backend": backend,
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "prompt_id": "one_sentence_strict",
    }
    
    console.print(f"[bold blue]Testing {backend}/{model}...[/bold blue]")
    
    try:
        generator = get_generator(config)
        pm = get_prompt_manager()
        prompt = pm.format_prompt("one_sentence_strict", question)
        
        console.print(f"\n[bold]Question:[/bold] {question}")
        console.print(f"\n[bold]Prompt:[/bold]\n{prompt[:200]}...")
        
        result = generator.generate(prompt)
        
        console.print(f"\n[bold]Response:[/bold] {result.prediction}")
        console.print(f"[dim]Latency: {result.latency_ms:.1f}ms[/dim]")
        
        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()
