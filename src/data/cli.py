"""CLI for data operations."""

import typer
from rich.console import Console
from rich.table import Table

from src.data.download import download_dataset, load_raw_dataset, get_dataset_info
from src.data.split import create_splits, load_splits, get_splits_metadata

app = typer.Typer(help="Data management commands")
console = Console()


@app.command()
def download(
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
):
    """Download the Medical Flashcards dataset."""
    console.print("[bold blue]Downloading Medical Flashcards dataset...[/bold blue]")
    
    path = download_dataset(output_dir=output_dir, force=force)
    
    console.print(f"[green]Dataset saved to: {path}[/green]")


@app.command()
def info():
    """Show dataset information."""
    records = load_raw_dataset()
    info = get_dataset_info(records)
    
    table = Table(title="Dataset Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total examples", str(info["num_examples"]))
    table.add_row("", "")
    table.add_row("[bold]Questions[/bold]", "")
    table.add_row("  Mean length (chars)", f"{info['question_stats']['mean_length']:.1f}")
    table.add_row("  Min length", str(info['question_stats']['min_length']))
    table.add_row("  Max length", str(info['question_stats']['max_length']))
    table.add_row("  Mean words", f"{info['question_stats']['mean_words']:.1f}")
    table.add_row("", "")
    table.add_row("[bold]Answers[/bold]", "")
    table.add_row("  Mean length (chars)", f"{info['answer_stats']['mean_length']:.1f}")
    table.add_row("  Min length", str(info['answer_stats']['min_length']))
    table.add_row("  Max length", str(info['answer_stats']['max_length']))
    table.add_row("  Mean words", f"{info['answer_stats']['mean_words']:.1f}")
    
    console.print(table)


@app.command()
def split(
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    train_ratio: float = typer.Option(0.80, "--train", help="Training ratio"),
    dev_ratio: float = typer.Option(0.10, "--dev", help="Development ratio"),
    test_ratio: float = typer.Option(0.10, "--test", help="Test ratio"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    tiny_size: int = typer.Option(200, "--tiny-size", help="Size of tiny test set"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recreation"),
):
    """Create train/dev/test splits."""
    console.print("[bold blue]Creating dataset splits...[/bold blue]")
    console.print(f"Ratios: train={train_ratio}, dev={dev_ratio}, test={test_ratio}")
    console.print(f"Seed: {seed}")
    
    splits = create_splits(
        output_dir=output_dir,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
        tiny_test_size=tiny_size,
        force=force,
    )
    
    table = Table(title="Split Sizes")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", style="green")
    
    for name, data in splits.items():
        table.add_row(name, str(len(data)))
    
    console.print(table)
    console.print("[green]Splits created successfully![/green]")


@app.command()
def show_metadata():
    """Show splits metadata."""
    metadata = get_splits_metadata()
    
    if not metadata:
        console.print("[yellow]No metadata found. Run 'split' first.[/yellow]")
        return
    
    table = Table(title="Splits Metadata")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Seed", str(metadata.get("seed", "N/A")))
    table.add_row("Total examples", str(metadata.get("total_examples", "N/A")))
    table.add_row("Dataset hash", metadata.get("dataset_hash", "N/A"))
    
    for split_name, size in metadata.get("split_sizes", {}).items():
        table.add_row(f"  {split_name}", str(size))
    
    console.print(table)


@app.command()
def sample(
    split_name: str = typer.Argument("train", help="Split to sample from"),
    n: int = typer.Option(5, "--n", "-n", help="Number of samples"),
):
    """Show sample examples from a split."""
    splits = load_splits()
    
    if split_name not in splits:
        console.print(f"[red]Unknown split: {split_name}[/red]")
        return
    
    data = splits[split_name][:n]
    
    for i, example in enumerate(data, 1):
        console.print(f"\n[bold cyan]Example {i}[/bold cyan] (ID: {example['id']})")
        console.print(f"[bold]Q:[/bold] {example['question'][:200]}...")
        console.print(f"[bold]A:[/bold] {example['answer'][:200]}...")


if __name__ == "__main__":
    app()
