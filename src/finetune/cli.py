"""CLI for fine-tuning experiments."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.finetune.trainer import FineTuner, FineTuneConfig, run_symbolic_finetune
from src.finetune.data_prep import prepare_training_data, create_train_sizes
from src.utils.io_utils import load_yaml, get_project_root
from src.utils.logger import setup_logger

app = typer.Typer(help="Fine-tuning commands")
console = Console()


@app.command()
def prepare(
    split: str = typer.Option("train", "--split", "-s", help="Split to prepare"),
    max_examples: Optional[int] = typer.Option(None, "--max", "-n", help="Max examples"),
    style: str = typer.Option("alpaca", "--style", help="Prompt style"),
):
    """Prepare training data for fine-tuning."""
    console.print(f"[bold blue]Preparing training data...[/bold blue]")
    console.print(f"Split: {split}, Style: {style}")
    
    if max_examples:
        console.print(f"Max examples: {max_examples}")
    
    path = prepare_training_data(
        split_name=split,
        max_examples=max_examples,
        prompt_style=style,
    )
    
    console.print(f"[green]Saved to: {path}[/green]")


@app.command()
def prepare_ablation(
    sizes: str = typer.Option("1000,5000,20000", "--sizes", help="Comma-separated sizes"),
    style: str = typer.Option("alpaca", "--style", help="Prompt style"),
):
    """Prepare multiple training sizes for ablation study."""
    size_list = [int(s.strip()) for s in sizes.split(",")]
    
    console.print(f"[bold blue]Preparing ablation data...[/bold blue]")
    console.print(f"Sizes: {size_list}")
    
    paths = create_train_sizes(sizes=size_list, prompt_style=style)
    
    table = Table(title="Prepared Training Files")
    table.add_column("Size", style="cyan")
    table.add_column("Path", style="green")
    
    for size, path in paths.items():
        table.add_row(str(size), str(path))
    
    console.print(table)


@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to config YAML"),
    train_data: str = typer.Option(None, "--train", "-t", help="Training data path"),
    eval_data: str = typer.Option(None, "--eval", "-e", help="Evaluation data path"),
    symbolic: bool = typer.Option(False, "--symbolic", help="Run symbolic (no GPU) training"),
):
    """Run fine-tuning from config."""
    config = load_yaml(config_path)
    
    # Setup logging
    logger = setup_logger(
        "finetune",
        log_file=get_project_root() / "logs" / "finetune.log",
    )
    
    if symbolic:
        console.print("[yellow]Running SYMBOLIC training (no GPU)...[/yellow]")
        result = run_symbolic_finetune(
            train_size=config.get("train_size", 1000),
            epochs=config.get("num_epochs", 1),
            output_name=config.get("name", "symbolic"),
        )
        console.print(f"[green]Symbolic output: {result['output_dir']}[/green]")
        return
    
    # Create config
    ft_config = FineTuneConfig(
        model_name=config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        use_lora=config.get("use_lora", True),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        load_in_4bit=config.get("load_in_4bit", True),
        num_epochs=config.get("num_epochs", 3),
        batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        output_dir=config.get("output_dir", "models/finetuned"),
    )
    
    console.print(f"[bold blue]Starting fine-tuning...[/bold blue]")
    console.print(f"Model: {ft_config.model_name}")
    console.print(f"LoRA: r={ft_config.lora_r}, alpha={ft_config.lora_alpha}")
    console.print(f"Epochs: {ft_config.num_epochs}")
    
    # Determine training data path
    if train_data is None:
        train_data = get_project_root() / "data" / "finetune" / "train_alpaca.jsonl"
        if not Path(train_data).exists():
            console.print("[yellow]Training data not found. Preparing...[/yellow]")
            train_data = prepare_training_data()
    
    # Create trainer and train
    trainer = FineTuner(ft_config)
    
    try:
        model_path = trainer.train(train_data, eval_data)
        console.print(f"[green]Training complete! Model saved to: {model_path}[/green]")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        console.print("[yellow]Falling back to symbolic training...[/yellow]")
        result = run_symbolic_finetune(
            train_size=config.get("train_size", 1000),
            epochs=config.get("num_epochs", 1),
            output_name=config.get("name", "fallback"),
        )
        console.print(f"[green]Symbolic output: {result['output_dir']}[/green]")


@app.command()
def check_gpu():
    """Check GPU availability and VRAM."""
    try:
        import torch
        
        if torch.cuda.is_available():
            console.print("[green]CUDA is available![/green]")
            console.print(f"Device: {torch.cuda.get_device_name(0)}")
            
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"Total VRAM: {total_memory:.1f} GB")
            
            # Recommendations
            if total_memory >= 16:
                console.print("[green]Sufficient VRAM for most fine-tuning tasks.[/green]")
            elif total_memory >= 8:
                console.print("[yellow]Limited VRAM. Use QLoRA (4-bit) and small batch size.[/yellow]")
            else:
                console.print("[red]Very limited VRAM. Consider using Colab or cloud GPU.[/red]")
        else:
            console.print("[red]CUDA is NOT available.[/red]")
            console.print("Fine-tuning will run on CPU (very slow).")
            console.print("Consider using Google Colab for GPU access.")
            
    except ImportError:
        console.print("[red]PyTorch not installed.[/red]")


@app.command()
def generate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    question: str = typer.Option(
        "What is the function of insulin?",
        "--question", "-q",
        help="Question to answer",
    ),
):
    """Generate answer using a fine-tuned model."""
    from src.finetune.data_prep import format_instruction
    
    console.print(f"[bold blue]Loading model from {model_path}...[/bold blue]")
    
    config = FineTuneConfig()
    trainer = FineTuner(config)
    
    try:
        trainer.load_trained_model(model_path)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return
    
    # Format prompt
    prompt = format_instruction(question, "", prompt_style="alpaca").replace(
        "\n\n### Response:\n", "\n\n### Response:\n"
    ).rstrip()
    
    console.print(f"[bold]Question:[/bold] {question}")
    
    response = trainer.generate(prompt)
    
    console.print(f"[bold]Answer:[/bold] {response}")


if __name__ == "__main__":
    app()
