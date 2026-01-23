"""CLI for report generation."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from src.report.tables import (
    generate_results_table,
    generate_ablation_table,
    generate_prompt_comparison_table,
    generate_splits_table,
    generate_cost_table,
)
from src.report.figures import generate_all_figures
from src.report.latex import build_report, compile_latex
from src.utils.io_utils import get_project_root

app = typer.Typer(help="Report generation commands")
console = Console()


@app.command()
def tables(
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Generate LaTeX tables from results."""
    console.print("[bold blue]Generating LaTeX tables...[/bold blue]")
    
    if output_dir is None:
        output_dir = get_project_root() / "results" / "report_assets"
    
    output_dir = Path(output_dir)
    
    # Main results table
    generate_results_table(output_file=output_dir / "main_results.tex")
    console.print(f"Main results table: {output_dir / 'main_results.tex'}")
    
    # Ablation table
    generate_ablation_table(output_file=output_dir / "ablation.tex")
    console.print(f"Ablation table: {output_dir / 'ablation.tex'}")

    # Prompt comparison table
    generate_prompt_comparison_table(output_file=output_dir / "prompts.tex")
    console.print(f"Prompt comparison table: {output_dir / 'prompts.tex'}")

    # Splits table (from metadata)
    generate_splits_table(output_file=output_dir / "splits.tex")
    console.print(f"Splits table: {output_dir / 'splits.tex'}")

    # Cost table
    generate_cost_table(output_file=output_dir / "costs.tex")
    console.print(f"Cost table: {output_dir / 'costs.tex'}")
    
    console.print("[green]Tables generated successfully![/green]")


@app.command()
def figures(
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Generate figures from results."""
    console.print("[bold blue]Generating figures...[/bold blue]")
    
    if output_dir is None:
        output_dir = get_project_root() / "results" / "figures"
    
    figures = generate_all_figures(output_dir=output_dir)
    
    for fig in figures:
        console.print(f"Generated: {fig}")
    
    console.print("[green]Figures generated successfully![/green]")


@app.command()
def build(
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    no_compile: bool = typer.Option(False, "--no-compile", help="Skip PDF compilation"),
):
    """Build the complete report (tables + figures + PDF)."""
    console.print("[bold blue]Building report...[/bold blue]")
    
    result = build_report(
        output_dir=output_dir,
        compile_pdf=not no_compile,
    )
    
    console.print(f"[green]Report built: {result}[/green]")


@app.command()
def compile(
    tex_file: str = typer.Argument(..., help="Path to .tex file"),
):
    """Compile a LaTeX file to PDF."""
    console.print(f"[bold blue]Compiling {tex_file}...[/bold blue]")
    
    result = compile_latex(tex_file)
    
    if result:
        console.print(f"[green]PDF compiled: {result}[/green]")
    else:
        console.print("[red]Compilation failed. Check log file.[/red]")


if __name__ == "__main__":
    app()
