"""Qualitative analysis and HTML report generation."""

from pathlib import Path
from typing import Optional
from datetime import datetime

from src.utils.io_utils import load_jsonl, ensure_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def select_examples(
    predictions: list[dict],
    n_best: int = 10,
    n_worst: int = 10,
    n_controversial: int = 10,
) -> dict[str, list[dict]]:
    """
    Select examples for qualitative analysis.
    
    Args:
        predictions: List of predictions with judge_score and rouge scores
        n_best: Number of best examples
        n_worst: Number of worst examples
        n_controversial: Number of controversial examples (score disagreement)
        
    Returns:
        Dictionary with 'best', 'worst', 'controversial' lists
    """
    # Filter examples that have been judged
    judged = [p for p in predictions if "judge_score" in p]
    
    if not judged:
        logger.warning("No judged examples found")
        return {"best": [], "worst": [], "controversial": []}
    
    # Sort by judge score (and ROUGE-L as tiebreaker if available)
    def sort_key(p):
        score = p.get("judge_score", 0)
        rouge = p.get("rouge_l", 0)
        return (score, rouge)
    
    sorted_by_score = sorted(judged, key=sort_key, reverse=True)
    
    # Best examples (high judge score)
    best = sorted_by_score[:n_best]
    
    # Worst examples (low judge score)
    worst = sorted_by_score[-n_worst:]
    
    # Controversial: partial scores or high ROUGE but low judge (or vice versa)
    controversial = []
    for p in judged:
        score = p.get("judge_score", 0)
        rouge = p.get("rouge_l", 0) if "rouge_l" in p else None
        
        # Partial matches are inherently interesting
        if score == 1:
            controversial.append(p)
        # Or high metric disagreement
        elif rouge is not None:
            if (score == 2 and rouge < 0.3) or (score == 0 and rouge > 0.5):
                controversial.append(p)
    
    controversial = controversial[:n_controversial]
    
    return {
        "best": best,
        "worst": worst,
        "controversial": controversial,
    }


def generate_qualitative_report(
    predictions_file: str | Path,
    output_file: Optional[str | Path] = None,
    experiment_name: Optional[str] = None,
    n_examples: int = 30,
) -> str:
    """
    Generate an HTML qualitative analysis report.
    
    Args:
        predictions_file: Path to predictions JSONL with judge scores
        output_file: Output HTML file path
        experiment_name: Name of the experiment
        n_examples: Total number of examples to include
        
    Returns:
        Path to generated HTML file
    """
    # Load predictions
    predictions = load_jsonl(predictions_file)
    
    if not experiment_name:
        experiment_name = Path(predictions_file).stem
    
    # Select examples
    n_each = n_examples // 3
    selected = select_examples(
        predictions,
        n_best=n_each,
        n_worst=n_each,
        n_controversial=n_each,
    )
    
    # Generate HTML
    html = _generate_html_report(
        experiment_name=experiment_name,
        selected_examples=selected,
        total_predictions=len(predictions),
    )
    
    # Save
    if output_file is None:
        output_dir = get_project_root() / "results" / "qualitative"
        ensure_dir(output_dir)
        output_file = output_dir / f"{experiment_name}.html"
    
    Path(output_file).write_text(html, encoding="utf-8")
    
    logger.info(f"Qualitative report saved to {output_file}")
    
    return str(output_file)


def _generate_html_report(
    experiment_name: str,
    selected_examples: dict[str, list[dict]],
    total_predictions: int,
) -> str:
    """Generate HTML content for the qualitative report."""
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qualitative Analysis: {experiment_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .warning strong {{
            color: #856404;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .example {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }}
        .example.best {{
            border-left-color: #28a745;
        }}
        .example.worst {{
            border-left-color: #dc3545;
        }}
        .example.controversial {{
            border-left-color: #ffc107;
        }}
        .example-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .example-id {{
            font-size: 0.85em;
            color: #666;
        }}
        .score {{
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .score-2 {{ background: #d4edda; color: #155724; }}
        .score-1 {{ background: #fff3cd; color: #856404; }}
        .score-0 {{ background: #f8d7da; color: #721c24; }}
        .field {{
            margin-bottom: 10px;
        }}
        .field-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 3px;
        }}
        .field-value {{
            background: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }}
        .reasoning {{
            font-style: italic;
            color: #666;
            font-size: 0.9em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #f8f9fa;
        }}
        .sortable {{
            cursor: pointer;
        }}
        .sortable:hover {{
            background: #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Qualitative Analysis</h1>
        <p>Experiment: {experiment_name}</p>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="warning">
        <strong>Medical Disclaimer:</strong> This analysis is for educational and research purposes only.
        The predictions shown are generated by AI models and may contain errors or hallucinations.
        Do not use any information here as medical advice. Always consult qualified healthcare professionals.
    </div>
    
    <div class="section">
        <h2 class="section-title">Summary Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{n_best}</div>
                <div class="stat-label">Best Examples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{n_worst}</div>
                <div class="stat-label">Worst Examples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{n_controversial}</div>
                <div class="stat-label">Controversial</div>
            </div>
        </div>
    </div>
    
    {best_section}
    {worst_section}
    {controversial_section}
    
    <div class="section">
        <h2 class="section-title">Score Interpretation</h2>
        <table>
            <tr>
                <th>Score</th>
                <th>Meaning</th>
            </tr>
            <tr>
                <td><span class="score score-2">2 - Correct</span></td>
                <td>Prediction is semantically equivalent to reference answer</td>
            </tr>
            <tr>
                <td><span class="score score-1">1 - Partial</span></td>
                <td>Contains some correct information but incomplete or has minor errors</td>
            </tr>
            <tr>
                <td><span class="score score-0">0 - Wrong</span></td>
                <td>Incorrect, irrelevant, contradicts reference, or empty</td>
            </tr>
        </table>
    </div>
</body>
</html>"""
    
    def render_example(ex: dict, category: str) -> str:
        score = ex.get("judge_score", "N/A")
        reasoning = ex.get("judge_reasoning", "")
        
        return f"""
        <div class="example {category}">
            <div class="example-header">
                <span class="example-id">ID: {ex.get('id', 'N/A')}</span>
                <span class="score score-{score}">{score}</span>
            </div>
            <div class="field">
                <div class="field-label">Question</div>
                <div class="field-value">{_escape_html(ex.get('question', ''))}</div>
            </div>
            <div class="field">
                <div class="field-label">Reference Answer</div>
                <div class="field-value">{_escape_html(ex.get('reference', ''))}</div>
            </div>
            <div class="field">
                <div class="field-label">Predicted Answer</div>
                <div class="field-value">{_escape_html(ex.get('prediction', ''))}</div>
            </div>
            <div class="reasoning">Judge reasoning: {_escape_html(reasoning)}</div>
        </div>"""
    
    def render_section(title: str, examples: list[dict], category: str) -> str:
        if not examples:
            return ""
        
        examples_html = "\n".join(render_example(ex, category) for ex in examples)
        
        return f"""
    <div class="section">
        <h2 class="section-title">{title}</h2>
        {examples_html}
    </div>"""
    
    best_section = render_section(
        "Best Examples (Score = 2)",
        selected_examples.get("best", []),
        "best"
    )
    
    worst_section = render_section(
        "Worst Examples (Score = 0)",
        selected_examples.get("worst", []),
        "worst"
    )
    
    controversial_section = render_section(
        "Controversial Examples (Score = 1 or Metric Disagreement)",
        selected_examples.get("controversial", []),
        "controversial"
    )
    
    return html_template.format(
        experiment_name=experiment_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total=total_predictions,
        n_best=len(selected_examples.get("best", [])),
        n_worst=len(selected_examples.get("worst", [])),
        n_controversial=len(selected_examples.get("controversial", [])),
        best_section=best_section,
        worst_section=worst_section,
        controversial_section=controversial_section,
    )


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
