"""LaTeX report generation."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.report.tables import generate_results_table, generate_ablation_table
from src.report.figures import generate_all_figures
from src.utils.io_utils import ensure_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_report(
    output_dir: Optional[str | Path] = None,
    compile_pdf: bool = True,
) -> str:
    """
    Build the complete LaTeX report.
    
    Args:
        output_dir: Output directory for the report
        compile_pdf: Whether to compile to PDF
        
    Returns:
        Path to the generated report
    """
    if output_dir is None:
        output_dir = get_project_root() / "report"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info("Building report...")
    
    # Generate tables
    logger.info("Generating tables...")
    tables_dir = output_dir / "tables"
    ensure_dir(tables_dir)
    
    generate_results_table(output_file=tables_dir / "main_results.tex")
    generate_ablation_table(output_file=tables_dir / "ablation.tex")
    
    # Generate figures
    logger.info("Generating figures...")
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)
    
    generate_all_figures(output_dir=figures_dir)
    
    # Copy main template if not exists
    template_path = get_project_root() / "report" / "report.tex"
    if not template_path.exists():
        logger.info("Creating report template...")
        _create_report_template(output_dir)
    
    # Compile PDF if requested
    if compile_pdf:
        pdf_path = compile_latex(output_dir / "report.tex")
        if pdf_path:
            logger.info(f"Report compiled: {pdf_path}")
            return str(pdf_path)
    
    return str(output_dir / "report.tex")


def compile_latex(
    tex_file: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Compile LaTeX file to PDF.
    
    Args:
        tex_file: Path to .tex file
        output_dir: Output directory (default: same as tex file)
        
    Returns:
        Path to PDF if successful, None otherwise
    """
    tex_file = Path(tex_file)
    
    if output_dir is None:
        output_dir = tex_file.parent
    
    output_dir = Path(output_dir)
    
    # Check for pdflatex
    if not shutil.which("pdflatex"):
        logger.warning("pdflatex not found. Install LaTeX to compile PDF.")
        logger.warning("On Ubuntu: sudo apt-get install texlive-latex-base texlive-latex-extra")
        return None
    
    # Compile (run twice for references)
    try:
        for _ in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory", str(output_dir),
                    str(tex_file),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        
        pdf_path = output_dir / tex_file.with_suffix(".pdf").name
        
        if pdf_path.exists():
            return str(pdf_path)
        else:
            logger.error(f"PDF compilation failed. Check {tex_file.stem}.log")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("LaTeX compilation timed out")
        return None
    except Exception as e:
        logger.error(f"LaTeX compilation error: {e}")
        return None


def _create_report_template(output_dir: Path):
    """Create the main report LaTeX template."""
    template = r"""\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

\geometry{margin=2.5cm}

% Title
\title{Medical Question Answering:\\A Comparative Study of LLM Approaches}
\author{Deep Learning Course Project}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report presents a comparative study of different approaches for medical question answering using the Medical Flashcards dataset. We evaluate closed-book generation, prompt engineering, fine-tuning with LoRA/QLoRA, and retrieval-augmented generation (RAG). Our evaluation combines automatic metrics (ROUGE, BLEU) with LLM-as-a-judge for semantic equivalence assessment. We provide a detailed analysis of each approach's strengths, limitations, and practical considerations.

\textbf{Medical Disclaimer:} This is an educational research project. The AI-generated answers may contain errors or hallucinations. This content should not be used for medical advice.
\end{abstract}

\section{Introduction}

Medical question answering is a challenging task that requires both factual accuracy and appropriate response formatting. This project explores various approaches to answering medical questions from a flashcard-style dataset, comparing traditional language model generation with retrieval-augmented and fine-tuned methods.

Our objectives are:
\begin{itemize}
    \item Compare closed-book LLM performance across model sizes and providers
    \item Evaluate the impact of different prompting strategies
    \item Assess the benefits of fine-tuning on domain-specific data
    \item Analyze retrieval-augmented generation for knowledge grounding
    \item Provide honest discussion of limitations and practical considerations
\end{itemize}

\section{Dataset}

We use the \texttt{medalpaca/medical\_meadow\_medical\_flashcards} dataset from HuggingFace. This dataset contains medical questions with concise, single-sentence answers in a flashcard format.

\subsection{Dataset Statistics}
\begin{itemize}
    \item Total examples: Loaded at runtime
    \item Format: Question-Answer pairs (flashcard style)
    \item Answer length: Typically 1-2 sentences
    \item Domain: General medical knowledge
\end{itemize}

\subsection{Data Splits}
We create reproducible splits with seed=42:
\begin{itemize}
    \item Train: 80\% (for fine-tuning experiments)
    \item Development: 10\% (for hyperparameter selection)
    \item Test: 10\% (for final evaluation)
    \item Tiny Test: 200 examples (for rapid iteration)
\end{itemize}

\section{Methods}

\subsection{Closed-Book Generation}
Direct question answering using pre-trained language models without additional context. We compare:
\begin{itemize}
    \item Local models via Ollama (Llama 3.1 8B, Mistral 7B)
    \item Cloud APIs (OpenAI GPT-4o-mini, OpenRouter free tier)
    \item HuggingFace models (Flan-T5 as CPU fallback)
\end{itemize}

\subsection{Prompt Engineering}
We evaluate three prompting strategies:
\begin{enumerate}
    \item \textbf{One-sentence strict}: Explicitly requests single-sentence answers
    \item \textbf{Flashcard style}: Frames the task as flashcard completion
    \item \textbf{Uncertainty allowed}: Permits ``Insufficient information'' responses
\end{enumerate}

\subsection{Fine-tuning with LoRA/QLoRA}
Parameter-efficient fine-tuning using:
\begin{itemize}
    \item QLoRA with 4-bit quantization for memory efficiency
    \item LoRA rank 16, alpha 32
    \item Training on different data sizes (1k, 5k, 20k examples)
    \item Target model: TinyLlama-1.1B or similar small model
\end{itemize}

\subsection{Retrieval-Augmented Generation (RAG)}
Combining retrieval with generation:
\begin{itemize}
    \item Wikipedia retrieval for medical context
    \item Optional web search (DuckDuckGo)
    \item Context injection into prompts
    \item Ablations on top-k and chunk size
\end{itemize}

\section{Evaluation Methodology}

\subsection{Automatic Metrics}
\begin{itemize}
    \item \textbf{ROUGE-L}: Longest common subsequence F1
    \item \textbf{BLEU}: N-gram precision with brevity penalty
\end{itemize}

Note: These metrics measure surface-level similarity and may not capture semantic equivalence well for medical QA.

\subsection{LLM-as-a-Judge}
We use an LLM judge to assess semantic equivalence:
\begin{itemize}
    \item Score 2: Correct (semantically equivalent)
    \item Score 1: Partial (some correct information)
    \item Score 0: Wrong (incorrect or irrelevant)
\end{itemize}

We verify judge consistency by re-evaluating a subset of examples.

\subsection{Qualitative Analysis}
Manual inspection of:
\begin{itemize}
    \item Best predictions (high judge scores)
    \item Worst predictions (failures)
    \item Controversial cases (metric disagreements)
\end{itemize}

\section{Experimental Setup}

\subsection{Compute Environment}
\begin{itemize}
    \item Python 3.11
    \item Local: Ollama for inference
    \item Cloud: OpenAI API (cost-controlled)
    \item Fine-tuning: Google Colab T4 (or symbolic mode)
\end{itemize}

\subsection{Reproducibility}
\begin{itemize}
    \item Random seed: 42
    \item All prompts versioned
    \item Caching for API calls
    \item Full configuration logging
\end{itemize}

\section{Results}

% Main results table (auto-generated)
\input{tables/main_results.tex}

\subsection{Metrics Comparison}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/metrics_comparison.png}
    \caption{Comparison of ROUGE-L, BLEU, and Judge scores across experiments.}
    \label{fig:metrics}
\end{figure}

\subsection{Judge Score Distribution}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/judge_distribution.png}
    \caption{Distribution of judge scores (Correct/Partial/Wrong) by experiment.}
    \label{fig:judge}
\end{figure}

\subsection{Fine-tuning Ablation}
\input{tables/ablation.tex}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/ablation_curve.png}
    \caption{Effect of training data size on model performance.}
    \label{fig:ablation}
\end{figure}

\section{Discussion}

\subsection{Key Findings}
% To be filled based on actual results
\begin{itemize}
    \item Findings will be populated after running experiments
    \item Compare closed-book vs RAG vs fine-tuned
    \item Analyze prompt engineering effects
    \item Discuss cost-performance trade-offs
\end{itemize}

\subsection{Limitations}

\subsubsection{Dataset Limitations}
\begin{itemize}
    \item Flashcard format may not represent real clinical queries
    \item Single reference answer limits metric reliability
    \item Potential biases in medical knowledge coverage
\end{itemize}

\subsubsection{Metric Limitations}
\begin{itemize}
    \item ROUGE/BLEU penalize valid paraphrases
    \item LLM judge may have its own biases
    \item Semantic equivalence is subjective in medical domain
\end{itemize}

\subsubsection{Hallucination Risk}
\begin{itemize}
    \item All models can generate plausible but incorrect information
    \item RAG can retrieve irrelevant or outdated content
    \item Fine-tuning may reinforce dataset biases
\end{itemize}

\subsection{Cost Analysis}
\begin{itemize}
    \item Local models: Free but require hardware
    \item OpenAI API: \$X estimated for full evaluation
    \item Fine-tuning: Requires GPU (Colab free tier feasible)
\end{itemize}

\section{Conclusion}

This study provides a comprehensive comparison of approaches for medical question answering. Our findings suggest [to be completed based on results].

\textbf{Recommendations:}
\begin{itemize}
    \item For production: [based on results]
    \item For research: [based on results]
    \item Important: Always validate AI-generated medical content
\end{itemize}

\section*{Ethical Considerations}
This project is for educational purposes only. The generated answers should never be used as medical advice. Healthcare decisions should always involve qualified professionals.

\appendix

\section{Prompt Templates}
\label{app:prompts}

\subsection{One-Sentence Strict}
\begin{verbatim}
Answer the following medical question in exactly 
one clear, concise sentence.

Question: {question}

Answer:
\end{verbatim}

\subsection{Flashcard Style}
\begin{verbatim}
Complete this medical flashcard with a brief, 
accurate answer.

Medical Question: {question}

Flashcard Answer:
\end{verbatim}

\subsection{Uncertainty Allowed}
\begin{verbatim}
Answer the medical question below. If you are 
uncertain, respond with "Insufficient information."

Question: {question}

Answer:
\end{verbatim}

\section{Model Configurations}
\label{app:configs}
Full configuration details are available in the repository's \texttt{configs/} directory.

\end{document}
"""
    
    (output_dir / "report.tex").write_text(template, encoding="utf-8")
    logger.info(f"Report template created at {output_dir / 'report.tex'}")
