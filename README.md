# Medical Question Answering - Deep Learning Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Medical Disclaimer**: This is an educational research project. AI-generated medical answers may contain errors or hallucinations. **Never use this content for medical advice.** Always consult qualified healthcare professionals.

## TL;DR - Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Set up API keys (optional but recommended)
export OPENAI_API_KEY="your-key-here"

# 3. Run the full pipeline (quick test mode)
./scripts/run_all.sh --quick

# 4. Generate report
python -m src.report.cli build
```

## Project Overview

This project explores different approaches for answering medical questions using the [Medical Flashcards dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards). We compare:

1. **Closed-book generation**: LLMs answering from their training knowledge
2. **Prompt engineering**: Different prompting strategies to improve responses
3. **Fine-tuning**: LoRA/QLoRA adaptation on domain data
4. **RAG**: Retrieval-augmented generation with Wikipedia

### Key Features

- **Reproducible**: Seed 42, cached API calls, versioned prompts
- **Comprehensive evaluation**: ROUGE, BLEU, LLM-as-a-judge, qualitative analysis
- **Cost-aware**: Estimates API costs, supports free alternatives
- **Auto-generated report**: LaTeX tables and figures from results

## Repository Structure

```
medical-qa-project/
├── configs/                    # Experiment configurations (YAML)
│   ├── exp_01_openai_baseline.yaml
│   ├── exp_02_ollama_llama.yaml
│   └── ...
├── data/                       # Dataset and splits
│   ├── raw/                    # Downloaded dataset
│   └── splits/                 # Train/dev/test splits
├── results/                    # Experiment outputs
│   ├── preds/                  # Predictions (JSONL)
│   ├── scores/                 # Evaluation metrics (CSV)
│   ├── qualitative/            # HTML analysis reports
│   └── figures/                # Generated plots
├── report/                     # LaTeX report
│   ├── report.tex              # Main document
│   ├── tables/                 # Auto-generated tables
│   └── figures/                # Copied figures
├── scripts/                    # Automation scripts
│   ├── run_all.sh              # Linux/Mac pipeline
│   └── run_all.ps1             # Windows PowerShell
├── src/                        # Source code
│   ├── data/                   # Data loading and splitting
│   ├── generation/             # Closed-book generators
│   ├── rag/                    # RAG implementation
│   ├── finetune/               # LoRA/QLoRA training
│   ├── eval/                   # Metrics and judging
│   ├── report/                 # Report generation
│   ├── prompts/                # Prompt templates
│   └── utils/                  # Utilities (IO, cache, etc.)
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- (Optional) Ollama for local model inference
- (Optional) GPU for fine-tuning (or use Google Colab)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd medical-qa-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# For GPU support (fine-tuning)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### API Keys Configuration

```bash
# OpenAI (for GPT-4o-mini experiments)
export OPENAI_API_KEY="sk-..."

# OpenRouter (optional, for free tier models)
export OPENROUTER_API_KEY="sk-or-..."
```

### Ollama Setup (Local Models)

```bash
# Install Ollama: https://ollama.ai
# Then pull models:
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## Usage Guide

### 1. Data Preparation

```bash
# Download dataset and create splits
python -m src.data.cli download
python -m src.data.cli split --seed 42

# View dataset info
python -m src.data.cli info

# Sample examples
python -m src.data.cli sample train -n 5
```

Split sizes (default with seed=42):
- Train: 80% (~33,600 examples)
- Dev: 10% (~4,200 examples)  
- Test: 10% (~4,200 examples)
- Tiny Test: 200 examples (for quick iteration)

### 2. Run Generation Experiments

```bash
# Single experiment
python -m src.generation.cli run configs/exp_01_openai_baseline.yaml --split test

# Test with tiny set
python -m src.generation.cli run configs/exp_01_openai_baseline.yaml --split tiny_test

# Test a generator
python -m src.generation.cli test-generator openai gpt-4o-mini -q "What is insulin?"
```

### 3. Run RAG Experiments

```bash
# RAG with Wikipedia
python -m src.rag.cli run configs/exp_06_rag_wikipedia.yaml --split test

# Test retrieval
python -m src.rag.cli test-retrieval "What causes diabetes?" --type wikipedia

# Test full RAG
python -m src.rag.cli test-rag "What is the function of the pancreas?"
```

### 4. Fine-tuning (Requires GPU)

```bash
# Check GPU availability
python -m src.finetune.cli check-gpu

# Prepare training data
python -m src.finetune.cli prepare --max 5000 --style alpaca

# Run training (GPU required)
python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml

# Or run symbolic (for testing pipeline without GPU)
python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml --symbolic
```

**For Colab**: See the [Colab notebook section](#google-colab-setup) below.

### 5. Evaluation

```bash
# Compute metrics only
python -m src.eval.cli metrics results/preds/exp_01_openai_baseline.jsonl

# Run LLM judge
python -m src.eval.cli judge results/preds/exp_01_openai_baseline.jsonl

# Full evaluation (metrics + judge + qualitative)
python -m src.eval.cli full results/preds/exp_01_openai_baseline.jsonl

# Generate qualitative report
python -m src.eval.cli qualitative results/preds/exp_01_openai_baseline_judged.jsonl
```

### 6. Generate Report

```bash
# Generate tables and figures
python -m src.report.cli tables
python -m src.report.cli figures

# Build complete report
python -m src.report.cli build

# Compile PDF (requires LaTeX)
python -m src.report.cli compile report/report.tex
```

### 7. Run Full Pipeline

```bash
# Full pipeline (all experiments)
./scripts/run_all.sh

# Quick test mode (tiny_test, 50 examples)
./scripts/run_all.sh --quick

# Skip API experiments (local only)
./scripts/run_all.sh --skip-api

# Windows PowerShell
.\scripts\run_all.ps1 -Quick
```

## Experiment Plan

| # | Experiment | Strategy | Objective |
|---|-----------|----------|-----------|
| 01 | OpenAI Baseline | Closed-book | Cloud API baseline |
| 02 | Ollama Llama | Closed-book | Local model comparison |
| 03 | OpenRouter Free | Closed-book | Free API option |
| 04 | Flashcard Prompt | Prompt Eng. | Format-matching prompt |
| 05 | Uncertainty Prompt | Prompt Eng. | Reduce hallucinations |
| 06 | RAG Wikipedia | RAG | Knowledge grounding |
| 07 | RAG Top-K=5 | RAG | Ablation on retrieval |
| 08 | Fine-tune 1k | Fine-tuning | Small data baseline |
| 09 | Fine-tune 5k | Fine-tuning | Medium data |
| 10 | Fine-tune 20k | Fine-tuning | Large data |
| 11 | Flan-T5 CPU | Closed-book | CPU-only fallback |

## Evaluation Methodology

### Automatic Metrics

- **ROUGE-L**: Longest common subsequence F1
- **BLEU**: N-gram precision (sacrebleu)

### LLM-as-a-Judge

Semantic equivalence scoring:
- **2 (Correct)**: Semantically equivalent to reference
- **1 (Partial)**: Some correct information, incomplete
- **0 (Wrong)**: Incorrect, irrelevant, or empty

### Format Statistics

- Average answer length
- Multi-sentence percentage
- Empty response percentage
- "Insufficient information" rate

## Cost Estimation

| Backend | Model | Est. Cost/500 examples |
|---------|-------|------------------------|
| OpenAI | gpt-4o-mini | ~$0.50 |
| OpenAI | gpt-4o | ~$5.00 |
| OpenRouter | mistral:free | $0.00 |
| Ollama | llama3.1:8b | $0.00 (local) |
| HuggingFace | flan-t5-base | $0.00 (local) |

**Fine-tuning**: Free with Google Colab T4, or ~$1-5 on cloud GPU.

## Google Colab Setup

For fine-tuning without local GPU:

1. Open Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run:

```python
# Install dependencies
!pip install transformers peft bitsandbytes accelerate datasets

# Clone repo
!git clone <repo-url>
%cd medical-qa-project
!pip install -e .

# Prepare data
!python -m src.data.cli download
!python -m src.data.cli split
!python -m src.finetune.cli prepare --max 5000

# Train
!python -m src.finetune.cli train configs/exp_09_finetune_5k.yaml

# Download trained model
from google.colab import files
!zip -r model.zip models/finetune_5k/
files.download('model.zip')
```

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="your-key-here"
# Or add to .env file
```

### "Ollama connection refused"
```bash
# Start Ollama server
ollama serve
# In another terminal, pull model
ollama pull llama3.1:8b
```

### "CUDA out of memory" (fine-tuning)
- Use smaller batch size: `batch_size: 2`
- Enable gradient checkpointing
- Use 4-bit quantization (default)
- Use Google Colab with T4

### "pdflatex not found"
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra

# macOS
brew install --cask mactex

# Windows
# Install MiKTeX: https://miktex.org/
```

## Understanding the Code

### Prompts (`src/prompts/`)

Three main prompt types:
1. **`one_sentence_strict`**: Enforces single-sentence answers
2. **`flashcard_style`**: Matches dataset format
3. **`uncertainty_allowed`**: Permits abstention

### Caching (`src/utils/cache.py`)

API calls are cached to:
- Avoid duplicate costs
- Enable reproducibility
- Speed up reruns

Cache location: `.cache/`

### Output Format

Every experiment produces:
```json
{
  "id": "flashcard_00001",
  "question": "What is the function of insulin?",
  "reference": "Insulin regulates blood glucose levels.",
  "prediction": "Insulin helps regulate blood sugar levels in the body.",
  "meta": {
    "model": "gpt-4o-mini",
    "prompt_id": "one_sentence_strict",
    "latency_ms": 523.4,
    "token_usage": {"prompt_tokens": 50, "completion_tokens": 15},
    "cost_estimate": 0.00012
  }
}
```

## Checklist Before Submission

- [ ] All experiments completed (check `results/preds/`)
- [ ] Metrics computed (check `results/scores/`)
- [ ] Judge evaluation done (files ending in `_judged.jsonl`)
- [ ] Qualitative reports generated (check `results/qualitative/`)
- [ ] Figures generated (check `results/figures/`)
- [ ] Report compiled (`report/report.pdf`)
- [ ] Limitations discussed honestly in report
- [ ] Medical disclaimer included
- [ ] No fabricated results
- [ ] Seeds and configs logged

## License

MIT License - See LICENSE file.

## Acknowledgments

- Dataset: [MedAlpaca Medical Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- Course: Deep Learning, Master Probabilités et Finance

---

**Remember**: This is educational research. AI-generated medical content should never replace professional medical advice.
