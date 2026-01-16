#!/bin/bash
# =============================================================================
# Medical QA Project - Full Pipeline Script
# =============================================================================
# This script runs the complete experimental pipeline:
# 1. Data preparation (download + split)
# 2. Generation experiments (closed-book, prompting, RAG)
# 3. Evaluation (metrics + LLM judge)
# 4. Report generation (tables + figures + PDF)
#
# Usage:
#   ./scripts/run_all.sh              # Run everything
#   ./scripts/run_all.sh --quick      # Quick test with tiny_test split
#   ./scripts/run_all.sh --skip-api   # Skip experiments requiring API keys
#
# Prerequisites:
#   - Python 3.11+ with dependencies installed
#   - OPENAI_API_KEY environment variable set (for OpenAI experiments)
#   - Ollama running (for local model experiments, optional)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
SKIP_API=false
SKIP_FINETUNE=true  # Default skip fine-tuning (needs GPU)

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-api)
            SKIP_API=true
            shift
            ;;
        --with-finetune)
            SKIP_FINETUNE=false
            shift
            ;;
        *)
            ;;
    esac
done

# Configuration
if [ "$QUICK_MODE" = true ]; then
    SPLIT="tiny_test"
    LIMIT="--limit 50"
    echo -e "${YELLOW}Running in QUICK MODE (tiny_test, 50 examples)${NC}"
else
    SPLIT="test"
    LIMIT=""
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Medical QA Project - Full Pipeline${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Configuration:"
echo "  Split: $SPLIT"
echo "  Quick mode: $QUICK_MODE"
echo "  Skip API: $SKIP_API"
echo "  Skip fine-tuning: $SKIP_FINETUNE"
echo ""

# =============================================================================
# Step 1: Data Preparation
# =============================================================================
echo -e "${GREEN}[1/5] Data Preparation${NC}"

echo "Downloading dataset..."
python -m src.data.cli download

echo "Creating data splits..."
python -m src.data.cli split --seed 42 --tiny-size 200

echo "Data info:"
python -m src.data.cli info

echo ""

# =============================================================================
# Step 2: Generation Experiments
# =============================================================================
echo -e "${GREEN}[2/5] Running Generation Experiments${NC}"

# Function to run a generation experiment
run_gen_experiment() {
    local config=$1
    local name=$(basename $config .yaml)
    
    echo -e "  ${BLUE}Running $name...${NC}"
    python -m src.generation.cli run "$config" --split $SPLIT $LIMIT || {
        echo -e "  ${YELLOW}Warning: $name failed, continuing...${NC}"
    }
}

# Function to run a RAG experiment
run_rag_experiment() {
    local config=$1
    local name=$(basename $config .yaml)
    
    echo -e "  ${BLUE}Running RAG $name...${NC}"
    python -m src.rag.cli run "$config" --split $SPLIT $LIMIT || {
        echo -e "  ${YELLOW}Warning: $name failed, continuing...${NC}"
    }
}

# Closed-book experiments
if [ "$SKIP_API" = false ]; then
    echo "Running OpenAI experiments..."
    run_gen_experiment configs/exp_01_openai_baseline.yaml
    run_gen_experiment configs/exp_04_prompt_flashcard.yaml
    run_gen_experiment configs/exp_05_prompt_uncertainty.yaml
    
    echo "Running RAG experiments..."
    run_rag_experiment configs/exp_06_rag_wikipedia.yaml
    run_rag_experiment configs/exp_07_rag_topk_ablation.yaml
fi

# Local model experiments (Ollama)
echo "Checking Ollama availability..."
if command -v ollama &> /dev/null && ollama list &> /dev/null; then
    echo "Ollama is available, running local experiments..."
    run_gen_experiment configs/exp_02_ollama_llama.yaml
else
    echo -e "${YELLOW}Ollama not available, skipping local model experiments${NC}"
fi

# HuggingFace CPU fallback
echo "Running HuggingFace CPU experiment..."
run_gen_experiment configs/exp_11_hf_flan_t5.yaml

echo ""

# =============================================================================
# Step 3: Fine-tuning (Optional, requires GPU)
# =============================================================================
echo -e "${GREEN}[3/5] Fine-tuning Experiments${NC}"

if [ "$SKIP_FINETUNE" = true ]; then
    echo -e "${YELLOW}Skipping fine-tuning (use --with-finetune to enable)${NC}"
    echo "Running symbolic fine-tune for pipeline testing..."
    python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml --symbolic
else
    echo "Preparing fine-tuning data..."
    python -m src.finetune.cli prepare --max 1000 --style alpaca
    python -m src.finetune.cli prepare --max 5000 --style alpaca
    python -m src.finetune.cli prepare --max 20000 --style alpaca
    
    echo "Running fine-tuning experiments..."
    echo -e "${YELLOW}Note: This requires GPU. Use Google Colab if not available locally.${NC}"
    
    python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml || {
        echo -e "${YELLOW}Fine-tuning failed, running symbolic mode...${NC}"
        python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml --symbolic
    }
fi

echo ""

# =============================================================================
# Step 4: Evaluation
# =============================================================================
echo -e "${GREEN}[4/5] Running Evaluation${NC}"

# Find all prediction files
PRED_DIR="results/preds"
if [ -d "$PRED_DIR" ]; then
    for pred_file in $PRED_DIR/*.jsonl; do
        if [ -f "$pred_file" ]; then
            name=$(basename $pred_file .jsonl)
            
            # Skip already judged files
            if [[ "$name" == *"_judged"* ]]; then
                continue
            fi
            
            echo -e "  ${BLUE}Evaluating $name...${NC}"
            
            if [ "$SKIP_API" = false ]; then
                # Full evaluation with LLM judge
                python -m src.eval.cli full "$pred_file" --judge-model gpt-4o-mini || {
                    echo -e "  ${YELLOW}Full eval failed, running metrics only...${NC}"
                    python -m src.eval.cli metrics "$pred_file"
                }
            else
                # Metrics only (no LLM judge)
                python -m src.eval.cli metrics "$pred_file" --output "results/scores/${name}.csv"
            fi
        fi
    done
else
    echo -e "${YELLOW}No predictions found in $PRED_DIR${NC}"
fi

echo ""

# =============================================================================
# Step 5: Report Generation
# =============================================================================
echo -e "${GREEN}[5/5] Generating Report${NC}"

echo "Generating tables and figures..."
python -m src.report.cli tables
python -m src.report.cli figures

echo "Building report..."
python -m src.report.cli build --no-compile || {
    echo -e "${YELLOW}Report build completed (PDF compilation may require LaTeX)${NC}"
}

# Try to compile PDF if pdflatex is available
if command -v pdflatex &> /dev/null; then
    echo "Compiling PDF..."
    python -m src.report.cli compile report/report.tex || {
        echo -e "${YELLOW}PDF compilation failed. Install texlive for PDF output.${NC}"
    }
else
    echo -e "${YELLOW}pdflatex not found. Install LaTeX for PDF compilation.${NC}"
    echo "  Ubuntu: sudo apt-get install texlive-latex-base texlive-latex-extra"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results:"
echo "  - Predictions: results/preds/"
echo "  - Scores: results/scores/"
echo "  - Qualitative: results/qualitative/"
echo "  - Figures: results/figures/"
echo "  - Report: report/report.tex"
echo ""

# List generated files
echo "Generated files:"
ls -la results/preds/*.jsonl 2>/dev/null || echo "  No prediction files"
ls -la results/scores/*.csv 2>/dev/null || echo "  No score files"
ls -la report/*.pdf 2>/dev/null || echo "  No PDF generated"

echo ""
echo -e "${BLUE}Done!${NC}"
