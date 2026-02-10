# =============================================================================
# Medical QA Project - Full Pipeline Script (PowerShell)
# =============================================================================
# Windows PowerShell version of the pipeline script
#
# Usage:
#   .\scripts\run_all.ps1              # Run everything
#   .\scripts\run_all.ps1 -Quick       # Quick test with tiny_test split
#   .\scripts\run_all.ps1 -SkipApi     # Skip experiments requiring API keys
# =============================================================================

param(
    [switch]$Quick,
    [switch]$SkipApi,
    [switch]$WithFinetune
)

$ErrorActionPreference = "Continue"

# Configuration
if ($Quick) {
    $Split = "tiny_test"
    $Limit = "--limit 50"
    Write-Host "Running in QUICK MODE (tiny_test, 50 examples)" -ForegroundColor Yellow
} else {
    $Split = "test"
    $Limit = ""
}

Write-Host "================================================" -ForegroundColor Blue
Write-Host "  Medical QA Project - Full Pipeline" -ForegroundColor Blue
Write-Host "================================================" -ForegroundColor Blue
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Split: $Split"
Write-Host "  Quick mode: $Quick"
Write-Host "  Skip API: $SkipApi"
Write-Host "  With fine-tuning: $WithFinetune"
Write-Host ""

# =============================================================================
# Step 1: Data Preparation
# =============================================================================
Write-Host "[1/5] Data Preparation" -ForegroundColor Green

Write-Host "Downloading dataset..."
python -m src.data.cli download

Write-Host "Creating data splits..."
python -m src.data.cli split --seed 42 --tiny-size 200

Write-Host "Data info:"
python -m src.data.cli info

Write-Host ""

# =============================================================================
# Step 2: Generation Experiments
# =============================================================================
Write-Host "[2/5] Running Generation Experiments" -ForegroundColor Green

function Run-GenExperiment {
    param($ConfigPath)
    $Name = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    Write-Host "  Running $Name..." -ForegroundColor Cyan
    
    try {
        python -m src.generation.cli run $ConfigPath --split $Split $Limit
    } catch {
        Write-Host "  Warning: $Name failed, continuing..." -ForegroundColor Yellow
    }
}

function Run-RagExperiment {
    param($ConfigPath)
    $Name = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    Write-Host "  Running RAG $Name..." -ForegroundColor Cyan
    
    try {
        python -m src.rag.cli run $ConfigPath --split $Split $Limit
    } catch {
        Write-Host "  Warning: $Name failed, continuing..." -ForegroundColor Yellow
    }
}

if (-not $SkipApi) {
    Write-Host "Running OpenAI experiments..."
    Run-GenExperiment "configs/exp_01_openai_baseline.yaml"
    Run-GenExperiment "configs/exp_04_prompt_flashcard.yaml"
    Run-GenExperiment "configs/exp_05_prompt_uncertainty.yaml"
    
    Write-Host "Running RAG experiments..."
    Run-RagExperiment "configs/exp_06_rag_wikipedia.yaml"
    Run-RagExperiment "configs/exp_07_rag_topk_ablation.yaml"
}

# Check for Ollama
Write-Host "Checking Ollama availability..."
$OllamaAvailable = $false
try {
    $null = Get-Command ollama -ErrorAction Stop
    $OllamaAvailable = $true
} catch {
    $OllamaAvailable = $false
}

if ($OllamaAvailable) {
    Write-Host "Ollama is available, running local experiments..."
    Run-GenExperiment "configs/exp_02_ollama_llama.yaml"
} else {
    Write-Host "Ollama not available, skipping local model experiments" -ForegroundColor Yellow
}

# HuggingFace CPU fallback
Write-Host "Running HuggingFace CPU experiment..."
Run-GenExperiment "configs/exp_11_hf_flan_t5.yaml"

Write-Host ""

# =============================================================================
# Step 3: Fine-tuning (Optional)
# =============================================================================
Write-Host "[3/5] Fine-tuning Experiments" -ForegroundColor Green

if (-not $WithFinetune) {
    Write-Host "Skipping fine-tuning (use -WithFinetune to enable)" -ForegroundColor Yellow
    Write-Host "Running symbolic fine-tune for pipeline testing..."
    python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml --symbolic
} else {
    Write-Host "Preparing fine-tuning data..."
    python -m src.finetune.cli prepare --max 1000 --style alpaca
    python -m src.finetune.cli prepare --max 5000 --style alpaca
    
    Write-Host "Note: Fine-tuning requires GPU. Use Google Colab if not available locally." -ForegroundColor Yellow
    
    try {
        python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml
    } catch {
        Write-Host "Fine-tuning failed, running symbolic mode..." -ForegroundColor Yellow
        python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml --symbolic
    }
}

Write-Host ""

# =============================================================================
# Step 4: Evaluation
# =============================================================================
Write-Host "[4/5] Running Evaluation" -ForegroundColor Green

$PredDir = "results/preds"
if (Test-Path $PredDir) {
    $PredFiles = Get-ChildItem -Path $PredDir -Filter "*.jsonl"
    
    foreach ($PredFile in $PredFiles) {
        $Name = $PredFile.BaseName
        
        # Skip already judged files
        if ($Name -like "*_judged*") {
            continue
        }
        
        Write-Host "  Evaluating $Name..." -ForegroundColor Cyan
        
        if (-not $SkipApi) {
            try {
                python -m src.eval.cli full $PredFile.FullName --judge-model gpt-4o-mini
            } catch {
                Write-Host "  Full eval failed, running metrics only..." -ForegroundColor Yellow
                python -m src.eval.cli metrics $PredFile.FullName
            }
        } else {
            python -m src.eval.cli metrics $PredFile.FullName --output "results/scores/$Name.csv"
        }
    }
} else {
    Write-Host "No predictions found in $PredDir" -ForegroundColor Yellow
}

Write-Host ""

# =============================================================================
# Step 5: Report Generation
# =============================================================================
Write-Host "[5/5] Generating Report" -ForegroundColor Green

Write-Host "Generating tables and figures..."
python -m src.report.cli tables
python -m src.report.cli figures

Write-Host "Building report..."
python -m src.report.cli build --no-compile

# Check for pdflatex
$PdfLatexAvailable = $false
try {
    $null = Get-Command pdflatex -ErrorAction Stop
    $PdfLatexAvailable = $true
} catch {
    $PdfLatexAvailable = $false
}

if ($PdfLatexAvailable) {
    Write-Host "Compiling PDF..."
    python -m src.report.cli compile report/report.tex
} else {
    Write-Host "pdflatex not found. Install MiKTeX or TeX Live for PDF compilation." -ForegroundColor Yellow
}

Write-Host ""

# =============================================================================
# Summary
# =============================================================================
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Pipeline Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:"
Write-Host "  - Predictions: results/preds/"
Write-Host "  - Scores: results/scores/"
Write-Host "  - Qualitative: results/qualitative/"
Write-Host "  - Figures: results/figures/"
Write-Host "  - Report: report/report.tex"
Write-Host ""
Write-Host "Done!" -ForegroundColor Blue
