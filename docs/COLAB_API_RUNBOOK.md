# Colab + API Runbook (Final Compute Boost)

This runbook describes how to upgrade the current local report with:

1. **Real GPU fine-tuning on Google Colab (T4)**
2. **Cloud API experiments (OpenAI/OpenRouter)**

The goal is to regenerate scores/figures with a stronger compute setup before final submission.

---

## 1) Google Colab setup (GPU)

1. Open Colab: https://colab.research.google.com  
2. Runtime -> Change runtime type -> **T4 GPU**
3. Run this setup cell exactly (do not keep `<...>` placeholders):

```python
!git clone "https://github.com/salma-zr/Deep-Learning.git"
%cd /content/Deep-Learning
!git checkout cursor/compr-hension-d-taill-e-des-documents-aa59
!git pull origin cursor/compr-hension-d-taill-e-des-documents-aa59
!pip install -e .
!python -m src.finetune.cli check-gpu
!python -m src.data.cli download
!python -m src.data.cli split --seed 42 --tiny-size 200
```

### Fine-tuning runs (recommended)

```python
!python -m src.finetune.cli prepare --max 1000 --style alpaca
!python -m src.finetune.cli prepare --max 5000 --style alpaca
!python -m src.finetune.cli prepare --max 20000 --style alpaca

# Sanity check: expected line counts ~ 1000 / 5000 / 20000
!wc -l data/finetune/train_alpaca_1000.jsonl data/finetune/train_alpaca_5000.jsonl data/finetune/train_alpaca_20000.jsonl

# Important: pass --train explicitly so each config uses the right dataset size.
!python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml --train data/finetune/train_alpaca_1000.jsonl
!python -m src.finetune.cli train configs/exp_09_finetune_5k.yaml --train data/finetune/train_alpaca_5000.jsonl
!python -m src.finetune.cli train configs/exp_10_finetune_20k.yaml --train data/finetune/train_alpaca_20000.jsonl
```

Notes:
- Warnings about `warmup_ratio` and `use_reentrant` are expected and non-blocking.
- If Colab time is limited, run `1k` and `5k` first, then `20k` in a fresh session.

### Export artifacts back to local machine

```python
!zip -r colab_results.zip results models logs
from google.colab import files
files.download("colab_results.zip")
```

Unzip `colab_results.zip` into repository root before regenerating report.

---

## 2) Cloud API setup

If API credits are available, set keys locally:

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

Then run cloud experiments:

```bash
python -m src.generation.cli run configs/exp_01_openai_baseline.yaml --split test --limit 500
python -m src.generation.cli run configs/exp_03_openrouter_free.yaml --split test --limit 500
python -m src.generation.cli run configs/exp_04_prompt_flashcard.yaml --split test --limit 500
python -m src.generation.cli run configs/exp_05_prompt_uncertainty.yaml --split test --limit 500
python -m src.rag.cli run configs/exp_06_rag_wikipedia.yaml --split test --limit 500
python -m src.rag.cli run configs/exp_07_rag_topk_ablation.yaml --split test --limit 500
```

---

## 3) Re-evaluate and rebuild final report

Run full evaluation for each prediction file (replace `<experiment>.jsonl`):

```bash
python -m src.eval.cli full results/preds/<experiment>.jsonl --judge-model gpt-4o-mini --backend openai
```

Then regenerate report artifacts:

```bash
python -m src.report.cli tables
python -m src.report.cli figures
python -m src.report.cli compile report/report.tex
```

Final file to submit: `report/report.pdf`.

