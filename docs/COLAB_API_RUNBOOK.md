# Colab + API Runbook (Final Compute Boost)

This runbook describes how to upgrade the current local report with:

1. **Real GPU fine-tuning on Google Colab (T4)**
2. **Cloud API experiments (OpenAI/OpenRouter)**

The goal is to regenerate scores/figures with a stronger compute setup before final submission.

---

## 1) Google Colab setup (GPU)

1. Open Colab: https://colab.research.google.com  
2. Runtime -> Change runtime type -> **T4 GPU**
3. Run:

```python
!git clone <YOUR_REPO_URL>
%cd Deep-Learning
!pip install -e .
!python -m src.data.cli download
!python -m src.data.cli split --seed 42 --tiny-size 200
```

### Fine-tuning runs (recommended)

```python
!python -m src.finetune.cli prepare --max 1000 --style alpaca
!python -m src.finetune.cli prepare --max 5000 --style alpaca
!python -m src.finetune.cli prepare --max 20000 --style alpaca

!python -m src.finetune.cli train configs/exp_08_finetune_1k.yaml
!python -m src.finetune.cli train configs/exp_09_finetune_5k.yaml
!python -m src.finetune.cli train configs/exp_10_finetune_20k.yaml
```

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

Run full evaluation for each prediction file:

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

