# Guide: How to understand and discuss this project

## 1) Strategies in one page

### Closed-book
The model answers using only its internal knowledge. This is the cheapest to run but often the least factual for specific medical details. It is useful as a baseline.

### Prompt engineering
Changing the instruction changes output format and uncertainty behavior. This project includes:
- one sentence strict
- flashcard style
- uncertainty allowed (fallback to "Insufficient information.")
We also compare post-processing that enforces one sentence vs no post-processing.

### Fine-tuning (LoRA/QLoRA)
Fine-tuning adapts a base model to the dataset style and factual patterns. LoRA reduces the number of trainable parameters and can be done on a smaller GPU. We test three data sizes (1k, 5k, 20k) to see how performance scales.

### RAG (retrieval augmented generation)
RAG retrieves text from external sources (Wikipedia by default) and conditions the answer on that context. This can improve factuality but depends strongly on retrieval quality and can introduce outdated info.

### LLM-as-a-judge
A judge model scores semantic equivalence on a 0/1/2 scale. It is not perfect but adds a semantic check that lexical metrics lack. A second pass on a small subset estimates judge variance.

## 2) Why ROUGE/BLEU are imperfect for medical QA
- They reward lexical overlap, not clinical correctness.
- A short correct answer can score low if phrased differently.
- A long incorrect answer can score high if it shares keywords.
Use ROUGE/BLEU as rough signals, not final truth.

## 3) Split design and interpretation
- We fix a random seed (42) and save split files for reproducibility.
- 80/10/10 avoids cross-validation and keeps a clean test set.
- A small `tiny_test` split speeds up debugging without paying full costs.
Interpret results on the test split only after the pipeline is stable.

## 4) How to write a strong discussion section
- Be honest about failures and hallucinations.
- Discuss metric bias and judge instability.
- Report compute constraints and cost estimates.
- Explain tradeoffs between methods (cost vs factuality).

## 5) Integrity checklist
- Do not insert results by hand.
- Generate tables and figures from scripts only.
- Log configs, seeds, and data sizes.
- Keep all results reproducible and documented.
