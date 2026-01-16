$ErrorActionPreference = "Stop"

python -m src.data.download_dataset --output data/raw/flashcards.jsonl
python -m src.data.build_splits --input data/raw/flashcards.jsonl --seed 42

python -m src.rag.build_index --config configs/exp_rag_wiki_k3.yaml
python -m src.rag.build_index --config configs/exp_rag_wiki_k5.yaml

python -m src.finetune.prepare_finetune_data --config configs/exp_ft_1k.yaml
python -m src.finetune.finetune_lora --config configs/exp_ft_1k.yaml
python -m src.finetune.prepare_finetune_data --config configs/exp_ft_5k.yaml
python -m src.finetune.finetune_lora --config configs/exp_ft_5k.yaml
python -m src.finetune.prepare_finetune_data --config configs/exp_ft_20k.yaml
python -m src.finetune.finetune_lora --config configs/exp_ft_20k.yaml

python -m src.generation.closed_book --config configs/exp_cb_ollama_llama3_8b.yaml
python -m src.generation.closed_book --config configs/exp_cb_openai_mini.yaml
python -m src.generation.closed_book --config configs/exp_cb_openrouter_free.yaml
python -m src.generation.closed_book --config configs/exp_prompt_one_sentence_post.yaml
python -m src.generation.closed_book --config configs/exp_prompt_one_sentence_nopost.yaml
python -m src.generation.closed_book --config configs/exp_prompt_flashcard.yaml
python -m src.generation.closed_book --config configs/exp_prompt_uncertainty.yaml
python -m src.generation.closed_book --config configs/exp_ft_1k.yaml
python -m src.generation.closed_book --config configs/exp_ft_5k.yaml
python -m src.generation.closed_book --config configs/exp_ft_20k.yaml
python -m src.rag.rag_generate --config configs/exp_rag_wiki_k3.yaml
python -m src.rag.rag_generate --config configs/exp_rag_wiki_k5.yaml

$experiments = @(
  "cb_ollama_llama3_8b",
  "cb_openai_mini",
  "cb_openrouter_free",
  "prompt_one_sentence_post",
  "prompt_one_sentence_nopost",
  "prompt_flashcard",
  "prompt_uncertainty",
  "ft_1k",
  "ft_5k",
  "ft_20k",
  "rag_wiki_k3",
  "rag_wiki_k5"
)

foreach ($exp in $experiments) {
  python -m src.eval.judge --config configs/judge.yaml `
    --preds "results/preds/$exp.jsonl" `
    --output "results/judge/$exp.jsonl" `
    --exp-name $exp
  python -m src.eval.judge --config configs/judge.yaml `
    --preds "results/preds/$exp.jsonl" `
    --output "results/judge/${exp}_pass2.jsonl" `
    --exp-name $exp `
    --limit 50
}

python -m src.eval.compute_metrics --config configs/exp_cb_ollama_llama3_8b.yaml
python -m src.eval.compute_metrics --config configs/exp_cb_openai_mini.yaml
python -m src.eval.compute_metrics --config configs/exp_cb_openrouter_free.yaml
python -m src.eval.compute_metrics --config configs/exp_prompt_one_sentence_post.yaml
python -m src.eval.compute_metrics --config configs/exp_prompt_one_sentence_nopost.yaml
python -m src.eval.compute_metrics --config configs/exp_prompt_flashcard.yaml
python -m src.eval.compute_metrics --config configs/exp_prompt_uncertainty.yaml
python -m src.eval.compute_metrics --config configs/exp_ft_1k.yaml
python -m src.eval.compute_metrics --config configs/exp_ft_5k.yaml
python -m src.eval.compute_metrics --config configs/exp_ft_20k.yaml
python -m src.eval.compute_metrics --config configs/exp_rag_wiki_k3.yaml
python -m src.eval.compute_metrics --config configs/exp_rag_wiki_k5.yaml

python -m src.eval.qualitative --config configs/exp_cb_ollama_llama3_8b.yaml
python -m src.eval.qualitative --config configs/exp_cb_openai_mini.yaml
python -m src.eval.qualitative --config configs/exp_cb_openrouter_free.yaml
python -m src.eval.qualitative --config configs/exp_prompt_one_sentence_post.yaml
python -m src.eval.qualitative --config configs/exp_prompt_one_sentence_nopost.yaml
python -m src.eval.qualitative --config configs/exp_prompt_flashcard.yaml
python -m src.eval.qualitative --config configs/exp_prompt_uncertainty.yaml
python -m src.eval.qualitative --config configs/exp_ft_1k.yaml
python -m src.eval.qualitative --config configs/exp_ft_5k.yaml
python -m src.eval.qualitative --config configs/exp_ft_20k.yaml
python -m src.eval.qualitative --config configs/exp_rag_wiki_k3.yaml
python -m src.eval.qualitative --config configs/exp_rag_wiki_k5.yaml

python -m src.report.build_report --config configs/report.yaml
