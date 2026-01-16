import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.utils import ensure_dir, load_config, log_config, log_versions, setup_logger, set_seed, write_json


def tokenize_seq2seq(tokenizer, batch, max_length):
    model_inputs = tokenizer(batch["input"], max_length=max_length, truncation=True)
    labels = tokenizer(batch["output"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_causal(tokenizer, batch, max_length):
    tokens = tokenizer(batch["text"], max_length=max_length, truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def load_model(cfg: Dict) -> tuple:
    model_name = cfg["model"]["name"]
    model_type = cfg["model"].get("model_type", "seq2seq")
    use_qlora = cfg["finetune"].get("use_qlora", False)
    quant_kwargs = {}
    if use_qlora:
        quant_kwargs = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **quant_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **quant_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["exp_name"]
    logger = setup_logger(exp_name, cfg.get("logging", {}).get("log_path"))
    log_versions(logger)
    log_config(logger, cfg)
    set_seed(cfg.get("seed", 42))

    output_dir = Path(cfg["finetune"]["output_dir"])
    ensure_dir(output_dir)

    if cfg["finetune"].get("symbolic", False):
        note = {
            "status": "symbolic",
            "message": "Symbolic run (no GPU). Enable GPU and set symbolic=false to train.",
        }
        write_json(note, output_dir / "symbolic.json")
        logger.info("Symbolic run, saved %s", output_dir / "symbolic.json")
        return

    prepared_path = cfg["finetune"]["prepared_path"]
    dataset = load_dataset("json", data_files={"train": prepared_path})["train"]
    logger.info("Loaded %s training rows from %s", len(dataset), prepared_path)

    tokenizer, model = load_model(cfg)
    max_length = cfg["finetune"].get("max_length", 256)
    fmt = cfg["finetune"].get("format", "seq2seq")
    if fmt == "seq2seq":
        tokenized = dataset.map(lambda b: tokenize_seq2seq(tokenizer, b, max_length), batched=True)
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    else:
        tokenized = dataset.map(lambda b: tokenize_causal(tokenizer, b, max_length), batched=True)
        collator = None

    lora_cfg = cfg["finetune"]["lora"]
    peft_cfg = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", None),
        bias="none",
        task_type="SEQ_2_SEQ_LM" if fmt == "seq2seq" else "CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["finetune"].get("batch_size", 2),
        num_train_epochs=cfg["finetune"].get("epochs", 1),
        learning_rate=cfg["finetune"].get("learning_rate", 2e-4),
        logging_steps=cfg["finetune"].get("logging_steps", 10),
        save_steps=cfg["finetune"].get("save_steps", 200),
        save_total_limit=1,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    metrics = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "train_metrics.json"
    write_json(metrics.metrics, metrics_path)
    logger.info("Saved model and metrics to %s", output_dir)


if __name__ == "__main__":
    main()
