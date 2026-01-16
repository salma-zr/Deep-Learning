"""Fine-tuning trainer with LoRA/QLoRA support."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.io_utils import save_yaml, ensure_dir, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""
    
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_length: int = 512
    
    # Output
    output_dir: str = "models/finetuned"
    
    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "output_dir": self.output_dir,
            "seed": self.seed,
        }


class FineTuner:
    """
    Fine-tuner for medical QA using LoRA/QLoRA.
    
    Supports:
    - LoRA (Low-Rank Adaptation)
    - QLoRA (Quantized LoRA with 4-bit)
    - Standard fine-tuning (if GPU memory allows)
    """
    
    def __init__(self, config: FineTuneConfig):
        """Initialize fine-tuner with config."""
        self.config = config
        self._model = None
        self._tokenizer = None
        self._peft_model = None
    
    def setup(self):
        """Load model and prepare for training."""
        logger.info(f"Setting up fine-tuner for {self.config.model_name}")
        
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as e:
            logger.error(f"Missing required packages: {e}")
            logger.error("Install with: pip install transformers peft bitsandbytes accelerate")
            raise
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training will be very slow on CPU.")
            logger.warning("Consider using Google Colab or a GPU instance.")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Quantization config
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        # Prepare for k-bit training if quantized
        if self.config.load_in_4bit or self.config.load_in_8bit:
            self._model = prepare_model_for_kbit_training(self._model)
        
        # Apply LoRA if configured
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self._peft_model = get_peft_model(self._model, lora_config)
            self._peft_model.print_trainable_parameters()
        
        logger.info("Model setup complete")
    
    def train(
        self,
        train_data_path: str | Path,
        eval_data_path: Optional[str | Path] = None,
    ) -> str:
        """
        Train the model.
        
        Args:
            train_data_path: Path to training data JSONL
            eval_data_path: Optional path to evaluation data
            
        Returns:
            Path to saved model
        """
        logger.info("Starting training...")
        
        if self._model is None:
            self.setup()
        
        try:
            from transformers import (
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
        except ImportError as e:
            logger.error(f"Missing required packages: {e}")
            raise
        
        # Load and tokenize data
        from src.finetune.data_prep import load_training_dataset
        
        train_dataset = load_training_dataset(
            train_data_path,
            self._tokenizer,
            max_length=self.config.max_length,
        )
        
        eval_dataset = None
        if eval_data_path:
            eval_dataset = load_training_dataset(
                eval_data_path,
                self._tokenizer,
                max_length=self.config.max_length,
            )
        
        # Output directory
        output_dir = Path(get_project_root()) / self.config.output_dir
        ensure_dir(output_dir)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_total_limit=2,
            seed=self.config.seed,
            fp16=True,
            report_to="none",  # Disable wandb etc.
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )
        
        # Create trainer
        model_to_train = self._peft_model if self._peft_model else self._model
        
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        self._tokenizer.save_pretrained(str(final_path))
        
        # Save config
        save_yaml(self.config.to_dict(), final_path / "training_config.yaml")
        
        logger.info(f"Training complete. Model saved to {final_path}")
        
        return str(final_path)
    
    def load_trained_model(self, model_path: str | Path):
        """Load a previously trained model."""
        logger.info(f"Loading trained model from {model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            logger.error(f"Missing required packages: {e}")
            raise
        
        model_path = Path(model_path)
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
        )
        
        # Load LoRA weights
        if self.config.use_lora:
            self._peft_model = PeftModel.from_pretrained(
                self._model,
                model_path,
            )
        
        logger.info("Trained model loaded")
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text using the trained model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call setup() or load_trained_model() first.")
        
        import torch
        
        model = self._peft_model if self._peft_model else self._model
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        
        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated


def run_symbolic_finetune(
    train_size: int = 1000,
    epochs: int = 1,
    output_name: str = "symbolic",
) -> dict:
    """
    Run a 'symbolic' fine-tune for demonstration when no GPU is available.
    
    This doesn't actually train but creates the expected outputs for the pipeline.
    Used for testing the full workflow on CPU-only environments.
    
    Returns:
        Dictionary with symbolic training results
    """
    logger.warning("Running SYMBOLIC fine-tune (no actual training)")
    logger.warning("This is for pipeline testing only. Use GPU for real training.")
    
    output_dir = get_project_root() / "models" / "symbolic" / output_name
    ensure_dir(output_dir)
    
    # Create symbolic output files
    results = {
        "status": "symbolic",
        "message": "No GPU available. This is a symbolic fine-tune for pipeline testing.",
        "config": {
            "train_size": train_size,
            "epochs": epochs,
            "model": "symbolic/no-gpu",
        },
        "output_dir": str(output_dir),
    }
    
    save_yaml(results, output_dir / "training_results.yaml")
    
    # Create a marker file
    (output_dir / "SYMBOLIC_TRAINING.txt").write_text(
        "This model was NOT actually trained.\n"
        "GPU was not available during training.\n"
        "For real results, use a GPU environment (e.g., Google Colab).\n"
    )
    
    return results
