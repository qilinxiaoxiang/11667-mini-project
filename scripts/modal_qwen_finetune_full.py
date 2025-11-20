"""
Full Fine-tuning script for Qwen2.5-0.5B on Modal
- Model: Qwen/Qwen2.5-0.5B
- Dataset: Hierarchical medical responses (3,311 samples, up to 768 tokens)
- GPU: A100-80GB required (seq_len² scaling: 768 tokens uses more memory)
- Batch Size: 16 (effective 32 with gradient accumulation)
- Training Time: ~12-15 minutes per epoch
- Memory Usage: ~55-65 GB (full GPU utilization)
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import load_from_disk
import modal
from modal import Volume


class EpochLossCallback(TrainerCallback):
    """Print loss at the end of each epoch"""
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        # Get the last training loss for this epoch
        train_loss = state.log_history[-1].get("loss")
        eval_loss = state.log_history[-1].get("eval_loss")

        print(f"\n{'='*60}")
        print(f"Epoch {int(epoch)} Summary:")
        print(f"{'='*60}")
        if train_loss is not None:
            print(f"  Train Loss: {train_loss:.4f}")
        if eval_loss is not None:
            print(f"  Eval Loss:  {eval_loss:.4f}")
        print(f"{'='*60}\n")

# Create Modal app and volumes
app = modal.App("medical-response-finetuning-qwen")

# Create volumes for dataset, models, and checkpoints
dataset_volume = Volume.from_name("medical-dataset-volume", create_if_missing=True)
model_volume = Volume.from_name("medical-models-volume", create_if_missing=True)

# Docker image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
)


def print_memory_info(device):
    """Print GPU memory information"""
    if device.type == "cuda":
        print("\n" + "=" * 60)
        print("GPU Memory Information")
        print("=" * 60)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Allocated: {allocated:.2f}GB")
        print(f"Reserved: {reserved:.2f}GB")
        print(f"Total GPU: {total:.2f}GB")
        print(f"Available: {(total - reserved):.2f}GB")
        print("=" * 60 + "\n")


def preprocess_function(examples, tokenizer, max_length=768):
    """
    Tokenize and prepare data for training with input masking.

    This masks the input prompt (Question + Background) so the model
    only learns to predict the Doctor Response (the hierarchical part).
    """
    input_ids_list = []
    labels_list = []

    for i in range(len(examples["Description"])):
        # Build the prompt (input that we DON'T want to train on)
        prompt = (
            f"Medical Question: {examples['Description'][i]}\n"
            f"Patient Background: {examples['Patient'][i]}\n"
            f"Doctor Response:\n"
        )

        # Build the full text (prompt + response)
        response = examples['Doctor'][i]
        full_text = prompt + response

        # Tokenize the full sequence
        full_encoded = tokenizer(
            full_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = full_encoded["input_ids"]

        # Tokenize ONLY the prompt to find where it ends
        prompt_encoded = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            return_tensors=None,
        )
        prompt_length = len(prompt_encoded["input_ids"])

        # Create labels: copy input_ids initially
        labels = input_ids.copy()

        # Mask the prompt tokens (set to -100 so they don't contribute to loss)
        labels[:prompt_length] = [-100] * prompt_length

        # Also mask padding tokens
        for j in range(len(labels)):
            if input_ids[j] == tokenizer.pad_token_id:
                labels[j] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    # Return as dictionary with proper structure
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": [[1 if token_id != tokenizer.pad_token_id else 0
                           for token_id in ids] for ids in input_ids_list],
    }


def finetune_qwen(
    epochs=5,
    batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=100,
    max_seq_length=768,
):
    """
    Full fine-tune Qwen2.5-0.5B on hierarchical medical dataset

    Memory Estimation (with batch_size=16, seq_length=768):
    - Model weights (FP16): 0.93 GB
    - Gradients (FP16): 0.93 GB
    - Optimizer states (AdamW): 3.72 GB
    - Activations (attention: 16 × 14 heads × 768²): ~4-5 GB (quadratic with seq_len)
    - Total: ~55-65 GB (full GPU utilization on A100-80GB)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print_memory_info(device)

    # Load dataset
    print("Loading dataset from /dataset/hierarchical_dataset_clean...")
    dataset = load_from_disk("/dataset/hierarchical_dataset_clean")

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample fields: {dataset.column_names}")

    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Eval set: {len(eval_dataset)} samples")

    # Load tokenizer and model
    print("\nLoading Qwen2.5-0.5B model...")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded: {model.config}")
    print(f"Number of parameters: {model.num_parameters():,}")

    # Preprocess datasets
    print("\nPreprocessing datasets...")
    preprocess = lambda examples: preprocess_function(
        examples, tokenizer, max_seq_length
    )
    train_dataset = train_dataset.map(
        preprocess, batched=True, remove_columns=dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess, batched=True, remove_columns=dataset.column_names
    )

    print_memory_info(device)

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/models/qwen_full_{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,  # Load best model checkpoint at end
        metric_for_best_model="eval_loss",  # Track eval loss for best model
        greater_is_better=False,  # Lower loss is better
        logging_dir=f"{output_dir}/logs",
        fp16=False,  # Use FP32 to avoid FP16 + gradient clipping conflict
        bf16=False,  # Disable BF16 as well
        max_grad_norm=1.0,  # Standard gradient clipping (safe with FP32)
        save_total_limit=3,
        push_to_hub=False,
        report_to=[],  # Disable wandb/tensorboard reporting
        disable_tqdm=True,  # Reduce progress bar clutter
    )

    print(f"\nTraining Configuration:")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  FP16 precision: DISABLED (using FP32)")
    print(f"  Gradient clipping: ENABLED (max_norm=1.0)")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            EpochLossCallback(),
            EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001),
        ],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Print final memory usage
    print_memory_info(device)

    # Save training summary
    summary = {
        "model": "Qwen/Qwen2.5-0.5B",
        "training_type": "Full Fine-tuning",
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_seq_length": max_seq_length,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "output_dir": output_dir,
        "timestamp": timestamp,
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

    return summary


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={
        "/dataset": dataset_volume,
        "/models": model_volume,
    },
    timeout=86400,
)
def train(
    epochs=5,
    batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=100,
):
    """Entrypoint for modal run command"""
    return finetune_qwen(
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
    )
