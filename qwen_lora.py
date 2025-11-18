# =========================================
# BACKEND MODAL → QLoRA Training Qwen
# =========================================
import modal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json, os

stub = modal.Stub("qwen_lora_train")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "datasets",
        "sentencepiece",
    )
)

@stub.function(image=image, gpu="A10G")
def train_lora_qwen(config: dict):
    model_name = config["model_name"]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # DATASET LOAD
    # Modal gives file paths via /modal/...
    dataset_files = config["dataset_files"]
    ds = load_dataset("json", data_files=[f"/modal/{f}" for f in dataset_files])

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=2048,
        )

    ds = ds.map(tokenize_fn)

    # Training
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="/model_out",
        per_device_train_batch_size=config["micro_batch"],
        gradient_accumulation_steps=config["gradient_accum"],
        learning_rate=config["lr"],
        num_train_epochs=config["epochs"],
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
    )

    trainer.train()

    # Save LoRA
    model.save_pretrained("/model_out/lora")

    print("Training selesai. Model disimpan di /model_out/lora")
