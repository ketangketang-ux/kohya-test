# ============================================================
# BACKEND MODAL - QWEN LoRA Training (2025 compatible)
# ============================================================

import modal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json, os, io

app = modal.App("qwen_lora_train_full")


@app.function(gpu="A10G", timeout=86400)
def train_qwen_lora_full(config: dict, dataset_payload: list):
    print("Starting QLoRA training backend...")

    # Load model
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    print("Preparing dataset...")
    dataset_texts = []

    for item in dataset_payload:
        name = item["name"]
        data = item["bytes"]

        if name.lower().endswith((".txt", ".json", ".jsonl")):
            dataset_texts.append(data.decode("utf-8"))

        elif name.lower().endswith((".jpg", ".png", ".webp", ".jpeg")):
            # SIMPLE caption placeholder (nanti bisa ditambah Qwen-VL)
            dataset_texts.append(f"Caption for image {name}")

    # Build dataset
    dataset = [{"text": t} for t in dataset_texts]

    def tokenize_fn(e):
        return tokenizer(
            e["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
        )

    tokenized = list(map(tokenize_fn, dataset))

    print("Starting training...")

    from transformers import TrainingArguments, Trainer

    args = TrainingArguments(
        output_dir="/model_out",
        per_device_train_batch_size=config["micro_batch"],
        gradient_accumulation_steps=config["gradient_accum"],
        learning_rate=config["lr"],
        num_train_epochs=config["epochs"],
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained("/model_out/lora")

    print("\n=== Training finished ===")
    return "Training selesai. LoRA ada di /model_out/lora"
