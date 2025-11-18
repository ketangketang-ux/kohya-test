# ============================================================
# Qwen LoRA Trainer + Auto Caption (Modal 2025 Compatible)
# ============================================================

import modal
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForVision2Seq,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
import json
import io
import os


# Nama App
app = modal.App("qwen_lora_train_full")


# ============================================================
# AUTO CAPTION FUNCTION (Qwen-VL 2B)
# ============================================================
def generate_caption(img_bytes):
    """
    Auto caption gambar menggunakan Qwen2-VL-2B-Instruct.
    """
    try:
        print("Auto-captioning image...")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        vl_model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        ).eval()

        vl_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )

        caption = vl_model.chat(
            vl_tokenizer,
            query="Deskripsikan gambar ini secara singkat.",
            image=image
        )

        return caption.strip()

    except Exception as e:
        print("Caption failed:", e)
        return "Gambar tanpa caption."


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
@app.function(gpu="A10G", timeout=86400)
def train_qwen_lora_full(config: dict, dataset_payload: list):
    """
    config: dict konfigurasi training
    dataset_payload: list of {name, bytes}
    """
    print("Starting Qwen LoRA Backend (with auto caption)...")

    model_name = config["model_name"]

    # Load tokenizer + model
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # ============================================================
    # BUILD DATASET (AUTO CAPTION)
    # ============================================================
    print("Preparing dataset with auto caption...")

    dataset_texts = []

    for item in dataset_payload:
        name = item["name"].lower()
        data = item["bytes"]

        # ====== TEXT FILES ======
        if name.endswith((".txt", ".json", ".jsonl")):
            try:
                dataset_texts.append(data.decode("utf-8"))
            except:
                dataset_texts.append("")

        # ====== IMAGE FILES ======
        elif name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            cap = generate_caption(data)
            dataset_texts.append(cap)

        else:
            print(f"Unknown file type (ignored): {name}")

    # convert ke dataset
    dataset = [{"text": t} for t in dataset_texts]

    # tokenize function
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=1024,
        )

    tokenized_dataset = list(map(tokenize_fn, dataset))

    # ============================================================
    # TRAINING
    # ============================================================
    print("Starting training...\n")

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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # Save LoRA weights
    model.save_pretrained("/model_out/lora")

    print("\n=== TRAINING SELESAI ===")
    return "Training selesai. LoRA saved → /model_out/lora"
