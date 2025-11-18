# ====================================================
# MODAL BACKEND
# - Auto Caption Qwen
# - LoRA Training
# - Sample Output Generator
# ====================================================

import modal
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from PIL import Image
import json, os, io

stub = modal.Stub("qwen_lora_train_full")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "datasets",
        "sentencepiece",
        "Pillow",
    )
)

# ====================================================
# AUTO CAPTION (gunakan Qwen2-VL lightweight)
# ====================================================
def auto_caption_image(img_path):
    model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    image = Image.open(img_path)
    query = "Deskripsikan gambar ini secara singkat."
    res = model.chat(tok, query=query, image=image)
    return res.strip()


# ====================================================
# TRAINING FUNCTION
# ====================================================
@stub.function(image=image, gpu="A10G")
def train_qwen_lora_full(config: dict):
    model_name = config["model_name"]
    preview_prompt = config["preview_prompt"]
    preview_interval = config["preview_interval"]

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
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # Load dataset files
    dataset = []
    for fid in config["dataset_files"]:
        path = f"/modal/{fid}"

        if any(path.endswith(ext) for ext in ["png", "jpg", "jpeg", "webp"]):
            caption = auto_caption_image(path)
            dataset.append({"text": caption})
        else:
            # text file
            with open(path, "r") as f:
                dataset.append({"text": f.read()})

    # Save dataset to JSON
    with open("/tmp/data.jsonl", "w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    ds = load_dataset("json", data_files="/tmp/data.jsonl")

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=2048,
        )

    ds = ds.map(tokenize_fn)

    # Trainer
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="/model_out",
        per_device_train_batch_size=config["micro_batch"],
        gradient_accumulation_steps=config["gradient_accum"],
        learning_rate=config["lr"],
        num_train_epochs=config["epochs"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    step_counter = 0

    def custom_callback(trainer, state, control):
        nonlocal step_counter
        step_counter += 1

        if step_counter % preview_interval == 0:
            print("\n=== Generating preview sample ===\n")
            out = model.generate(
                **tokenizer(preview_prompt, return_tensors="pt").to(model.device),
                max_new_tokens=200,
            )
            txt = tokenizer.decode(out[0], skip_special_tokens=True)
            print(txt)
            print("\n=== End of preview ===\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        callbacks=[custom_callback],
    )

    trainer.train()
    model.save_pretrained("/model_out/lora")

    print("Training selesai. LoRA saved di /model_out/lora")
