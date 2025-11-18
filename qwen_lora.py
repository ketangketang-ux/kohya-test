# qwen_lora_auto_full.py
# ============================================================
# Qwen LoRA Trainer + Auto-Caption (Gemini Vision optional)
# + Preview generation setiap N step
# Modal (2025-compatible)
# ============================================================

import modal
import torch
import os
import io
import json
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from peft import LoraConfig, get_peft_model

# Nama app (sesuaikan jika mau)
app = modal.App("qwen_lora_train_full")

# ----------------------------
# Helper: Gemini Vision (optional)
# ----------------------------
def caption_with_gemini(img_bytes: bytes) -> str:
    """
    Optional: use Gemini Vision if environment is configured.
    To enable:
      - Set env var USE_GEMINI="1"
      - Provide GOOGLE_APPLICATION_CREDENTIALS or mount service account json
      - Install google-cloud-aiplatform in image (see note)
    If not configured, function raises Exception to let caller fallback.
    """
    # NOTE: This is a template. To enable in Modal, you must:
    # 1) pip install google-cloud-aiplatform in modal image
    # 2) set service account credentials in env or mount file
    # 3) set proper endpoint & model id
    # The code below is illustrative; uncomment & adapt when enabling.
    raise RuntimeError("Gemini Vision not configured in runtime.")


# ----------------------------
# Helper: Qwen-VL fallback caption
# ----------------------------
def caption_with_qwen_vl(img_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        vl_model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
        ).eval()
        vl_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
        )
        out = vl_model.chat(vl_tokenizer, query="Deskripsikan gambar ini secara singkat.", image=image)
        return out.strip()
    except Exception as e:
        print("Qwen-VL caption failed:", e)
        return "Gambar tanpa caption."


# ----------------------------
# Unified caption function
# ----------------------------
def generate_caption(img_bytes: bytes) -> str:
    # If user set USE_GEMINI=1, try Gemini first, else fallback to Qwen-VL
    use_gemini = os.environ.get("USE_GEMINI", "0") == "1"
    if use_gemini:
        try:
            return caption_with_gemini(img_bytes)
        except Exception as e:
            print("Gemini caption failed or not configured:", e)
            print("Falling back to Qwen-VL...")
    return caption_with_qwen_vl(img_bytes)


# ----------------------------
# TrainerCallback for preview generation
# ----------------------------
from transformers import TrainerCallback, TrainerControl, TrainerState

class PreviewCallback(TrainerCallback):
    def __init__(self, model, tokenizer, preview_prompt, preview_interval, out_dir, device):
        self.model = model
        self.tokenizer = tokenizer
        self.preview_prompt = preview_prompt
        self.preview_interval = preview_interval
        self.out_dir = out_dir
        self.device = device
        self._step = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # state.global_step might be 0 at start
        self._step = int(state.global_step)
        if self._step > 0 and (self._step % self.preview_interval == 0):
            try:
                prompt = self.preview_prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                # generate (small generation length)
                out_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.8)
                txt = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            except Exception as e:
                txt = f"Preview generation failed: {e}"
            # write to file
            fname = os.path.join(self.out_dir, f"preview_step_{self._step}.txt")
            with open(fname, "w", encoding="utf-8") as fh:
                fh.write(txt)
            # also print to logs so frontend can show it
            print(f"\n=== Preview (step {self._step}) ===\n{txt}\n=== End Preview ===\n")


# ----------------------------
# Main training function (Modal)
# ----------------------------
@app.function(gpu="A10G", timeout=86400)
def train_qwen_lora_full(config: dict, dataset_payload: list):
    """
    config: dict konfigurasi training
    dataset_payload: list of {"name": str, "bytes": bytes}
    """
    print("Starting Qwen LoRA training (auto-caption + preview)...")
    model_name = config.get("model_name", "Qwen2.5-7B")
    preview_prompt = config.get("preview_prompt", "Jelaskan apa itu arwana silver secara singkat.")
    preview_interval = int(config.get("preview_interval", 200))
    out_dir = "/model_out"
    preview_dir = os.path.join(out_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    # ----------------------------
    # Load tokenizer & base model (4-bit)
    # ----------------------------
    print("Loading tokenizer and base model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # ----------------------------
    # Apply LoRA (PEFT)
    # ----------------------------
    print("Applying LoRA config...")
    lora = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----------------------------
    # Build dataset (auto-caption images)
    # ----------------------------
    print("Preparing dataset (auto-caption images + text files)...")
    dataset_texts = []
    for item in dataset_payload:
        name = item.get("name", "").lower()
        data = item.get("bytes", b"")
        if name.endswith((".txt", ".json", ".jsonl")):
            try:
                dataset_texts.append(data.decode("utf-8"))
            except Exception as e:
                print("Failed decode text file", name, e)
                dataset_texts.append("")
        elif name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            try:
                caption = generate_caption(data)
                dataset_texts.append(caption)
            except Exception as e:
                print("Captioning failed for", name, e)
                dataset_texts.append("Gambar tanpa caption.")
        else:
            print("Ignoring unknown file type:", name)

    if len(dataset_texts) == 0:
        raise RuntimeError("No dataset text items prepared. Abort.")

    # convert to simple tokenized list for Trainer
    encodings = []
    max_length = int(config.get("max_length", 1024))
    for txt in dataset_texts:
        enc = tokenizer(txt, truncation=True, padding="max_length", max_length=max_length)
        encodings.append({k: torch.tensor(v) for k, v in enc.items()})

    # ----------------------------
    # Setup Trainer & PreviewCallback
    # ----------------------------
    print("Configuring Trainer...")
    from transformers import TrainingArguments, Trainer, default_data_collator

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(config.get("micro_batch", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accum", 8)),
        learning_rate=float(config.get("lr", 2e-4)),
        num_train_epochs=float(config.get("epochs", 1)),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    preview_cb = PreviewCallback(model=model, tokenizer=tokenizer,
                                 preview_prompt=preview_prompt, preview_interval=preview_interval,
                                 out_dir=preview_dir, device=device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encodings,
        data_collator=default_data_collator,
        callbacks=[preview_cb],
    )

    # ----------------------------
    # Train
    # ----------------------------
    print("Starting trainer.train() ...")
    trainer.train()
    print("Trainer finished. Saving LoRA...")

    # Save
    save_dir = os.path.join(out_dir, "lora")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Saved LoRA to:", save_dir)

    # list previews
    previews = sorted(os.listdir(preview_dir))
    print("Available previews:", previews)

    # Return a simple JSON-like summary (Modal will return as string)
    summary = {
        "status": "done",
        "lora_path": save_dir,
        "previews": previews,
    }
    return json.dumps(summary)
