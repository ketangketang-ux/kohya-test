import modal
import torch
import os, io, json
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from peft import LoraConfig, get_peft_model

app = modal.App("qwen_lora_train_full")

# ----------------------------
# Qwen-VL Caption
# ----------------------------
def caption_image(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        ).eval()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        out = model.chat(tok, query="Deskripsikan gambar ini secara singkat.", image=image)
        return out.strip()
    except:
        return "Gambar tanpa caption."

# ----------------------------
# Preview callback
# ----------------------------
from transformers import TrainerCallback

class PreviewCB(TrainerCallback):
    def __init__(self, model, tokenizer, interval, prompt, out_dir, device):
        self.model = model
        self.tokenizer = tokenizer
        self.interval = interval
        self.prompt = prompt
        self.out_dir = out_dir
        self.device = device
        os.makedirs(self.out_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.interval == 0 and step > 0:
            inp = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inp, max_new_tokens=150)
            txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
            fname = os.path.join(self.out_dir, f"preview_{step}.txt")
            with open(fname, "w") as f:
                f.write(txt)
            print(f"\n=== Preview step {step} ===\n{txt}\n==========================\n")

# ----------------------------
# MAIN TRAIN FUNCTION
# ----------------------------
@app.function(gpu="A10G", timeout=86400)
def train_qwen_lora_full(config: dict, dataset_payload: list):
    print("Running backend…")

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    device = torch.device("cuda")

    # ---------------- DATASET ----------------
    texts = []
    for item in dataset_payload:
        n = item["name"].lower()
        b = item["bytes"]

        if n.endswith((".jpg",".png",".webp",".jpeg")):
            caption = caption_image(b)
            texts.append(caption)
        else:
            texts.append(b.decode("utf-8"))

    encs = []
    for t in texts:
        enc =
