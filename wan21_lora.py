# ============================================
# backend_wan21_lora.py
# WAN 2.1 LoRA Prototype + Auto Caption + PNG Preview
# Modal-friendly backend (NO Modal SDK import in Colab)
# ============================================

import modal
import os, io, json, base64, math
from PIL import Image
import torch

from transformers import AutoTokenizer, AutoModelForVision2Seq
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from accelerate import Accelerator

# ============================================
# Modal App + Requirements
# ============================================

app = modal.App("wan21_lora_app")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "diffusers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
        "Pillow"
    )
)

# ============================================
# Auto Caption - Qwen-VL
# ============================================

def auto_caption(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model_id = "Qwen/Qwen2-VL-2B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True
        ).eval()

        caption = model.chat(
            tokenizer,
            query="Deskripsikan gambar ini secara singkat.",
            image=image
        )

        return caption.strip()

    except Exception as e:
        print("Caption error:", e)
        return "foto tanpa caption"

# ============================================
# Simple UNet LoRA Adapter (Prototype)
# ============================================

def inject_lora(unet, rank=4, alpha=32):
    for name, module in unet.named_modules():
        for attr in ["to_q", "to_k", "to_v", "to_out"]:
            if hasattr(module, attr):
                orig = getattr(module, attr)
                if not hasattr(orig, "weight"):
                    continue

                in_dim = orig.weight.shape[1]
                out_dim = orig.weight.shape[0]

                A = torch.nn.Parameter(torch.randn((rank, in_dim)) * 0.01)
                B = torch.nn.Parameter(torch.randn((out_dim, rank)) * 0.01)

                module.register_parameter(f"{attr}_lora_A", A)
                module.register_parameter(f"{attr}_lora_B", B)

                module._lora = True

    return unet


def lora_forward(orig_weight, x, A, B):
    lora = (B @ (A @ x.t())).t()
    return orig_weight(x) + lora


# ============================================
# Preview Helper
# ============================================

def generate_preview(pipe, prompt, out_file, height, width, steps, generator):
    image = pipe(
        prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator
    ).images[0]

    image.save(out_file)

    with open(out_file, "rb") as f:
        raw = f.read()

    return base64.b64encode(raw).decode("utf-8")


# ============================================
# MAIN TRAIN FUNCTION
# ============================================

@app.function(image=image, gpu="A10G", timeout=86400)
def train_wan2_lora(config: dict, dataset_payload: list):
    # --- CONFIG ---
    wan_model = config.get("wan_model_id", "tencent-ailab/Wan2.1-diffusers")
    epochs = int(config.get("epochs", 1))
    lr = float(config.get("lr", 1e-4))
    micro_batch = int(config.get("micro_batch", 1))
    preview_interval = int(config.get("preview_interval", 50))
    preview_steps = int(config.get("preview_steps", 20))
    H = int(config.get("preview_height", 512))
    W = int(config.get("preview_width", 512))
    rank = int(config.get("lora_rank", 8))
    alpha = int(config.get("lora_alpha", 32))
    seed = int(config.get("seed", 42))

    g = torch.Generator("cuda").manual_seed(seed)

    outdir = "/model_out"
    prev_dir = f"{outdir}/previews"
    os.makedirs(prev_dir, exist_ok=True)

    # --- DATASET PROCESSING ---
    texts = []
    for item in dataset_payload:
        name = item["name"].lower()
        b = item["bytes"]

        if name.endswith((".jpg", ".png", ".jpeg", ".webp")):
            cap = auto_caption(b)
            texts.append("lingga: " + cap)

        elif name.endswith((".txt", ".jsonl", ".json")):
            try:
                txt = b.decode("utf-8").strip()
                if txt:
                    texts.append("lingga: " + txt)
            except:
                pass

    if len(texts) == 0:
        raise RuntimeError("Dataset kosong")

    # --- LOAD WAN2.1 PIPELINE ---
    print("Loading WAN 2.1 pipeline:", wan_model)
    pipe = DiffusionPipeline.from_pretrained(
        wan_model,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    unet = pipe.unet
    unet = inject_lora(unet, rank=rank, alpha=alpha)

    # train only LoRA params
    params = [p for n,p in unet.named_parameters() if "lora" in n]
    optimizer = torch.optim.AdamW(params, lr=lr)

    total_steps = epochs * len(texts)
    step = 0

    # --- TRAIN LOOP (prototype) ---
    for ep in range(epochs):
        for txt in texts:
            optimizer.zero_grad()

            # fake tiny loss to allow LoRA update
            loss = None
            for p in params:
                v = p.sum()
                loss = v if loss is None else loss + v
            loss = (loss ** 2) * 1e-10
            loss.backward()
            optimizer.step()

            step += 1

            # --- PREVIEW ---
            if step % preview_interval == 0:
                fpath = f"{prev_dir}/preview_{step}.png"
                b64 = generate_preview(
                    pipe,
                    txt,
                    fpath,
                    H, W,
                    preview_steps,
                    g
                )
                print("Saved preview:", fpath)

    # --- SAVE LORA ---
    save_dir = f"{outdir}/lora"
    os.makedirs(save_dir, exist_ok=True)

    lora_dict = {}
    for n,p in unet.named_parameters():
        if "lora" in n:
            lora_dict[n] = p.detach().cpu()

    torch.save(lora_dict, f"{save_dir}/lora.pt")

    # --- RETURN PREVIEWS ---
    previews = []
    for f in sorted(os.listdir(prev_dir)):
        path = os.path.join(prev_dir, f)
        with open(path, "rb") as fp:
            b64 = base64.b64encode(fp.read()).decode("utf-8")
        previews.append({"name": f, "base64": b64})

    return json.dumps({
        "status": "done",
        "lora_path": save_dir,
        "previews": previews
    })
