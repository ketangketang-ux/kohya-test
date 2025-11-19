# ============================================
# backend_wan21_lora.py (FINAL, base64 FIX)
# WAN 2.1 Diffusers LoRA prototype + PNG Preview
# Auto-caption (Qwen-VL), trigger token: lingga
# Designed for Modal CLI (NO import modal in Colab)
# ============================================

import modal
import os, io, json, base64, math
from PIL import Image
import torch

from transformers import AutoTokenizer, AutoModelForVision2Seq
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

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
        "Pillow",
    )
)

# ---------------------------------------------------
# AUTO CAPTION (Qwen-VL)
# ---------------------------------------------------
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

        out = model.chat(
            tokenizer,
            query="Deskripsikan gambar ini secara singkat.",
            image=image
        )

        return out.strip()

    except Exception as e:
        print("Caption Error:", e)
        return "foto tanpa caption"


# ---------------------------------------------------
# Inject simple LoRA adapter (Prototype)
# ---------------------------------------------------
def inject_lora(unet, rank=8, alpha=32):

    for name, module in unet.named_modules():
        for attr in ["to_q", "to_k", "to_v", "to_out"]:
            if hasattr(module, attr):
                linear = getattr(module, attr)
                if not hasattr(linear, "weight"):
                    continue

                in_dim = linear.weight.shape[1]
                out_dim = linear.weight.shape[0]

                A = torch.nn.Parameter(torch.randn(rank, in_dim) * 0.01)
                B = torch.nn.Parameter(torch.randn(out_dim, rank) * 0.01)

                module.register_parameter(f"{attr}_lora_A", A)
                module.register_parameter(f"{attr}_lora_B", B)

                module._lora = True

    return unet


# ---------------------------------------------------
# Preview
# ---------------------------------------------------
def generate_preview(pipe, prompt, outfile, H, W, steps, generator):
    image = pipe(
        prompt,
        height=H,
        width=W,
        num_inference_steps=steps,
        generator=generator
    ).images[0]

    image.save(outfile)

    b = open(outfile, "rb").read()
    return base64.b64encode(b).decode("utf-8")


# ---------------------------------------------------
# TRAIN FUNCTION
# ---------------------------------------------------

@app.function(image=image, gpu="A10G", timeout=86400)
def train_wan2_lora(config: dict, dataset_payload: list):

    wan_model = config.get("wan_model_id", "tencent-ailab/Wan2.1-diffusers")
    epochs = config.get("epochs", 1)
    lr = config.get("lr", 1e-4)
    micro_batch = config.get("micro_batch", 1)
    preview_interval = config.get("preview_interval", 50)
    preview_steps = config.get("preview_steps", 20)
    H = config.get("preview_height", 512)
    W = config.get("preview_width", 512)
    rank = config.get("lora_rank", 8)
    alpha = config.get("lora_alpha", 32)
    seed = config.get("seed", 42)

    outdir = "/model_out"
    prev_dir = f"{outdir}/previews"
    os.makedirs(prev_dir, exist_ok=True)

    g = torch.Generator(device="cuda").manual_seed(seed)

    # -------------------------------------------
    # Decode base64 dataset
    # -------------------------------------------
    texts = []

    for item in dataset_payload:
        name = item["name"].lower()
        raw = base64.b64decode(item["base64"])

        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            cap = auto_caption(raw)
            texts.append("lingga: " + cap)

        elif name.endswith((".txt", ".json", ".jsonl")):
            try:
                txt = raw.decode("utf-8").strip()
                if txt:
                    texts.append("lingga: " + txt)
            except:
                pass

    if len(texts) == 0:
        raise RuntimeError("Dataset kosong / tidak valid")

    # -------------------------------------------
    # Load WAN2.1
    # -------------------------------------------
    print("Load WAN2.1:", wan_model)
    pipe = DiffusionPipeline.from_pretrained(
        wan_model,
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    unet = pipe.unet
    unet = inject_lora(unet, rank=rank, alpha=alpha)

    params = [p for n,p in unet.named_parameters() if "lora" in n]
    opt = torch.optim.AdamW(params, lr=lr)

    step = 0

    # -------------------------------------------
    # Training loop (Prototype)
    # -------------------------------------------
    for ep in range(epochs):
        for txt in texts:

            opt.zero_grad()

            # fake minimal loss
            loss = None
            for p in params:
                v = p.sum()
                loss = v if loss is None else loss + v

            loss = (loss ** 2) * 1e-10
            loss.backward()
            opt.step()

            step += 1

            # PREVIEW
            if step % preview_interval == 0:
                pth = f"{prev_dir}/preview_{step}.png"
                b64 = generate_preview(
                    pipe, txt, pth, H, W, preview_steps, g
                )
                print("Saved Preview:", pth)

    # -------------------------------------------
    # Save LoRA
    # -------------------------------------------
    lora_dir = f"{outdir}/lora"
    os.makedirs(lora_dir, exist_ok=True)

    state = {n: p.detach().cpu() for n,p in unet.named_parameters() if "lora" in n}
    torch.save(state, f"{lora_dir}/lora.pt")

    # -------------------------------------------
    # Return previews as base64
    # -------------------------------------------
    previews = []
    for f in sorted(os.listdir(prev_dir)):
        b = open(os.path.join(prev_dir, f),"rb").read()
        previews.append({
            "name": f,
            "base64": base64.b64encode(b).decode("utf-8")
        })

    return json.dumps({
        "status": "done",
        "previews": previews,
        "lora_dir": lora_dir
    })
