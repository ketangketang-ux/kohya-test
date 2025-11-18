# ==========================
# kohya_modal.py
# Kohya-SS backend di Modal + auto rename & caption
# ==========================
import os
import subprocess
import modal
from huggingface_hub import hf_hub_download
import shutil
from transformers import pipeline, AutoTokenizer

# ---------- CONFIG ----------
DATA_ROOT = "/data/kohya"
KOHYA_DIR = os.path.join(DATA_ROOT, "kohya_ss")
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "A100-40GB")
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # isi token HF kalau pakai model gated

# Caption & rename pipeline (bisa ganti model)
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# ---------- IMAGE ----------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "build-essential", "python3-tk", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands([
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install transformers accelerate huggingface_hub[hf_transfer]",
        "pip install -U -r https://raw.githubusercontent.com/kohya-ss/kohya_ss/master/requirements.txt",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------- VOLUME ----------
vol = modal.Volume.from_name("kohya-app", create_if_missing=True)
app = modal.App(name="kohya-modal", image=image)

# ---------- APP FUNCTION ----------
@app.function(
    gpu=GPU_TYPE,
    timeout=7200,
    volumes={DATA_ROOT: vol},
)
def kohya_backend():
    # 1. Clone / update Kohya-SS
    if not os.path.exists(os.path.join(KOHYA_DIR, "train_network.py")):
        print("⬇️ Clone Kohya-SS ...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/kohya-ss/kohya_ss.git", KOHYA_DIR
        ], check=True)
    os.chdir(KOHYA_DIR)
    subprocess.run("git pull --ff-only", shell=True, check=False)

    # 2. Download model latihan (SDXL-Lightning 4-step)
    print("⬇️ Download SDXL-Lightning ...")
    model_dir = os.path.join(DATA_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    hf_hub_download(
        repo_id="ByteDance/SDXL-Lightning",
        filename="sdxl_lightning_4step.safetensors",
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )

    # 3. Siapkan folder dataset (bisa di-mount atau upload)
    dataset_dir = os.path.join(DATA_ROOT, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # 4. Auto-rename & caption setiap gambar
    print("📝 Auto caption & rename ...")
    idx = 1
    for file in os.listdir(dataset_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            old_path = os.path.join(dataset_dir, file)
            caption = captioner(old_path)[0]["generated_text"].replace(" ", "_")[:50]
            new_name = f"{idx:04d}_{caption}.jpg"
            new_path = os.path.join(dataset_dir, new_name)
            shutil.move(old_path, new_path)

            # buat txt caption (format Kohya)
            with open(new_path.replace(".jpg", ".txt"), "w", encoding="utf-8") as f:
                f.write(caption.replace("_", " "))
            idx += 1

    # 5. Buat config minimal
    config_toml = os.path.join(DATA_ROOT, "config.toml")
    with open(config_toml, "w") as f:
        f.write(f"""
pretrained_model_name_or_path = "{model_dir}/sdxl_lightning_4step.safetensors"
train_data_dir = "{dataset_dir}"
output_dir = "{DATA_ROOT}/output"
resolution = 1024
batch_size = 1
max_train_epochs = 3
save_every_n_epochs = 1
lr_scheduler = "constant_with_warmup"
learning_rate = 1e-5
warmup_steps = 100
optimizer_type = "AdamW8bit"
mixed_precision = "fp16"
xformers = true
cache_latents = true
        """)

    # 6. Latih
    print("🏋️ Start training ...")
    subprocess.run([
        "accelerate", "launch", "--num_cpu_threads_per_process", "2",
        "train_network.py",
        "--config_file", config_toml
    ], check=True)

    # 7. Hasil
    output_file = os.path.join(DATA_ROOT, "output", "last.safetensors")
    if os.path.exists(output_file):
        print("✅ Training selesai →", output_file)
        return output_file
    else:
        raise RuntimeError("❌ Output model tidak ditemukan")


@app.local_entrypoint()
def main():
    result = kohya_backend.remote()
    print("Model tersimpan di volume:", result)
