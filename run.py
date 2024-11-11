import modal
import os
import sys
import shlex

# 使用 App 替代 Stub
app = modal.App("stable-diffusion-webui")

# 创建持久化存储，使用新的 API
volume = modal.NetworkFileSystem.from_name("stable-diffusion-webui-data", create_if_missing=True)

def download_models():
    models = [
        ("counterfeitxl_v10.safetensors", "https://huggingface.co/ckpt/counterfeit-xl/resolve/main/counterfeitxl_v10.safetensors"),
        ("juggernautXL_version2.safetensors", "https://huggingface.co/ckpt/juggernaut-xl/resolve/main/juggernautXL_version2.safetensors"),
        ("sd_xl_refiner_1.0_0.9vae.safetensors", "https://huggingface.co/ckpt/sd_xl_refiner_1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors")
    ]
    
    os.makedirs("/content/stable-diffusion-webui/models/Stable-diffusion", exist_ok=True)
    for model_name, url in models:
        output_path = f"/content/stable-diffusion-webui/models/Stable-diffusion/{model_name}"
        if not os.path.exists(output_path):
            os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {url} -d /content/stable-diffusion-webui/models/Stable-diffusion -o {model_name}")

@app.function(
    image=modal.Image.from_registry("nvidia/cuda:12.2.0-base-ubuntu22.04", add_python="3.11")
    .run_commands([
        "apt update -y",
        "apt install -y software-properties-common",
        "add-apt-repository -y ppa:git-core/ppa",
        "apt update -y",
        "apt install -y git git-lfs aria2 libgl1 libglib2.0-0 wget",
        "pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118",
        "pip install -q xformers==0.0.20 triton==2.0.0 packaging==23.1"
    ]),
    network_file_systems={"/content/stable-diffusion-webui": volume},
    gpu="T4",
    timeout=60000,
)
def run():
    # Clone repository if it doesn't exist
    if not os.path.exists("/content/stable-diffusion-webui/.git"):
        os.system("git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui /content/stable-diffusion-webui")
    
    os.chdir("/content/stable-diffusion-webui")
    os.system("git reset --hard")
    
    # Download models
    download_models()
    
    # Set environment variables
    os.environ['HF_HOME'] = '/content/stable-diffusion-webui/cache/huggingface'
    
    # Set up system path and arguments
    sys.path.append('/content/stable-diffusion-webui')
    sys.argv = shlex.split("--cors-allow-origins=* --xformers --theme dark --gradio-debug --share")
    
    # Import and start the WebUI
    from modules import launch_utils
    launch_utils.startup_timer.record("initial startup")
    launch_utils.prepare_environment()
    launch_utils.start()

@app.local_entrypoint()
def main():
    run.remote()
