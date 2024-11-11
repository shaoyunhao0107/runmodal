import modal, os, sys, shlex

# 创建一个 Modal 应用的存根
stub = modal.Stub("stable-diffusion-webui")

# 使用 from_name 创建持久化的网络文件系统
volume = modal.NetworkFileSystem.from_name("stable-diffusion-webui")

@stub.function(
    image=modal.Image.from_registry("nvidia/cuda:12.2.0-base-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt update -y && \
        apt install -y software-properties-common && \
        apt update -y && \
        add-apt-repository -y ppa:git-core/ppa && \
        apt update -y && \
        apt install -y git git-lfs && \
        git --version  && \
        apt install -y aria2 libgl1 libglib2.0-0 wget && \
        pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip install -q xformers==0.0.20 triton==2.0.0 packaging==23.1"
    ),
    network_file_systems={"/content/stable-diffusion-webui": volume},
    gpu="T4",
    timeout=60000,
)
async def run():
    # 克隆 stable-diffusion-webui 仓库
    os.system(f"git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui /content/stable-diffusion-webui")
    os.chdir(f"/content/stable-diffusion-webui")

    # 重置仓库
    os.system(f"git reset --hard")

    # 下载模型文件
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/counterfeit-xl/resolve/main/counterfeitxl_v10.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o counterfeitxl_v10.safetensors")
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/juggernaut-xl/resolve/main/juggernautXL_version2.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o juggernautXL_version2.safetensors")
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd_xl_refiner_1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o sd_xl_refiner_1.0_0.9vae.safetensors")

    # 设置 Hugging Face 缓存目录
    os.environ['HF_HOME'] = '/content/stable-diffusion-webui/cache/huggingface'

    # 准备环境并启动 Stable Diffusion Web UI
    sys.path.append('/content/stable-diffusion-webui')
    sys.argv = shlex.split("--cors-allow-origins=* --xformers --theme dark --gradio-debug --share")
    from modules import launch_utils
    launch_utils.startup_timer.record("initial startup")
    launch_utils.prepare_environment()
    launch_utils.start()

# 在本地入口点中运行异步任务
@stub.local_entrypoint()
def main():
    run.remote()
