import modal

app = modal.App("agentgate-pretraining")

volume = modal.Volume.from_name(
    "agentgate-storage",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "numpy",
        "datasets",
        "tokenizers",
        "tqdm",
    )
    .add_local_python_source("src")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 12,  # 12 hours
    volumes={
        "/storage": volume,
    },
)
def pretrain():

    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    from src.training.trainer import train

    train(
        checkpoint_dir="/storage/checkpoints",
        log_dir="/storage/logs",
    )


@app.local_entrypoint()
def main():
    pretrain.remote()
