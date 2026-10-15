import json
import os
import time

import modal

app = modal.App("agentgate-pretraining")

volume = modal.Volume.from_name("agentgate-storage", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "numpy")
    .add_local_python_source("src")
)


DATA_DIR = "/storage/pretraining_data"
RUN_DIR = "/storage/runs/actiongate_base_142m"

BATCH_SIZE = 4
CONTEXT_LENGTH = 2048
GRAD_ACCUM_STEPS = 16
SESSION_BOUNDARY = 10_000

TOKENS_PER_STEP = BATCH_SIZE * CONTEXT_LENGTH * GRAD_ACCUM_STEPS

MAX_STEPS = 20_599
WARMUP_STEPS = 206

PEAK_LR = 6e-4
MIN_LR = 6e-5

L4_PEAK_FLOPS = 121e12
L4_PRICE_PER_HOUR = 0.80

FULL_TRAIN_TOKENS = 2_699_952_128

BENCHMARK_STEPS = 20
BENCHMARK_MEASURE_FROM = 10


@app.function(image=image, gpu="L4", timeout=60 * 30, volumes={"/storage": volume})
def test_run():
    import torch

    from src.data.loader import BinaryShardLoader
    from src.model import AgentGate
    from src.training import CosineWarmupScheduler, build_optimizer

    device = torch.device("cuda")

    model = AgentGate().to(device)

    parameter_count = model.num_parameters()

    if parameter_count != 142_408_320:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")

    train_loader = BinaryShardLoader(
        data_dir=DATA_DIR,
        split="train",
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    optimizer = build_optimizer(
        model, lr=PEAK_LR, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=5,
        total_steps=BENCHMARK_STEPS,
        peak_lr=PEAK_LR,
        min_lr=MIN_LR,
    )

    # Step-0 sanity check without consuming training data.
    loader_state = train_loader.state_dict()

    x, y = train_loader.next_batch()

    train_loader.load_state_dict(loader_state)

    x = x.to(device)
    y = y.to(device)

    model.eval()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, initial_loss = model(x, y)

    initial_loss = initial_loss.item()

    model.train()

    norm_param_count = sum(
        p.numel() for name, p in model.named_parameters() if "norm" in name
    )

    matmul_param_count = parameter_count - norm_param_count

    config = model.config

    flops_per_token = (
        6 * matmul_param_count
        + 12
        * config.n_layers
        * config.n_heads
        * config.head_dim
        * config.context_length
    )

    torch.cuda.reset_peak_memory_stats(device)

    measured_time = 0.0
    measured_tokens = 0

    last_loss = None

    for step in range(1, BENCHMARK_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)

            total_loss += loss.detach()

            (loss / GRAD_ACCUM_STEPS).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        mean_loss = total_loss / GRAD_ACCUM_STEPS

        if not (torch.isfinite(mean_loss) & torch.isfinite(grad_norm)).item():
            raise FloatingPointError("Non-finite loss or gradient detected")

        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()

        step_time = time.perf_counter() - start

        last_loss = mean_loss.item()

        if step > BENCHMARK_MEASURE_FROM:
            measured_time += step_time
            measured_tokens += TOKENS_PER_STEP

    tokens_per_second = measured_tokens / measured_time

    mfu = flops_per_token * tokens_per_second / L4_PEAK_FLOPS

    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9

    estimated_hours = FULL_TRAIN_TOKENS / tokens_per_second / 3600

    estimated_gpu_cost = estimated_hours * L4_PRICE_PER_HOUR

    result = {
        "parameters": parameter_count,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "tokens_per_step": TOKENS_PER_STEP,
        "initial_loss": initial_loss,
        "final_benchmark_loss": last_loss,
        "tokens_per_second": tokens_per_second,
        "mfu": mfu,
        "peak_memory_gb": peak_memory_gb,
        "estimated_full_train_hours": estimated_hours,
        "estimated_l4_gpu_cost": estimated_gpu_cost,
    }

    print(json.dumps(result, indent=2))

    return result


@app.function(image=image, gpu="L4", timeout=24 * 60 * 60, volumes={"/storage": volume})
def pretrain():
    import torch

    from src.data.loader import BinaryShardLoader
    from src.model import AgentGate
    from src.training import (
        CosineWarmupScheduler,
        Trainer,
        TrainingConfig,
        build_optimizer,
    )

    device = torch.device("cuda")

    model = AgentGate().to(device)

    if model.num_parameters() != 142_408_320:
        raise RuntimeError(f"Unexpected parameter count: {model.num_parameters()}")

    train_loader = BinaryShardLoader(
        data_dir=DATA_DIR,
        split="train",
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    val_loader = BinaryShardLoader(
        data_dir=DATA_DIR,
        split="val",
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    optimizer = build_optimizer(
        model, lr=PEAK_LR, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=MAX_STEPS,
        peak_lr=PEAK_LR,
        min_lr=MIN_LR,
    )

    training_config = TrainingConfig(
        max_steps=MAX_STEPS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        tokens_per_step=TOKENS_PER_STEP,
        grad_clip=1.0,
        log_every=10,
        eval_every=400,
        subset_eval_batches=250,
        checkpoint_every=400,
        peak_flops=L4_PEAK_FLOPS,
        price_per_hour=L4_PRICE_PER_HOUR,
        out_dir=RUN_DIR,
    )

    metadata_path = os.path.join(DATA_DIR, "metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        data_metadata = json.load(f)

    run_config = {
        "optimizer": {
            "name": "AdamW",
            "peak_lr": PEAK_LR,
            "min_lr": MIN_LR,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
        "scheduler": {
            "name": "cosine_with_linear_warmup",
            "warmup_steps": WARMUP_STEPS,
            "total_steps": MAX_STEPS,
        },
        "data": data_metadata,
        "precision": "bf16",
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        run_config=run_config,
        persist_fn=volume.commit,
    )

    latest_checkpoint = os.path.join(RUN_DIR, "checkpoints", "latest.pt")

    if os.path.exists(latest_checkpoint):
        trainer.resume(latest_checkpoint)

    if trainer.step < SESSION_BOUNDARY:
        session_end_step = SESSION_BOUNDARY
    else:
        session_end_step = MAX_STEPS

    return trainer.train(session_end_step=session_end_step)


@app.local_entrypoint()
def main():
    call = pretrain.spawn()
    print(f"Training started: {call.object_id}")
