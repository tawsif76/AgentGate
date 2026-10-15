import os
import random

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, out_dir="checkpoints"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _unwrap_model(self, model):
        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        return model

    def save(
        self,
        filename,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        train_loader=None,
        step=0,
        tokens_seen=0,
        latest_val_loss=None,
        best_val_loss=float("inf"),
        training_compute_seconds=0.0,
        wall_gpu_seconds=0.0,
        model_args=None,
        config=None,
    ):
        temp_path = os.path.join(self.out_dir, f"{filename}.tmp")
        final_path = os.path.join(self.out_dir, filename)

        raw_model = self._unwrap_model(model)

        checkpoint = {
            "model": {k: v.cpu() for k, v in raw_model.state_dict().items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "train_loader": (
                train_loader.state_dict() if train_loader is not None else None
            ),
            "step": step,
            "tokens_seen": tokens_seen,
            "latest_val_loss": latest_val_loss,
            "best_val_loss": best_val_loss,
            "training_compute_seconds": training_compute_seconds,
            "wall_gpu_seconds": wall_gpu_seconds,
            "model_args": model_args,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

        torch.save(checkpoint, temp_path)
        os.replace(temp_path, final_path)

    def load(
        self,
        filepath,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        train_loader=None,
        device="cpu",
    ):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        # Always deserialize checkpoint tensors onto CPU.
        checkpoint = torch.load(
            filepath,
            map_location="cpu",
            weights_only=False,
        )

        raw_model = self._unwrap_model(model)

        state_dict = checkpoint["model"]
        prefix = "_orig_mod."

        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                state_dict[new_key] = state_dict.pop(key)

        raw_model.load_state_dict(state_dict)

        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        if train_loader is not None and checkpoint.get("train_loader") is not None:
            train_loader.load_state_dict(checkpoint["train_loader"])

        rng = checkpoint.get("rng_state")

        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])

            torch.set_rng_state(rng["torch_cpu"].cpu())

            if torch.cuda.is_available() and rng["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in rng["torch_cuda"]]
                )

        return checkpoint
