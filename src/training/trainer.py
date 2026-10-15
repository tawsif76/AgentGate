import json
import math
import os
import shutil
import statistics
import time
from collections import deque
from contextlib import nullcontext

import torch

from .checkpoint import CheckpointManager


class TrainingConfig:
    def __init__(
        self,
        max_steps=20_599,
        grad_accum_steps=16,
        tokens_per_step=131_072,
        grad_clip=1.0,
        log_every=10,
        eval_every=400,
        subset_eval_batches=250,
        checkpoint_every=400,
        early_eval_steps=(50, 100, 200),
        full_eval_tokens=(
            100_000_000,
            500_000_000,
            1_000_000_000,
            1_500_000_000,
            2_000_000_000,
        ),
        performance_warmup_steps=10,
        step0_loss_tolerance=1.0,
        peak_flops=121e12,
        price_per_hour=0.0,
        out_dir="runs/actiongate_base_142m",
    ):
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.tokens_per_step = tokens_per_step
        self.grad_clip = grad_clip

        self.log_every = log_every
        self.eval_every = eval_every
        self.subset_eval_batches = subset_eval_batches
        self.checkpoint_every = checkpoint_every

        self.early_eval_steps = early_eval_steps
        self.full_eval_tokens = full_eval_tokens

        self.performance_warmup_steps = performance_warmup_steps
        self.step0_loss_tolerance = step0_loss_tolerance

        self.peak_flops = peak_flops
        self.price_per_hour = price_per_hour
        self.out_dir = out_dir


def unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def build_optimizer(model, lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8):
    model = unwrap_model(model)

    decay = []
    no_decay = []

    embedding_weight = model.token_embedding.weight

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter is embedding_weight or parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    device = next(model.parameters()).device

    return torch.optim.AdamW(
        param_groups, lr=lr, betas=betas, eps=eps, fused=device.type == "cuda"
    )


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config,
        device="cuda",
        run_config=None,
        sample_fn=None,
        wall_session_start=None,
        persist_fn=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config
        self.device = torch.device(device)

        self.sample_fn = sample_fn
        self.persist_fn = persist_fn

        os.makedirs(config.out_dir, exist_ok=True)

        self.metrics_path = os.path.join(config.out_dir, "metrics.jsonl")
        self.summary_path = os.path.join(config.out_dir, "summary.json")

        self.checkpoints = CheckpointManager(
            os.path.join(config.out_dir, "checkpoints")
        )

        self.step = 0
        self.tokens_seen = 0

        self.latest_val_loss = None
        self.best_val_loss = float("inf")

        self.training_compute_seconds = 0.0
        self.wall_gpu_seconds_base = 0.0

        self.wall_session_start = (
            wall_session_start if wall_session_start is not None else time.time()
        )

        self.train_buffer = []

        self.recent_losses = deque(maxlen=50)
        self.recent_grad_norms = deque(maxlen=50)

        raw_model = unwrap_model(model)

        self.parameter_count = raw_model.num_parameters()
        model_config = raw_model.config

        # FLOPs/token = 6N + 12*L*H*Q*T.
        # N excludes RMSNorm parameters because they are not matrix multiplies.
        norm_param_count = sum(
            p.numel() for name, p in raw_model.named_parameters() if "norm" in name
        )

        matmul_param_count = self.parameter_count - norm_param_count

        self.flops_per_token = (
            6 * matmul_param_count
            + 12
            * model_config.n_layers
            * model_config.n_heads
            * model_config.head_dim
            * model_config.context_length
        )

        self.run_config = {
            "model": vars(model_config).copy(),
            "training": vars(config).copy(),
            "parameter_count": self.parameter_count,
            "matmul_param_count": matmul_param_count,
            "flops_per_token": self.flops_per_token,
            "flops_convention": (
                "6N + 12*L*H*Q*T; N = matmul-only parameters, excluding RMSNorm weights"
            ),
            "pytorch_version": torch.__version__,
            "torch_compile": hasattr(model, "_orig_mod"),
        }

        if self.device.type == "cuda":
            self.run_config["gpu"] = torch.cuda.get_device_name(self.device)

        if run_config is not None:
            self.run_config.update(run_config)

        self._write_run_config()
        self._persist()

        self._refresh_milestones()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _persist(self):
        if self.persist_fn is not None:
            self.persist_fn()

    def _wall_gpu_seconds(self):
        return self.wall_gpu_seconds_base + time.time() - self.wall_session_start

    def _cost_usd(self):
        return self._wall_gpu_seconds() / 3600 * self.config.price_per_hour

    def _write_run_config(self):
        path = os.path.join(self.config.out_dir, "run_config.json")

        if os.path.exists(path):
            return

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.run_config, f, indent=2)

    def _write_event(self, **event):
        event["time"] = time.time()

        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")

    def _safe_number(self, value):
        return value if math.isfinite(value) else str(value)

    def _refresh_milestones(self):
        self.pending_milestones = [
            milestone
            for milestone in self.config.full_eval_tokens
            if milestone > self.tokens_seen
        ]

    @torch.no_grad()
    def _step_zero_check(self):
        loader_state = self.train_loader.state_dict()

        x, y = self.train_loader.next_batch()
        self.train_loader.load_state_dict(loader_state)

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        was_training = self.model.training
        self.model.eval()

        with self._autocast():
            _, loss = self.model(x, y)

        loss = loss.item()
        expected = math.log(unwrap_model(self.model).config.vocab_size)

        if was_training:
            self.model.train()

        self._write_event(
            event="step0", loss=self._safe_number(loss), expected_loss=expected
        )

        self._persist()

        if (
            not math.isfinite(loss)
            or abs(loss - expected) > self.config.step0_loss_tolerance
        ):
            raise RuntimeError(
                f"Step-0 loss check failed: loss={loss:.4f}, expected≈{expected:.4f}"
            )

    def _train_step(self):
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=self.device)

        self._synchronize()
        start = time.perf_counter()

        for _ in range(self.config.grad_accum_steps):
            x, y = self.train_loader.next_batch()

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._autocast():
                _, raw_loss = self.model(x, y)

            total_loss += raw_loss.detach()

            loss = raw_loss / self.config.grad_accum_steps
            loss.backward()

        mean_loss = total_loss / self.config.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )

        update_is_finite = torch.isfinite(mean_loss) & torch.isfinite(grad_norm)

        if not update_is_finite.item():
            self._synchronize()

            step_time = time.perf_counter() - start
            self.training_compute_seconds += step_time

            loss_value = mean_loss.item()
            grad_value = grad_norm.item()

            self.optimizer.zero_grad(set_to_none=True)

            self._write_event(
                event="nonfinite_update",
                step=self.step + 1,
                tokens_seen=self.tokens_seen,
                loss=self._safe_number(loss_value),
                grad_norm=self._safe_number(grad_value),
                training_compute_hours=self.training_compute_seconds / 3600,
                wall_gpu_hours=self._wall_gpu_seconds() / 3600,
                cost_usd=self._cost_usd(),
            )

            self._persist()

            raise FloatingPointError("Non-finite loss or gradient detected")

        lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self._synchronize()

        step_time = time.perf_counter() - start

        self.step += 1
        self.tokens_seen += self.config.tokens_per_step
        self.training_compute_seconds += step_time

        train_loss = mean_loss.item()
        grad_norm = grad_norm.item()

        tokens_per_sec = self.config.tokens_per_step / step_time

        mfu = self.flops_per_token * tokens_per_sec / self.config.peak_flops

        peak_mem_gb = 0.0

        if self.device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9

        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "clipped": grad_norm > self.config.grad_clip,
            "step_time": step_time,
            "tokens_per_sec": tokens_per_sec,
            "mfu": mfu,
            "peak_mem_gb": peak_mem_gb,
        }

    def _record_train_step(self, metrics):
        loss = metrics["train_loss"]
        grad_norm = metrics["grad_norm"]

        if len(self.recent_losses) >= 20:
            recent_mean = sum(self.recent_losses) / len(self.recent_losses)

            if loss > 2 * recent_mean:
                self._write_event(
                    event="loss_spike",
                    step=self.step,
                    tokens_seen=self.tokens_seen,
                    loss=loss,
                    recent_loss_mean=recent_mean,
                )

        if len(self.recent_grad_norms) >= 20:
            recent_median = statistics.median(self.recent_grad_norms)

            if recent_median > 0 and grad_norm > 10 * recent_median:
                self._write_event(
                    event="grad_spike",
                    step=self.step,
                    tokens_seen=self.tokens_seen,
                    grad_norm=grad_norm,
                    recent_median=recent_median,
                )

        self.recent_losses.append(loss)
        self.recent_grad_norms.append(grad_norm)

        self.train_buffer.append(metrics)

        if len(self.train_buffer) >= self.config.log_every:
            self._flush_train_metrics()

    def _flush_train_metrics(self):
        if not self.train_buffer:
            return

        buffer = self.train_buffer

        def mean(key):
            return sum(item[key] for item in buffer) / len(buffer)

        clip_frac = sum(item["clipped"] for item in buffer) / len(buffer)

        self._write_event(
            event="train",
            step=self.step,
            tokens_seen=self.tokens_seen,
            window_steps=len(buffer),
            train_loss=mean("train_loss"),
            lr=buffer[-1]["lr"],
            grad_norm=mean("grad_norm"),
            clip_frac=clip_frac,
            step_time=mean("step_time"),
            tokens_per_sec=mean("tokens_per_sec"),
            mfu=mean("mfu"),
            peak_mem_gb=max(item["peak_mem_gb"] for item in buffer),
            training_compute_hours=self.training_compute_seconds / 3600,
            wall_gpu_hours=self._wall_gpu_seconds() / 3600,
            cost_usd=self._cost_usd(),
        )

        self.train_buffer.clear()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self._persist()

    @torch.no_grad()
    def evaluate(self, max_batches=None, eval_type="subset"):
        self.val_loader.reset()

        was_training = self.model.training
        self.model.eval()

        num_batches = self.val_loader.num_batches

        if max_batches is not None:
            num_batches = min(max_batches, num_batches)

        total_loss = torch.zeros((), device=self.device)
        total_tokens = 0

        for _ in range(num_batches):
            x, y = self.val_loader.next_batch()

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._autocast():
                _, loss = self.model(x, y)

            n_tokens = y.numel()

            total_loss += loss.detach() * n_tokens
            total_tokens += n_tokens

        val_loss = (total_loss / total_tokens).item()

        if was_training:
            self.model.train()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        if not math.isfinite(val_loss):
            self._write_event(
                event="nonfinite_eval",
                step=self.step,
                tokens_seen=self.tokens_seen,
                eval_type=eval_type,
                wall_gpu_hours=self._wall_gpu_seconds() / 3600,
                cost_usd=self._cost_usd(),
            )

            self._persist()

            raise FloatingPointError("Non-finite validation loss detected")

        val_ppl = math.exp(val_loss)

        self.latest_val_loss = val_loss

        if eval_type == "full":
            self.best_val_loss = min(self.best_val_loss, val_loss)

        self._write_event(
            event="eval",
            eval_type=eval_type,
            step=self.step,
            tokens_seen=self.tokens_seen,
            val_tokens=total_tokens,
            val_loss=val_loss,
            val_ppl=val_ppl,
            training_compute_hours=self.training_compute_seconds / 3600,
            wall_gpu_hours=self._wall_gpu_seconds() / 3600,
            cost_usd=self._cost_usd(),
        )

        self._persist()

        return val_loss

    def _save_checkpoint(self, filename="latest.pt"):
        self._flush_train_metrics()

        raw_model = unwrap_model(self.model)

        self.checkpoints.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            step=self.step,
            tokens_seen=self.tokens_seen,
            latest_val_loss=self.latest_val_loss,
            best_val_loss=self.best_val_loss,
            training_compute_seconds=self.training_compute_seconds,
            wall_gpu_seconds=self._wall_gpu_seconds(),
            model_args=vars(raw_model.config).copy(),
            config=self.run_config,
        )

        self._persist()

    def _save_milestone(self, filename):
        self._save_checkpoint("latest.pt")

        source = os.path.join(self.checkpoints.out_dir, "latest.pt")

        destination = os.path.join(self.checkpoints.out_dir, filename)

        temp_path = destination + ".tmp"

        shutil.copy2(source, temp_path)
        os.replace(temp_path, destination)

        self._persist()

    def _latest_logged_times(self):
        training_seconds = 0.0
        wall_seconds = 0.0

        if not os.path.exists(self.metrics_path):
            return training_seconds, wall_seconds

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                training_seconds = max(
                    training_seconds, event.get("training_compute_hours", 0.0) * 3600
                )

                wall_seconds = max(
                    wall_seconds, event.get("wall_gpu_hours", 0.0) * 3600
                )

        return training_seconds, wall_seconds

    def _trim_events_after_step(self, checkpoint_step):
        if not os.path.exists(self.metrics_path):
            return

        rollback_events = {"train", "eval", "loss_spike", "grad_spike"}

        kept_events = []

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_step = event.get("step")
                event_type = event.get("event")

                should_remove = (
                    event_type in rollback_events
                    and event_step is not None
                    and event_step > checkpoint_step
                )

                if not should_remove:
                    kept_events.append(event)

        temp_path = self.metrics_path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            for event in kept_events:
                f.write(json.dumps(event, allow_nan=False) + "\n")

        os.replace(temp_path, self.metrics_path)

    def resume(self, filepath):
        checkpoint = self.checkpoints.load(
            filepath=filepath,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            device=self.device,
        )

        # Read durable timing before rolling back uncheckpointed metrics.
        logged_training, logged_wall = self._latest_logged_times()

        self.step = checkpoint.get("step", 0)
        self.tokens_seen = checkpoint.get("tokens_seen", 0)

        self.latest_val_loss = checkpoint.get("latest_val_loss")
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        self.training_compute_seconds = max(
            checkpoint.get("training_compute_seconds", 0.0), logged_training
        )

        self.wall_gpu_seconds_base = max(
            checkpoint.get("wall_gpu_seconds", 0.0), logged_wall
        )

        self._trim_events_after_step(self.step)
        self._refresh_milestones()

        self._write_event(
            event="restart",
            step=self.step,
            tokens_seen=self.tokens_seen,
            training_compute_hours=self.training_compute_seconds / 3600,
            wall_gpu_hours=self._wall_gpu_seconds() / 3600,
            cost_usd=self._cost_usd(),
            checkpoint=filepath,
        )

        self._persist()

        return checkpoint

    def _next_milestone(self):
        if not self.pending_milestones:
            return None

        milestone = self.pending_milestones[0]

        if self.tokens_seen < milestone:
            return None

        self.pending_milestones.pop(0)

        return milestone

    def _read_events(self):
        if not os.path.exists(self.metrics_path):
            return []

        events = []

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return events

    def _final_train_loss(self, train_events, target_steps=100):
        remaining = target_steps
        weighted_loss = 0.0
        used_steps = 0

        for event in reversed(train_events):
            window_steps = event.get("window_steps", self.config.log_every)

            n = min(window_steps, remaining)

            weighted_loss += event["train_loss"] * n
            used_steps += n
            remaining -= n

            if remaining == 0:
                break

        if used_steps == 0:
            return None

        return weighted_loss / used_steps

    def _write_summary(self, final_val_loss):
        events = self._read_events()

        train_events = [event for event in events if event.get("event") == "train"]

        performance_events = [
            event
            for event in train_events
            if event["step"] > self.config.performance_warmup_steps
        ]

        def weighted_mean(field, records):
            if not records:
                return None

            numerator = 0.0
            denominator = 0

            for record in records:
                weight = record.get("window_steps", self.config.log_every)

                numerator += record[field] * weight
                denominator += weight

            return numerator / denominator

        def median(field, records):
            if not records:
                return None

            return statistics.median(record[field] for record in records)

        logged_steps = sum(
            event.get("window_steps", self.config.log_every) for event in train_events
        )

        clipped_steps = sum(
            event["clip_frac"] * event.get("window_steps", self.config.log_every)
            for event in train_events
        )

        overall_clip_fraction = clipped_steps / logged_steps if logged_steps else 0.0

        peak_memory = max((event["peak_mem_gb"] for event in train_events), default=0.0)

        summary = {
            "model": "ActionGate-Base-142M",
            "parameters": self.parameter_count,
            "training_tokens_seen": self.tokens_seen,
            "tokens_per_parameter": (self.tokens_seen / self.parameter_count),
            "final_train_loss_last_100_steps": self._final_train_loss(train_events),
            "final_full_val_loss": final_val_loss,
            "final_val_perplexity": math.exp(final_val_loss),
            "effective_batch_tokens": self.config.tokens_per_step,
            "mean_tokens_per_sec": weighted_mean("tokens_per_sec", performance_events),
            "median_tokens_per_sec": median("tokens_per_sec", performance_events),
            "mean_mfu": weighted_mean("mfu", performance_events),
            "median_mfu": median("mfu", performance_events),
            "peak_gpu_memory_gb": peak_memory,
            "overall_clip_fraction": overall_clip_fraction,
            "training_compute_hours": (self.training_compute_seconds / 3600),
            "wall_gpu_hours": self._wall_gpu_seconds() / 3600,
            "cost_usd": self._cost_usd(),
            "restart_count": sum(event.get("event") == "restart" for event in events),
            "loss_spike_count": sum(
                event.get("event") == "loss_spike" for event in events
            ),
            "grad_spike_count": sum(
                event.get("event") == "grad_spike" for event in events
            ),
            "nonfinite_update_count": sum(
                event.get("event") == "nonfinite_update" for event in events
            ),
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._persist()

        return summary

    def train(self, session_end_step=None):
        self.model.train()

        if self.step == 0:
            self._step_zero_check()

        if session_end_step is None:
            session_end_step = self.config.max_steps

        session_end_step = min(session_end_step, self.config.max_steps)

        if session_end_step <= self.step:
            raise ValueError(
                "session_end_step must be greater than the current training step"
            )

        while self.step < session_end_step:
            metrics = self._train_step()
            self._record_train_step(metrics)

            milestone = self._next_milestone()

            if milestone is not None:
                self.evaluate(max_batches=None, eval_type="full")

                filename = f"milestone_{milestone // 1_000_000}M.pt"

                self._save_milestone(filename)

                if self.sample_fn is not None:
                    self.sample_fn(self.model, self.tokens_seen)

                    self._persist()

                continue

            should_eval = (
                self.step in self.config.early_eval_steps
                or self.step % self.config.eval_every == 0
            )

            if should_eval:
                self.evaluate(
                    max_batches=self.config.subset_eval_batches, eval_type="subset"
                )

            if self.step % self.config.checkpoint_every == 0:
                self._save_checkpoint("latest.pt")

        self._flush_train_metrics()

        # Session ended, but full training is not finished yet.
        if self.step < self.config.max_steps:
            self._save_checkpoint("latest.pt")

            self._write_event(
                event="session_end",
                step=self.step,
                tokens_seen=self.tokens_seen,
                remaining_steps=self.config.max_steps - self.step,
                wall_gpu_hours=self._wall_gpu_seconds() / 3600,
                cost_usd=self._cost_usd(),
            )

            self._persist()

            return {
                "status": "session_complete",
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "remaining_steps": self.config.max_steps - self.step,
            }

        # Only the real end of pretraining reaches here.
        final_val_loss = self.evaluate(max_batches=None, eval_type="full")

        self._save_milestone("final.pt")

        if self.sample_fn is not None:
            self.sample_fn(self.model, self.tokens_seen)

            self._persist()

        summary = self._write_summary(final_val_loss)

        return {"status": "training_complete", "summary": summary}
