import math


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr=6e-4, min_lr=6e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.completed_steps = 0

        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be greater than 0")

        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")

        self._set_lr(self._get_lr(1))

    def _get_lr(self, update):
        if update <= self.warmup_steps:
            return self.peak_lr * update / self.warmup_steps

        progress = (update - self.warmup_steps) / (self.total_steps - self.warmup_steps)

        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.min_lr + cosine * (self.peak_lr - self.min_lr)

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self.completed_steps += 1

        next_update = min(self.completed_steps + 1, self.total_steps)

        self._set_lr(self._get_lr(next_update))

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state):
        self.completed_steps = state["completed_steps"]

        next_update = min(self.completed_steps + 1, self.total_steps)

        self._set_lr(self._get_lr(next_update))
