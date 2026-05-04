"""Early stopping utility — stops training when val metric stops improving."""
import torch
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0
        self.best_state = None

    def __call__(self, metric: float, model_state: dict) -> bool:
        if self.best is None:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
            return False

        improved = (metric - self.best > self.min_delta) if self.mode == "max" \
                   else (self.best - metric > self.min_delta)

        if improved:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
