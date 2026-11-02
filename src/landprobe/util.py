from __future__ import annotations

import random
import subprocess

import numpy as np
import torch


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def peak_memory_gb() -> float | None:
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e9
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return None


def git_commit() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], text=True)
        return head + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def thermal_state() -> str:
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True, timeout=5)
        return " ".join(x.strip() for x in out.splitlines() if x.strip())
    except Exception:
        return "unknown"
