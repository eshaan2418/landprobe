"""Self-supervised pretraining (SimCLR or BYOL) on the *unlabelled* train split.

Each run writes ``runs/<name>/`` (checkpoint, gitignored) and ``eval/results/pretrain/<name>.json``
(config, seed, per-epoch loss and kNN val accuracy, wall-clock, peak memory, commit).
kNN accuracy on the validation split is a monitor only; it is not used to pick checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .augment import AugConfig, augment
from .data import EuroSAT
from .evaluate import extract, knn_accuracy
from .models import BYOL, SimCLR
from .util import device, git_commit, peak_memory_gb, seed_everything, thermal_state


@dataclass
class PretrainConfig:
    name: str
    method: str = "simclr"  # simclr | byol
    seed: int = 0
    epochs: int = 40
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    temperature: float = 0.2
    tau_base: float = 0.99
    aug: dict = field(default_factory=dict)
    knn_every: int = 5
    limit_batches: int | None = None  # smoke tests


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def run(cfg: PretrainConfig, data: EuroSAT, out_root: str = ".") -> dict:
    commit = git_commit()  # recorded at start: the code this process actually loaded
    seed_everything(cfg.seed)
    dev = device()
    aug = AugConfig(**cfg.aug)
    g = torch.Generator().manual_seed(cfg.seed)
    model: SimCLR | BYOL = SimCLR(cfg.temperature) if cfg.method == "simclr" else BYOL()
    model.to(dev)
    params = model.online_parameters() if isinstance(model, BYOL) else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    train = torch.from_numpy(data.images[data.splits["train"]]).to(dev)  # uint8 on device
    n = train.shape[0]
    steps_per_epoch = n // cfg.batch_size
    if cfg.limit_batches:
        steps_per_epoch = min(steps_per_epoch, cfg.limit_batches)
    total = steps_per_epoch * cfg.epochs
    log = []
    step = 0
    peak = 0.0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        losses = []
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size : (i + 1) * cfg.batch_size].to(dev)
            x = train[idx]
            v1, v2 = augment(x, aug, g), augment(x, aug, g)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, cfg.warmup_epochs * steps_per_epoch, cfg.lr)
            loss = model.loss(v1, v2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if isinstance(model, BYOL):
                tau = 1 - (1 - cfg.tau_base) * (math.cos(math.pi * step / total) + 1) / 2
                model.update_target(tau)
            losses.append(loss.detach())
            step += 1
        rec = {"epoch": epoch + 1, "loss": float(torch.stack(losses).mean()),
               "elapsed_s": time.perf_counter() - t0}
        peak = max(peak, peak_memory_gb() or 0.0)
        if (epoch + 1) % cfg.knn_every == 0 or epoch + 1 == cfg.epochs:
            rec["knn_val_acc"] = knn_val(model.backbone, data, dev)
        if not math.isfinite(rec["loss"]):
            raise FloatingPointError(f"loss diverged at epoch {epoch + 1}")
        log.append(rec)
        print(json.dumps({"run": cfg.name, **rec}), flush=True)
    ckpt_dir = Path(out_root) / "runs" / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), ckpt_dir / "backbone.pt")
    summary = {
        "config": asdict(cfg),
        "log": log,
        "wall_s": time.perf_counter() - t0,
        "peak_memory_gb": peak,  # max of per-epoch MPS driver allocation samples
        "git_commit": commit,
        "thermal": thermal_state(),
        "device": str(dev),
        "torch": torch.__version__,
    }
    res = Path(out_root) / "eval" / "results" / "pretrain"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{cfg.name}.json").write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def knn_val(backbone: torch.nn.Module, data: EuroSAT, dev: torch.device) -> float:
    tr = data.splits["train"]
    va = data.splits["val"]
    ftr = extract(backbone, data.images[tr], dev)
    fva = extract(backbone, data.images[va], dev)
    return knn_accuracy(ftr, torch.from_numpy(data.labels[tr]), fva,
                        torch.from_numpy(data.labels[va]))


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
