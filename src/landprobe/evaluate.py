"""Evaluation protocols: frozen-feature kNN and linear probe, and full fine-tuning.

Hyperparameters that are selected (linear-probe weight decay) are selected on the validation
split only; test accuracy is computed once per configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .augment import AugConfig, augment, eval_transform

WD_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


@torch.no_grad()
def extract(backbone: nn.Module, images_u8: np.ndarray, dev: torch.device,
            batch: int = 512, size: int | None = None) -> torch.Tensor:
    backbone.eval()
    feats = []
    for i in range(0, len(images_u8), batch):
        x = eval_transform(torch.from_numpy(np.ascontiguousarray(images_u8[i : i + batch])).to(dev))
        if size is not None and size != x.shape[-1]:
            x = nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                          align_corners=False)
        feats.append(backbone(x).float().cpu())
    return torch.cat(feats)


def knn_accuracy(ftr: torch.Tensor, ytr: torch.Tensor, fte: torch.Tensor, yte: torch.Tensor,
                 k: int = 20, t: float = 0.07, n_classes: int = 10) -> float:
    """Weighted cosine kNN (as in Wu et al. 2018 / DINO)."""
    a = nn.functional.normalize(ftr, dim=1)
    b = nn.functional.normalize(fte, dim=1)
    correct = 0
    for i in range(0, len(b), 1024):
        sim = b[i : i + 1024] @ a.t()
        w, idx = sim.topk(k, dim=1)
        w = (w / t).exp()
        votes = torch.zeros(len(idx), n_classes)
        votes.scatter_add_(1, ytr[idx], w)
        correct += int((votes.argmax(1) == yte[i : i + 1024]).sum())
    return correct / len(b)


@dataclass
class ProbeResult:
    wd: float
    val_acc: float
    test_acc: float
    val_by_wd: dict[float, float]


def _fit_linear(x: torch.Tensor, y: torch.Tensor, wd: float, epochs: int = 300,
                n_classes: int = 10, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    clf = nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1, max_iter=epochs, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(clf(x), y) + wd * clf.weight.pow(2).sum() / 2
        loss.backward()
        return loss

    opt.step(closure)
    return clf


def linear_probe(ftr: torch.Tensor, ytr: torch.Tensor, fva: torch.Tensor, yva: torch.Tensor,
                 fte: torch.Tensor, yte: torch.Tensor) -> ProbeResult:
    """Multinomial logistic regression on standardised frozen features; wd chosen on val."""
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    s = lambda f: (f - mu) / sd  # noqa: E731
    xtr, xva, xte = s(ftr), s(fva), s(fte)
    val_by_wd = {}
    best: tuple[float, float, nn.Linear] | None = None
    for wd in WD_GRID:
        clf = _fit_linear(xtr, ytr, wd)
        with torch.no_grad():
            acc = float((clf(xva).argmax(1) == yva).float().mean())
        val_by_wd[wd] = acc
        if best is None or acc > best[1]:
            best = (wd, acc, clf)
    assert best is not None
    with torch.no_grad():
        test = float((best[2](xte).argmax(1) == yte).float().mean())
    return ProbeResult(best[0], best[1], test, val_by_wd)


@dataclass
class FinetuneConfig:
    steps: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup: int = 50
    aug: AugConfig | None = None
    input_size: int = 64


def finetune(backbone: nn.Module, data_images: np.ndarray, labels: np.ndarray,
             train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
             cfg: FinetuneConfig, dev: torch.device, seed: int) -> dict:
    """Fine-tune backbone + linear head for a fixed number of steps; report final val/test."""
    import math

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    aug = cfg.aug or AugConfig(crop_scale=(0.5, 1.0), jitter_p=0.5, gray_p=0.0)
    head = nn.Linear(512, 10)
    model = nn.Sequential(backbone, head).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xs = torch.from_numpy(data_images[train_idx]).to(dev)
    ys = torch.from_numpy(labels[train_idx]).to(dev)
    n = len(train_idx)
    t0 = time.perf_counter()
    model.train()
    for step in range(cfg.steps):
        lr = cfg.lr * (step + 1) / cfg.warmup if step < cfg.warmup else \
            cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup) / (cfg.steps - cfg.warmup)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=g).to(dev)
        x = augment(xs[idx], aug, g)
        if cfg.input_size != 64:
            x = nn.functional.interpolate(x, size=(cfg.input_size,) * 2, mode="bilinear",
                                          align_corners=False)
        loss = nn.functional.cross_entropy(model(x), ys[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0
    out: dict[str, Any] = {"train_s": train_s, "final_loss": float(loss)}
    for split, sidx in (("val", val_idx), ("test", test_idx)):
        f = extract(model, data_images[sidx], dev, size=cfg.input_size)
        out[f"{split}_acc"] = float((f.argmax(1) == torch.from_numpy(labels[sidx])).float().mean())
        out[f"{split}_pred"] = f.argmax(1).tolist() if split == "test" else None
    return out
