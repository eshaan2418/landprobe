"""Evaluate every backbone under every protocol.

    python scripts/evaluate_all.py probe      # kNN + linear probe, fractions 1/5/10/100%, 3 seeds
    python scripts/evaluate_all.py finetune   # fine-tune at 1/10/100%, 3 seeds

Backbones: random init, ImageNet-1k ResNet-18 (torchvision), SimCLR and BYOL (seeds 0-2).
Seed s of an SSL backbone is paired with label-subset seed s. Results go to
eval/results/{probe,finetune}/<backbone>_<fraction>_s<seed>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from landprobe.data import EuroSAT, label_subsets
from landprobe.evaluate import FinetuneConfig, extract, finetune, knn_accuracy, linear_probe
from landprobe.models import resnet18_imagenet, resnet18_small
from landprobe.util import device, git_commit, peak_memory_gb, thermal_state

SEEDS = (0, 1, 2)
FRACTIONS = (0.01, 0.05, 0.10, 1.0)
FT_FRACTIONS = (0.01, 0.10, 1.0)
FT_STEPS = {0.01: 1000, 0.10: 1000, 1.0: 2000}


def backbones(seed: int) -> dict[str, tuple[torch.nn.Module, int]]:
    """name -> (fresh backbone, input size). ImageNet input size chosen on val (see probe)."""
    torch.manual_seed(seed)
    out: dict[str, tuple[torch.nn.Module, int]] = {"random": (resnet18_small(), 64)}
    for size in (64, 128):
        out[f"imagenet{size}"] = (resnet18_imagenet(), size)
    for m in ("simclr", "byol"):
        ck = Path(f"runs/{m}_e40_s{seed}/backbone.pt")
        if ck.exists():
            b = resnet18_small()
            b.load_state_dict(torch.load(ck, map_location="cpu"))
            out[m] = (b, 64)
    return out


def meta() -> dict:
    return {"git_commit": git_commit(), "thermal": thermal_state(), "torch": torch.__version__,
            "peak_memory_gb": peak_memory_gb()}


def probe(data: EuroSAT, out: Path) -> None:
    dev = device()
    y = data.labels
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    for seed in SEEDS:
        subs = label_subsets(tr, y, seed)
        for name, (bb, size) in backbones(seed).items():
            bb = bb.to(dev)
            t0 = time.perf_counter()
            f_all = extract(bb, data.images, dev, size=size)
            ext_s = time.perf_counter() - t0
            for frac in FRACTIONS:
                idx = subs[frac]
                yt = torch.from_numpy(y)
                r = linear_probe(f_all[idx], yt[idx], f_all[va], yt[va], f_all[te], yt[te])
                knn_va = knn_accuracy(f_all[idx], yt[idx], f_all[va], yt[va],
                                      k=min(20, len(idx)))
                knn_te = knn_accuracy(f_all[idx], yt[idx], f_all[te], yt[te],
                                      k=min(20, len(idx)))
                rec = {"backbone": name, "input_size": size, "fraction": frac, "seed": seed,
                       "n_labels": int(len(idx)), "probe_wd": r.wd, "probe_val": r.val_acc,
                       "probe_test": r.test_acc, "probe_val_by_wd": r.val_by_wd,
                       "knn_val": knn_va, "knn_test": knn_te, "extract_s": ext_s, **meta()}
                (out / f"{name}_{frac}_s{seed}.json").write_text(json.dumps(rec, indent=2))
                print(json.dumps({k: rec[k] for k in ("backbone", "fraction", "seed",
                                                      "probe_test", "knn_test")}), flush=True)
            bb.cpu()


def pick_imagenet_size(probe_dir: Path) -> int:
    """Input size for the ImageNet baseline: best mean *val* probe accuracy at 10% labels."""
    best = (0.0, 64)
    for size in (64, 128):
        vals = [json.loads(p.read_text())["probe_val"]
                for p in probe_dir.glob(f"imagenet{size}_0.1_s*.json")]
        if vals and np.mean(vals) > best[0]:
            best = (float(np.mean(vals)), size)
    return best[1]


def run_finetune(data: EuroSAT, out: Path, probe_dir: Path) -> None:
    dev = device()
    y = data.labels
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    im_size = pick_imagenet_size(probe_dir)
    print(f"ImageNet baseline input size (chosen on val): {im_size}", flush=True)
    for seed in SEEDS:
        subs = label_subsets(tr, y, seed)
        bbs = backbones(seed)
        for name in ("random", f"imagenet{im_size}", "simclr", "byol"):
            if name not in bbs:
                continue
            for frac in FT_FRACTIONS:
                dst = out / f"{name}_{frac}_s{seed}.json"
                if dst.exists():
                    continue
                bb, size = backbones(seed)[name]  # fresh copy per run
                cfg = FinetuneConfig(steps=FT_STEPS[frac], input_size=size)
                r = finetune(bb, data.images, y, subs[frac], va, te, cfg, dev, seed)
                rec = {"backbone": name, "input_size": size, "fraction": frac, "seed": seed,
                       "n_labels": int(len(subs[frac])), "steps": cfg.steps,
                       "batch_size": cfg.batch_size, "lr": cfg.lr, **r, **meta()}
                dst.write_text(json.dumps(rec))
                print(json.dumps({k: rec[k] for k in ("backbone", "fraction", "seed", "val_acc",
                                                      "test_acc", "train_s")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["probe", "finetune"])
    a = ap.parse_args()
    data = EuroSAT.load()
    probe_dir = Path("eval/results/probe")
    if a.which == "probe":
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe(data, probe_dir)
    else:
        ft = Path("eval/results/finetune")
        ft.mkdir(parents=True, exist_ok=True)
        run_finetune(data, ft, probe_dir)


if __name__ == "__main__":
    main()
