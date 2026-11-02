"""landprobe CLI.

    landprobe classify IMAGES [--backbone runs/simclr_e40_s0/backbone.pt] [--labels-frac 0.01]

Fits a linear probe on a small labelled subset of the EuroSAT train split using frozen
features from the given backbone (seconds), then classifies IMAGES (a folder of 64x64 RGB
tiles, a single image, or an .npz with an ``images`` array) and prints the top-3 classes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def _load_images(path: Path) -> tuple[np.ndarray, list[str]]:
    from PIL import Image

    if path.suffix == ".npz":
        z = np.load(path)
        names = z["names"].tolist() if "names" in z else [str(i) for i in range(len(z["images"]))]
        return z["images"], names
    files = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png")) if path.is_dir() else [path]
    arr = []
    for f in files:
        with Image.open(f) as im:
            arr.append(np.asarray(im.convert("RGB").resize((64, 64))).transpose(2, 0, 1))
    return np.stack(arr), [f.name for f in files]


def cmd_classify(a: argparse.Namespace) -> int:
    from .data import CLASSES, EuroSAT, label_subsets
    from .evaluate import _fit_linear, extract
    from .models import resnet18_small
    from .util import device

    dev = device()
    bb = resnet18_small()
    bb.load_state_dict(torch.load(a.backbone, map_location="cpu"))
    bb.to(dev)
    data = EuroSAT.load()
    sub = label_subsets(data.splits["train"], data.labels, seed=0)[a.labels_frac]
    t0 = time.perf_counter()
    ftr = extract(bb, data.images[sub], dev)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    clf = _fit_linear((ftr - mu) / sd, torch.from_numpy(data.labels[sub]), wd=a.wd)
    fit_s = time.perf_counter() - t0
    imgs, names = _load_images(Path(a.images))
    with torch.no_grad():
        p = torch.softmax(clf((extract(bb, imgs, dev) - mu) / sd), 1)
    print(f"linear probe fit on {len(sub)} labelled tiles in {fit_s:.1f}s\n")
    for n, row in zip(names, p, strict=True):
        top = row.topk(3)
        cells = ", ".join(f"{CLASSES[int(i)]} {float(v):.2f}"
                          for v, i in zip(top.values, top.indices, strict=True))
        print(f"{n:45s} {cells}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landprobe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify", help="few-label land-use classification")
    c.add_argument("images")
    c.add_argument("--backbone", default="runs/simclr_e40_s0/backbone.pt")
    c.add_argument("--labels-frac", type=float, default=0.01, choices=[0.01, 0.05, 0.1, 1.0])
    c.add_argument("--wd", type=float, default=1e-2)
    c.set_defaults(fn=cmd_classify)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
