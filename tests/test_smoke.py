"""End-to-end smoke test on 60 real EuroSAT tiles (CPU): pretrain -> probe -> fine-tune."""

from pathlib import Path

import numpy as np
import torch

from landprobe.data import EuroSAT
from landprobe.evaluate import FinetuneConfig, extract, finetune, linear_probe
from landprobe.models import resnet18_small
from landprobe.pretrain import PretrainConfig, run

FIX = Path(__file__).parent / "fixtures" / "eurosat_tiny.npz"


def tiny() -> EuroSAT:
    z = np.load(FIX)
    n = len(z["labels"])
    idx = np.arange(n)
    splits = {"train": idx[idx % 3 == 0], "val": idx[idx % 3 == 1], "test": idx[idx % 3 == 2]}
    return EuroSAT(z["images"], z["labels"], z["names"].tolist(), splits)


def test_pipeline_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("landprobe.pretrain.device", lambda: torch.device("cpu"))
    d = tiny()
    for method in ("simclr", "byol"):
        s = run(PretrainConfig(name=f"t_{method}", method=method, epochs=1, batch_size=8,
                               knn_every=1), d, out_root=str(tmp_path))
        assert np.isfinite(s["log"][-1]["loss"])
        assert (tmp_path / "eval" / "results" / "pretrain" / f"t_{method}.json").exists()
    bb = resnet18_small()
    bb.load_state_dict(torch.load(tmp_path / "runs" / "t_simclr" / "backbone.pt"))
    cpu = torch.device("cpu")
    f = extract(bb, d.images, cpu)
    y = torch.from_numpy(d.labels)
    tr, va, te = (d.splits[k] for k in ("train", "val", "test"))
    r = linear_probe(f[tr], y[tr], f[va], y[va], f[te], y[te])
    assert 0.0 <= r.test_acc <= 1.0
    out = finetune(bb, d.images, d.labels, tr, va, te, FinetuneConfig(steps=3, batch_size=8,
                   warmup=1), cpu, seed=0)
    assert 0.0 <= out["test_acc"] <= 1.0 and len(out["test_pred"]) == len(te)
