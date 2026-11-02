import numpy as np
import torch

from landprobe.data import duplicate_report, label_subsets, stratified_split
from landprobe.evaluate import knn_accuracy, linear_probe


def _labels(n=1000, k=10, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


def test_split_disjoint_complete_stratified():
    y = _labels()
    sp = stratified_split(y, seed=0)
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == len(y) == len(set(allidx.tolist()))
    for c in range(10):
        frac = np.mean(y[sp["train"]] == c) - np.mean(y == c)
        assert abs(frac) < 0.01


def test_label_subsets_nested_and_cover_classes():
    y = _labels()
    tr = stratified_split(y)["train"]
    subs = label_subsets(tr, y, seed=3)
    fr = sorted(subs)
    for a, b in zip(fr, fr[1:], strict=False):
        assert set(subs[a].tolist()) <= set(subs[b].tolist())
    assert set(y[subs[0.01]].tolist()) == set(range(10))
    assert np.array_equal(subs[1.0], np.sort(tr))
    assert not np.array_equal(label_subsets(tr, y, seed=4)[0.05], subs[0.05])


def test_duplicate_report_detects_copies():
    imgs = np.random.default_rng(0).integers(0, 255, (6, 3, 4, 4), dtype=np.uint8)
    imgs[5] = imgs[0]
    rep = duplicate_report(imgs, {"train": np.array([0, 1]), "val": np.array([2, 3]),
                                  "test": np.array([4, 5])})
    assert rep["train&test"] == 1 and rep["within_all"] == 1


def _toy(n, seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    centres = torch.randn(10, 32, generator=torch.Generator().manual_seed(99)) * 3
    return centres[y] + torch.randn(n, 32, generator=g), y


def test_knn_and_probe_on_separable_toy():
    xtr, ytr = _toy(500, 0)
    xva, yva = _toy(200, 1)
    xte, yte = _toy(200, 2)
    assert knn_accuracy(xtr, ytr, xte, yte) > 0.9
    r = linear_probe(xtr, ytr, xva, yva, xte, yte)
    assert r.test_acc > 0.9 and r.wd in r.val_by_wd
