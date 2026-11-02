"""EuroSAT RGB: download, decode to a uint8 array cache, deterministic splits and label subsets.

Splits are stratified 60/20/20 (train/val/test) by class with a fixed seed and saved as file
lists under ``splits/`` so they are identical across machines. Label-scarce subsets of train
are stratified and *nested* (1% ⊂ 5% ⊂ 10% ⊂ 100%) per subset seed.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake",
]
FRACTIONS = (0.01, 0.05, 0.10, 1.0)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    z = cache / "EuroSAT_RGB.zip"
    if not z.exists() or md5(z) != MD5:
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, z)
    if md5(z) != MD5:
        raise SystemExit("EuroSAT checksum mismatch")
    return z


def decode(zip_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All images as uint8 (N, 3, 64, 64), labels (N,), and relative names, sorted by name."""
    from PIL import Image

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".jpg"))
        imgs = np.empty((len(names), 3, 64, 64), dtype=np.uint8)
        labels = np.empty(len(names), dtype=np.int64)
        rel = []
        for i, n in enumerate(names):
            parts = n.split("/")
            cls, fname = parts[-2], parts[-1]
            with Image.open(io.BytesIO(zf.read(n))) as im:
                arr = np.asarray(im.convert("RGB"))
            if arr.shape != (64, 64, 3):
                raise ValueError(f"{n}: unexpected shape {arr.shape}")
            imgs[i] = arr.transpose(2, 0, 1)
            labels[i] = CLASSES.index(cls)
            rel.append(f"{cls}/{fname}")
    return imgs, labels, rel


def stratified_split(labels: np.ndarray, seed: int = 0,
                     fracs: tuple[float, float] = (0.6, 0.2)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        rng.shuffle(idx)
        n_tr = int(round(fracs[0] * len(idx)))
        n_va = int(round(fracs[1] * len(idx)))
        out["train"] += idx[:n_tr].tolist()
        out["val"] += idx[n_tr:n_tr + n_va].tolist()
        out["test"] += idx[n_tr + n_va:].tolist()
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


def label_subsets(train_idx: np.ndarray, labels: np.ndarray, seed: int,
                  fractions: tuple[float, ...] = FRACTIONS) -> dict[float, np.ndarray]:
    """Nested, class-stratified subsets of ``train_idx`` (at least one example per class)."""
    rng = np.random.default_rng(10_000 + seed)
    perm_by_class = {}
    for c in np.unique(labels[train_idx]):
        idx = train_idx[labels[train_idx] == c].copy()
        rng.shuffle(idx)
        perm_by_class[c] = idx
    out = {}
    for f in fractions:
        take = [p[: max(1, int(round(f * len(p))))] for p in perm_by_class.values()]
        out[f] = np.sort(np.concatenate(take))
    return out


def duplicate_report(imgs: np.ndarray, splits: dict[str, np.ndarray]) -> dict[str, int]:
    """Count byte-identical images shared between splits (a leakage check)."""
    hashes = np.array([hashlib.blake2b(x.tobytes(), digest_size=16).hexdigest() for x in imgs])
    sets = {k: set(hashes[v].tolist()) for k, v in splits.items()}
    return {
        "train&val": len(sets["train"] & sets["val"]),
        "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
        "within_all": len(hashes) - len(set(hashes.tolist())),
    }


@dataclass
class EuroSAT:
    images: np.ndarray  # uint8 (N, 3, 64, 64)
    labels: np.ndarray  # int64 (N,)
    names: list[str]
    splits: dict[str, np.ndarray]

    @classmethod
    def load(cls, data_dir: str | Path = "data", split_dir: str | Path = "splits") -> EuroSAT:
        d = Path(data_dir)
        imgs = np.load(d / "eurosat_images.npy", mmap_mode="r")
        labels = np.load(d / "eurosat_labels.npy")
        names = (d / "eurosat_names.txt").read_text().split("\n")
        index = {n: i for i, n in enumerate(names)}
        splits = {}
        for k in ("train", "val", "test"):
            lst = (Path(split_dir) / f"{k}.txt").read_text().split()
            splits[k] = np.array([index[n] for n in lst], dtype=np.int64)
        return cls(np.ascontiguousarray(imgs), labels, names, splits)
