"""Download EuroSAT RGB, verify MD5, decode to data/*.npy, write splits/ (if absent).

Usage: python scripts/prepare_data.py [--cache DIR] [--out data] [--write-splits]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from landprobe.data import decode, download, duplicate_report, stratified_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--write-splits", action="store_true",
                    help="regenerate splits/ (they are committed; normally not needed)")
    a = ap.parse_args()
    z = download(Path(a.cache))
    imgs, labels, names = decode(z)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "eurosat_images.npy", imgs)
    np.save(out / "eurosat_labels.npy", labels)
    (out / "eurosat_names.txt").write_text("\n".join(names))
    print(f"decoded {len(names)} images")
    split_dir = Path("splits")
    if a.write_splits or not (split_dir / "train.txt").exists():
        sp = stratified_split(labels, seed=0)
        split_dir.mkdir(exist_ok=True)
        for k, v in sp.items():
            (split_dir / f"{k}.txt").write_text("\n".join(names[i] for i in v) + "\n")
        rep = duplicate_report(imgs, sp)
        (split_dir / "leakage_report.json").write_text(json.dumps(rep, indent=2))
        print("splits:", {k: len(v) for k, v in sp.items()}, "duplicates:", rep)


if __name__ == "__main__":
    main()
