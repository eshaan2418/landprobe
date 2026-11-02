"""python scripts/pretrain.py --method simclr --seed 0 [--epochs 40]"""

import argparse

from landprobe.data import EuroSAT
from landprobe.pretrain import PretrainConfig, run

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["simclr", "byol"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--name", default=None)
a = ap.parse_args()
name = a.name or f"{a.method}_e{a.epochs}_s{a.seed}"
run(PretrainConfig(name=name, method=a.method, seed=a.seed, epochs=a.epochs), EuroSAT.load())
