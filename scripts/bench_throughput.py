"""Training throughput of ResNet-18 variants on this machine (justifies the stem choice).

Writes eval/results/bench/throughput.json. Run when the GPU is otherwise idle.
"""

import json
import time
from pathlib import Path

import torch
from torchvision.models import resnet18

from landprobe.util import device, git_commit, thermal_state


def make(stem: str) -> torch.nn.Module:
    m = resnet18(num_classes=10)
    if stem in ("3x3_s1_nopool", "3x3_s1_pool"):
        m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    if stem == "3x3_s1_nopool":
        m.maxpool = torch.nn.Identity()  # type: ignore[assignment]
    return m


def main() -> None:
    dev = device()
    rows = []
    for stem in ("3x3_s1_nopool", "3x3_s1_pool", "torchvision_7x7_s2_pool"):
        for cl in (False, True):
            m = make(stem).to(dev)
            x = torch.randn(256, 3, 64, 64, device=dev)
            if cl:
                m = m.to(memory_format=torch.channels_last)
                x = x.to(memory_format=torch.channels_last)
            y = torch.randint(0, 10, (256,), device=dev)
            opt = torch.optim.SGD(m.parameters(), 0.1)
            for i in range(13):
                if i == 3:
                    torch.mps.synchronize() if dev.type == "mps" else None
                    t = time.perf_counter()
                loss = torch.nn.functional.cross_entropy(m(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.mps.synchronize() if dev.type == "mps" else None
            ips = 256 * 10 / (time.perf_counter() - t)
            rows.append({"stem": stem, "channels_last": cl, "train_img_per_s": ips})
            print(rows[-1], flush=True)
    out = Path("eval/results/bench")
    out.mkdir(parents=True, exist_ok=True)
    (out / "throughput.json").write_text(json.dumps(
        {"rows": rows, "batch": 256, "input": 64, "device": str(dev),
         "git_commit": git_commit(), "thermal": thermal_state()}, indent=2))


if __name__ == "__main__":
    main()
