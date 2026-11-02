"""Batched GPU augmentations (no per-image Python loop, no CPU data loader).

At 64x64 the CPU cannot keep a GPU fed with two augmented views per image through PIL, so
every op works on a whole ``(B, 3, H, W)`` float batch on the device:

* random resized crop + dihedral transform (flips and 90° rotations) as one affine
  ``grid_sample`` per batch -- overhead imagery has no canonical "up", so all 8 dihedral
  transforms are label-preserving;
* colour jitter (brightness, contrast, saturation, hue) applied with probability ``p``;
* random grayscale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.3445, 0.3804, 0.4078])  # EuroSAT train split (computed, see tests)
STD = torch.tensor([0.2033, 0.1373, 0.1157])


@dataclass
class AugConfig:
    crop_scale: tuple[float, float] = (0.2, 1.0)
    crop_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    dihedral: bool = True
    jitter_p: float = 0.8
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1
    gray_p: float = 0.2


def to_float(x_u8: torch.Tensor) -> torch.Tensor:
    return x_u8.float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    m = MEAN.to(x.device).view(1, 3, 1, 1)
    s = STD.to(x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _uniform(n: int, lo: float, hi: float, g: torch.Generator, device: torch.device):
    return torch.rand(n, generator=g, device="cpu").to(device) * (hi - lo) + lo


def crop_dihedral(x: torch.Tensor, cfg: AugConfig, g: torch.Generator) -> torch.Tensor:
    """Random resized crop (+ optional dihedral transform) via one batched affine warp."""
    b, dev = x.shape[0], x.device
    area = _uniform(b, *cfg.crop_scale, g, dev)
    logr = _uniform(b, math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1]), g, dev)
    ratio = torch.exp(logr)
    w = torch.sqrt(area * ratio).clamp(max=1.0)  # fraction of width
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (_uniform(b, 0, 1, g, dev) * 2 - 1) * (1 - w)  # centre in [-1+w, 1-w]
    cy = (_uniform(b, 0, 1, g, dev) * 2 - 1) * (1 - h)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = w
    theta[:, 1, 1] = h
    theta[:, 0, 2] = cx
    theta[:, 1, 2] = cy
    if cfg.dihedral:
        k = torch.randint(0, 4, (b,), generator=g).to(dev)  # rotation by k * 90°
        flip = (torch.rand(b, generator=g) < 0.5).to(dev)
        cos = torch.cos(k * math.pi / 2).round()
        sin = torch.sin(k * math.pi / 2).round()
        rot = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
        fl = torch.ones(b, 2, device=dev)
        fl[flip, 0] = -1
        lin = theta[:, :, :2] @ rot * fl[:, None, :]
        theta = torch.cat([lin, theta[:, :, 2:]], dim=2)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def _gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)


def _rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp(min=1e-8), torch.zeros_like(maxc))
    dc = delta.clamp(min=1e-8)
    rc, gc, bc = (maxc - r) / dc, (maxc - g) / dc, (maxc - b) / dc
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, v], 1)


def _hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    h, s, v = x[:, 0], x[:, 1], x[:, 2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], 1).gather(1, i[:, None]).squeeze(1)
    g = torch.stack([t, v, v, q, p, p], 1).gather(1, i[:, None]).squeeze(1)
    b = torch.stack([p, p, t, v, v, q], 1).gather(1, i[:, None]).squeeze(1)
    return torch.stack([r, g, b], 1)


def color_jitter(x: torch.Tensor, cfg: AugConfig, g: torch.Generator) -> torch.Tensor:
    b, dev = x.shape[0], x.device
    apply = (torch.rand(b, generator=g) < cfg.jitter_p).to(dev).view(b, 1, 1, 1)
    y = x
    bf = _uniform(b, 1 - cfg.brightness, 1 + cfg.brightness, g, dev).view(b, 1, 1, 1)
    y = (y * bf).clamp(0, 1)
    cf = _uniform(b, 1 - cfg.contrast, 1 + cfg.contrast, g, dev).view(b, 1, 1, 1)
    mean = _gray(y).mean(dim=(1, 2, 3), keepdim=True)
    y = ((y - mean) * cf + mean).clamp(0, 1)
    sf = _uniform(b, 1 - cfg.saturation, 1 + cfg.saturation, g, dev).view(b, 1, 1, 1)
    gr = _gray(y)
    y = ((y - gr) * sf + gr).clamp(0, 1)
    if cfg.hue > 0:
        hsv = _rgb_to_hsv(y)
        hf = _uniform(b, -cfg.hue, cfg.hue, g, dev).view(b, 1, 1)
        hsv = torch.stack([(hsv[:, 0] + hf) % 1.0, hsv[:, 1], hsv[:, 2]], 1)
        y = _hsv_to_rgb(hsv).clamp(0, 1)
    out = torch.where(apply, y, x)
    gray = (torch.rand(b, generator=g) < cfg.gray_p).to(dev).view(b, 1, 1, 1)
    return torch.where(gray, _gray(out), out)


def augment(x_u8: torch.Tensor, cfg: AugConfig, g: torch.Generator) -> torch.Tensor:
    """uint8 batch -> normalised float view."""
    x = to_float(x_u8)
    x = crop_dihedral(x, cfg, g)
    if cfg.jitter_p > 0 or cfg.gray_p > 0:
        x = color_jitter(x, cfg, g)
    return normalize(x)


def eval_transform(x_u8: torch.Tensor) -> torch.Tensor:
    return normalize(to_float(x_u8))
