"""Backbones and heads.

``resnet18_small``: ResNet-18 with a 3x3 stride-1 stem followed by the usual max-pool, so a 64x64
tile enters layer1 at 32x32. A stride-1 stem without the pool is 3.6x slower on MPS
(335 vs 1,204 img/s training throughput at batch 256, eval/results/bench/throughput.json).

``resnet18_imagenet``: the stock torchvision ResNet-18 with ImageNet-1k weights (baseline).
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

FEAT_DIM = 512


def resnet18_small() -> nn.Module:
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.fc = nn.Identity()  # type: ignore[assignment]
    return m


def resnet18_imagenet() -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()  # type: ignore[assignment]
    return m


def mlp(dims: list[int], last_bn: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1], bias=False), nn.BatchNorm1d(dims[i + 1]),
                   nn.ReLU(inplace=True)]
    layers.append(nn.Linear(dims[-2], dims[-1], bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(dims[-1], affine=False))
    return nn.Sequential(*layers)


class SimCLR(nn.Module):
    def __init__(self, temperature: float = 0.2, proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = resnet18_small()
        self.projector = mlp([FEAT_DIM, 2048, proj_dim])
        self.t = temperature

    def loss(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.projector(self.backbone(v1)), self.projector(self.backbone(v2))])
        return nt_xent(z, self.t)


def nt_xent(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent over 2B embeddings where rows i and i+B are positives."""
    z = nn.functional.normalize(z.float(), dim=1)
    n = z.shape[0]
    b = n // 2
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(b, n), torch.arange(0, b)]).to(z.device)
    return nn.functional.cross_entropy(sim, target)


class BYOL(nn.Module):
    def __init__(self, proj_dim: int = 256, hidden: int = 2048) -> None:
        super().__init__()
        self.backbone = resnet18_small()
        self.projector = mlp([FEAT_DIM, hidden, proj_dim])
        self.predictor = mlp([proj_dim, hidden, proj_dim])
        self.t_backbone = copy.deepcopy(self.backbone)
        self.t_projector = copy.deepcopy(self.projector)
        for p in list(self.t_backbone.parameters()) + list(self.t_projector.parameters()):
            p.requires_grad_(False)

    def online_parameters(self) -> list[nn.Parameter]:
        return (list(self.backbone.parameters()) + list(self.projector.parameters())
                + list(self.predictor.parameters()))

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        for o, t in ((self.backbone, self.t_backbone), (self.projector, self.t_projector)):
            for po, pt in zip(o.parameters(), t.parameters(), strict=True):
                pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
            for bo, bt in zip(o.buffers(), t.buffers(), strict=True):
                bt.copy_(bo)

    def loss(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        p1 = self.predictor(self.projector(self.backbone(v1)))
        p2 = self.predictor(self.projector(self.backbone(v2)))
        with torch.no_grad():
            t1 = self.t_projector(self.t_backbone(v1))
            t2 = self.t_projector(self.t_backbone(v2))
        return byol_loss(p1, t2) + byol_loss(p2, t1)


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = nn.functional.normalize(p.float(), dim=1)
    z = nn.functional.normalize(z.float(), dim=1)
    return (2 - 2 * (p * z).sum(1)).mean()
