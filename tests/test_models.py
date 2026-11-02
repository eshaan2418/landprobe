import math

import torch

from landprobe.models import BYOL, SimCLR, byol_loss, nt_xent, resnet18_small


def nt_xent_reference(z, t):
    z = torch.nn.functional.normalize(z, dim=1)
    n = len(z)
    b = n // 2
    total = 0.0
    for i in range(n):
        j = (i + b) % n
        num = math.exp(float(z[i] @ z[j]) / t)
        den = sum(math.exp(float(z[i] @ z[k]) / t) for k in range(n) if k != i)
        total += -math.log(num / den)
    return total / n


def test_nt_xent_matches_reference():
    torch.manual_seed(0)
    z = torch.randn(10, 7)
    assert abs(float(nt_xent(z, 0.3)) - nt_xent_reference(z, 0.3)) < 1e-4


def test_nt_xent_minimised_by_aligned_pairs():
    torch.manual_seed(0)
    a = torch.randn(8, 16)
    aligned = torch.cat([a, a])
    random = torch.cat([a, torch.randn(8, 16)])
    assert nt_xent(aligned, 0.2) < nt_xent(random, 0.2)


def test_byol_loss_range():
    p = torch.randn(5, 4)
    assert abs(float(byol_loss(p, p))) < 1e-6
    assert abs(float(byol_loss(p, -p)) - 4.0) < 1e-5


def test_backbone_shape_and_grad():
    m = resnet18_small()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 512)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_byol_ema_update():
    m = BYOL(proj_dim=8, hidden=16)
    with torch.no_grad():
        for p in m.backbone.parameters():
            p.add_(1.0)
    before = [p.clone() for p in m.t_backbone.parameters()]
    m.update_target(0.9)
    for b, t, o in zip(before, m.t_backbone.parameters(), m.backbone.parameters(), strict=True):
        assert torch.allclose(t, 0.9 * b + 0.1 * o, atol=1e-6)
    assert all(not p.requires_grad for p in m.t_backbone.parameters())


def test_ssl_losses_backprop():
    x1, x2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    for m in (SimCLR(), BYOL(proj_dim=8, hidden=16)):
        loss = m.loss(x1, x2)
        loss.backward()
        assert torch.isfinite(loss)
        assert m.backbone.conv1.weight.grad is not None
