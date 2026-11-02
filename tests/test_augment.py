import colorsys

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from landprobe.augment import (
    AugConfig,
    _hsv_to_rgb,
    _rgb_to_hsv,
    augment,
    color_jitter,
    crop_dihedral,
    normalize,
    to_float,
)


def _batch(b=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, 3, 16, 16), generator=g, dtype=torch.uint8)


def test_shapes_and_determinism():
    x = _batch()
    cfg = AugConfig()
    a = augment(x, cfg, torch.Generator().manual_seed(1))
    b = augment(x, cfg, torch.Generator().manual_seed(1))
    c = augment(x, cfg, torch.Generator().manual_seed(2))
    assert a.shape == (8, 3, 16, 16) and a.dtype == torch.float32
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_full_crop_dihedral_is_exact_symmetry():
    """With crop scale 1 and ratio 1 the warp must be one of the 8 dihedral transforms."""
    x = to_float(_batch(16))
    cfg = AugConfig(crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), jitter_p=0, gray_p=0)
    y = crop_dihedral(x, cfg, torch.Generator().manual_seed(0))
    for i in range(len(x)):
        cands = []
        for k in range(4):
            r = torch.rot90(x[i], k, dims=(1, 2))
            cands += [r, torch.flip(r, dims=(2,))]
        err = min(float((y[i] - c).abs().max()) for c in cands)
        assert err < 1e-5, err


def test_identity_when_disabled():
    x = to_float(_batch())
    cfg = AugConfig(crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), dihedral=False, jitter_p=0,
                    gray_p=0)
    y = crop_dihedral(x, cfg, torch.Generator().manual_seed(0))
    assert torch.allclose(x, y, atol=1e-5)
    assert torch.equal(color_jitter(x, cfg, torch.Generator().manual_seed(0)), x)


def test_jitter_range():
    x = to_float(_batch(64))
    y = color_jitter(x, AugConfig(jitter_p=1.0, gray_p=0.5), torch.Generator().manual_seed(0))
    assert float(y.min()) >= 0 and float(y.max()) <= 1


@settings(max_examples=200, deadline=None)
@given(st.tuples(*[st.floats(0, 1, width=32, allow_subnormal=False)] * 3))
def test_hsv_roundtrip_matches_colorsys(rgb):
    x = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1)
    hsv = _rgb_to_hsv(x).view(3).tolist()
    ref = colorsys.rgb_to_hsv(*rgb)
    if ref[1] > 1e-4 and ref[2] > 1e-4:
        dh = abs(hsv[0] - ref[0])
        assert min(dh, 1 - dh) < 1e-4
    assert abs(hsv[1] - ref[1]) < 1e-4 and abs(hsv[2] - ref[2]) < 1e-4
    back = _hsv_to_rgb(_rgb_to_hsv(x)).view(3).numpy()
    np.testing.assert_allclose(back, rgb, atol=1e-4)


def test_normalize_constants_match_train_split():
    from pathlib import Path

    import pytest

    if not Path("data/eurosat_images.npy").exists():
        pytest.skip("dataset not prepared")
    from landprobe.data import EuroSAT

    d = EuroSAT.load()
    idx = np.random.default_rng(0).choice(d.splits["train"], 4000, replace=False)
    x = to_float(torch.from_numpy(d.images[np.sort(idx)]))
    z = normalize(x)
    assert z.mean(dim=(0, 2, 3)).abs().max() < 0.05
    assert (z.std(dim=(0, 2, 3)) - 1).abs().max() < 0.05
