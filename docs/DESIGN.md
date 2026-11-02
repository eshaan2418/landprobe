# Design

## Pipeline
```
EuroSAT zip (MD5) -> uint8 array (27k x 3 x 64 x 64, 317 MB) -> stratified splits (file lists in splits/)
   train images (no labels) -> GPU augment x2 -> ResNet-18 (3x3 stem + max-pool) -> SimCLR / BYOL loss
   frozen backbone -> features -> kNN / linear probe (wd chosen on val) at 1/5/10/100% labels
   backbone -> full fine-tune (fixed recipe) at 1/10/100% labels
```

## Decisions
1. **All augmentation on the GPU.** At 64x64, a CPU/PIL data loader cannot produce two
   augmented views fast enough to keep the MPS GPU busy. Random resized crop and the dihedral
   transform are one batched `affine_grid` + `grid_sample`. Colour jitter is vectorised RGB/HSV
   arithmetic. Tests check that a full-scale crop reproduces one of the 8 dihedral symmetries
   exactly, and that the HSV conversion matches `colorsys`.
2. **Dihedral augmentation.** Overhead imagery has no canonical orientation, so all 8
   rotations and reflections preserve the label. This is a domain-specific change to the
   SimCLR recipe, which uses horizontal flips only.
3. **Stem choice from measurement.** Training throughput at batch 256 on the M5:
   3x3 stride-1 stem without max-pool 407 img/s, with max-pool 1,303 img/s, stock 7x7/2 +
   pool 3,079 img/s (`scripts/bench_throughput.py`). The pooled 3x3 stem keeps 32x32 spatial
   resolution in layer1 for 64 px tiles at 3.2x the speed of the unpooled one.
   `channels_last` was slower on MPS in every configuration and is not used.
4. **AdamW for SSL.** SimCLR/BYOL papers use LARS at batch 4096. At batch 256 on one device,
   AdamW with cosine decay is the common small-scale substitute. Both methods use identical
   optimiser settings.
5. **Protocols.** kNN (weighted cosine, k=20) needs no training. The linear probe uses
   full-batch L-BFGS on standardised features, with L2 chosen on val from 5 values.
   Fine-tuning uses one fixed recipe for all backbones (no per-backbone tuning), so
   differences come from initialisation, not from tuning effort.
6. **Pairing.** SSL seed s is evaluated with label-subset seed s. Random-init seed s uses
   the same subset. Paired differences across seeds remove subset difficulty from the
   comparison.
