# Model card

## Models
- **SimCLR and BYOL ResNet-18 encoders** (11.2M parameters, 3x3 stem + max-pool, 512-d features),
  pretrained from scratch for 40 epochs on the 16,200 unlabelled EuroSAT train tiles, 3 seeds
  each. Weights are not committed. `make train` reproduces them (~21 min per run on an Apple M5).
- **Baselines:** random-init ResNet-18 (frozen features or trained from scratch), and
  torchvision's ImageNet-1k ResNet-18 (`IMAGENET1K_V1`, BSD-3-Clause code; ImageNet weights are
  commonly used for research, so check your own terms for commercial use).

## Intended use
Feature extractors for 10-class land-use classification of 64x64 Sentinel-2 RGB tiles when only a
few hundred labels are available. Also a reproducible reference for label-efficiency studies on
Apple silicon.

## Out of scope
Imagery from other sensors or resolutions, other regions without validation, any
individual-level or surveillance use, and decisions with legal or financial consequences (zoning,
insurance) without human review.

## Evaluation
Linear probe, kNN and fine-tuning at 1/5/10/100% labels. Mean ± 95% CI over 3 seeds. See README.

## Limitations and biases
- Trained on European scenes only. Expect degraded accuracy elsewhere (different crops,
  architecture, vegetation).
- The random split is not spatially disjoint (see DATA_CARD), so absolute accuracy is
  optimistic.
- Small pretraining budget (40 epochs, 16k images). Conclusions may not hold at larger scale.
