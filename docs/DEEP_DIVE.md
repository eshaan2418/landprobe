# Deep dive: questions and answers

### 1. What is the one-sentence answer to "should I pretrain on my own unlabelled tiles?"
If you will freeze the encoder and have only ~150 labels, yes: 20 minutes of SimCLR beat ImageNet features (82.4% vs 80.7%, paired +1.6 ± 0.6). If you can fine-tune, or have ≥ 800 labels, start from ImageNet weights, which won every fine-tuning comparison by 2–8 points. Either form of pretraining beats training from scratch at every label budget.

### 2. Why might in-domain SSL win frozen but lose fine-tuned?
Frozen features are only as good as their invariances. SimCLR here was trained with augmentations chosen for overhead imagery (all 8 dihedral symmetries, aggressive colour jitter) on the exact distribution being classified, so a linear head on its features separates EuroSAT classes well with very few labels. ImageNet features encode natural-image priors (an "up" direction, object-centric crops). Fine-tuning lets the network re-learn invariances, and then the ImageNet model's far richer starting point (1.28M labelled images) dominates.

### 3. Is the ImageNet comparison fair?
It is fair to the question "reuse public weights or pretrain yourself?", but it is not a compute- or data-matched comparison. ImageNet weights came from about 80× more images with labels and far more compute. The ImageNet model also got a validation-chosen 128 px input, which costs 1.6× the fine-tuning time per step. Its stem differs (7×7/2 vs our 3×3 + pool) because changing it would discard pretrained weights. All of this is stated next to the numbers.

### 4. Why did BYOL do so much worse than SimCLR?
BYOL's kNN validation accuracy rose more slowly from the start (66% at epoch 5 against SimCLR's 78%) and ended 11 points lower. BYOL relies on the slow EMA target to bootstrap, and published results use hundreds of epochs. Our 40 epochs (~2.5k steps) is about 1/40 of a standard run. With AdamW at batch 256 rather than LARS at 4096, the hyperparameters are also further from BYOL's tuned regime than SimCLR's. Treat this as "BYOL needs more budget", not "BYOL is worse".

### 5. Why do all augmentations run on the GPU?
At 64×64, a training step on the M5 takes ~0.5 s for 512 views. PIL augmentation on CPU workers could not keep up without many processes, and the per-process memory limit made that unattractive. Crop plus dihedral transform is a single batched `affine_grid`/`grid_sample`, and colour jitter is vectorised RGB/HSV arithmetic. A test confirms that a full-scale "crop" reproduces one of the 8 symmetries exactly, which validates the affine composition.

### 6. How were hyperparameters kept off the test set?
The linear-probe L2 penalty is chosen per run from 5 values by validation accuracy. The ImageNet input size (64 vs 128) is chosen by mean validation probe accuracy at 10% labels. The SSL settings are fixed a priori (common small-scale defaults), and the fine-tuning recipe is fixed for all backbones. Test accuracy is computed once per configuration.

### 7. What does the paired design buy?
Seed s of every method uses label subset s, so subset difficulty (which 162 tiles you happened to label) cancels in the differences. Unpaired, the 1% SimCLR-vs-ImageNet probe difference would carry both methods' ±2-point subset variance. Paired, it is +1.6 ± 0.6.

### 8. What limits the absolute numbers?
The EuroSAT RGB split is random, not spatially disjoint, so neighbouring tiles from the same scene can appear in train and test, which inflates every method's accuracy. The inputs are 64 px RGB only, while Sentinel-2's other bands (near-infrared especially) carry most of the vegetation signal the confusion matrix shows the model struggling with.

### 9. What do the errors say?
The dominant confusions are AnnualCrop ↔ PermanentCrop ↔ HerbaceousVegetation, then River → Highway. Crops and grassland differ in texture and phenology. Rivers and highways are both thin linear features. Forest and Residential are nearly solved. Adding the near-infrared band, or multi-date imagery, would target exactly the confusable classes.

### 10. How much compute did this take and where did it go?
3.8 GPU-hours on the M5: six 40-epoch pretraining runs (2.2 h; SimCLR 19 min, BYOL 25 min each), 36 fine-tuning runs (1.6 h), and frozen-feature extraction for probes (2 min). Peak MPS memory was ~7 GB per pretraining run.

### 11. With 100x compute?
Pretrain for 400+ epochs on a much larger unlabelled pool (e.g. the full multispectral EuroSAT or BigEarthNet), with LARS at large batch. Evaluate on a spatially disjoint split. Add MAE- and DINO-style objectives and an ImageNet-initialised-then-SSL-continued variant (the usual best practice). Tune fine-tuning per backbone with equal budgets.
