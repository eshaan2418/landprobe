# Data card: EuroSAT (RGB)

- **Source:** Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
  Land Cover Classification* (IEEE JSTARS 2019). Archive `EuroSAT_RGB.zip` from Zenodo record
  7711810 (MD5 `f46e308c4d50d4bf32fedad2d3d62f3b`, checked by `scripts/prepare_data.py`).
- **Licence:** MIT (as declared on the Zenodo record). The 60 tiles in
  `tests/fixtures/eurosat_tiny.npz` are redistributed under that licence. The imagery derives
  from Copernicus Sentinel-2 data.
- **Content:** 27,000 RGB tiles of 64x64 pixels (10 m ground resolution), 10 land-use classes
  (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
  Residential, River, SeaLake), 2,000-3,000 per class, from European cities.
- **Splits (this repo):** stratified 60/20/20 train/val/test (16,200 / 5,400 / 5,400) with seed 0,
  stored as file lists in `splits/`. Self-supervised pretraining uses **train images only,
  without labels**. Hyperparameters (linear-probe weight decay, ImageNet input size) are chosen
  on **val**. Test is used only for final numbers.
- **Label-scarce subsets:** per seed (0, 1, 2), class-stratified nested subsets of train with
  1% / 5% / 10% / 100% of the labels (162 / 810 / 1,620 / 16,200 tiles), at least one per class.
- **Leakage checks:** no byte-identical images within the dataset or across splits
  (`splits/leakage_report.json`).
- **Known limitation, not controlled:** the RGB release has no geolocation, so the random split
  can place tiles from the same Sentinel-2 scene, possibly adjacent ones, in both train and
  test. Spatial autocorrelation between train and test is a known source of optimistic
  accuracy in remote-sensing benchmarks. Absolute
  accuracies here are therefore optimistic. The *comparisons* between methods share the same
  split and are less affected, though methods that memorise scene appearance may benefit
  more than others.
- **Personal data:** none (10 m imagery cannot resolve individuals).
