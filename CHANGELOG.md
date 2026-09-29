# Changelog

## 0.3.0

- PyTorch Lightning implementation (`zona_pellucida/backends/pytorch`) next to the Keras one, behind a common backend interface; `FRAMEWORK` switch in the notebook, `framework` in the config, one single-file export per framework. The PyTorch network is tested to give the same output as the Keras model with the same weights.
- Per-framework output folders, a `framework` column in the results, and the git commit of the code in every environment report.
- The training images are reshuffled every epoch by default (`train.shuffle: true`, reproducible from the seed).

## 0.2.0 (changes since the thesis notebook)

Code moved from the single notebook into the `zona_pellucida` package (config file, tests); the notebook runs the package step by step.

**Changes that affect the results**
- Train / validation / test split (131 / 33 / 41, stratified by video) instead of 164 / 41 with the test set used as validation data. The kept epoch (best validation IoU, all 100 epochs trained) and the decision threshold are chosen on the validation set only; the training order stays random, as before.
- Masks are resized with nearest-neighbour interpolation and binarised (the default bicubic filter created grey edge values).
- IoU and Dice are computed against the binarised ground truth (any value above 0 used to count as zona pellucida).
- Frequency-weighted binary cross-entropy (inverse frequency, or median frequency) next to the unweighted baseline; foreground IoU, Dice, precision and recall monitored during training.
- One seed for everything and deterministic ops; the split is saved to `split.csv`.
- Leave-one-embryo-out robustness check.

**Fixes**
- Images and masks are paired on sample and frame number (they were paired by position in two sorted lists).
- `model.evaluate` runs once and its values are unpacked by name (the accuracy was stored as `test_loss`, and the test set was evaluated three times).
- `random.randint` upper bounds (could index past the end of the array); random test images come from the whole test set (only the first 20% could be chosen); each figure draws its images from its own seeded generator.
- The distance-transform overlay is shown in RGB (it was shown in BGR).
- The trained model is saved next to the results (it was saved to the ephemeral Colab disk and loaded from Drive).
- The pinned `tensorflow==2.10.0` cannot be installed on current Python versions; the preinstalled TensorFlow/Keras 3 of Colab is used and all versions are recorded.
- The unused `pool5` layer was removed from the model (no effect on the network).
