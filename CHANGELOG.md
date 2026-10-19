# Changelog

## 0.4.0

- **New data:** 89 labelled images of 30 embryos from three culture dishes, each embryo at 25, 69 and 117 h after insemination, instead of the 205 frames of two time-lapse videos. Files are named `embryo<dish><P>_<hours>h.png` and `..._mask.png` (new default `data.image_pattern` / `mask_pattern`; Colab folder `MyDrive/g_embryos_dataset`).
- **Split by embryo** (`split.group_by_sample: true`, new default): 6 / 5 / 19 embryos for test / validation / training; all time points of an embryo stay in one subset.
- **5-fold cross-validation by embryo** (`cross_validation`, replacing `leave_one_embryo_out`; `RUN_CROSS_VALIDATION` in the notebook): every embryo is tested once. With one fold per embryo it is leave-one-embryo-out with the splits of 0.3.0. Outputs `cross_validation.csv`, `cross_validation_summary.csv`, `cross_validation_by_time.csv`, `splits/cv_fold*.csv`.
- **Per-image scores** (`per_image/<run>.csv`) and the scores per time point (`results_by_time.csv`).
- The `held_out_sample` column of the result tables is replaced by `fold` and `test_samples` (the test embryos).

## 0.3.0

- PyTorch Lightning implementation (`zona_pellucida/backends/pytorch`) next to the Keras one, behind a common backend interface; `FRAMEWORK` switch in the notebook, `framework` in the config, one single-file export per framework. The PyTorch network is tested to give the same output as the Keras model with the same weights.
- Per-framework output folders, a `framework` column in the results, and the git commit of the code (when run from a git clone) in every environment report.
- The training images are reshuffled every epoch by default (`train.shuffle: true`, reproducible from the seed).

## 0.2.0 (changes since the thesis notebook)

Code moved from the single notebook into the `zona_pellucida` package (config file, tests); the notebook runs the package step by step.

**Changes that affect the results**
- Train / validation / test split (131 / 33 / 41, stratified by video) instead of 164 / 41 with the test set used as validation data. The kept epoch (best validation IoU, all 100 epochs trained) and the decision threshold are chosen on the validation set only; the training order stays random, as before.
- Masks are resized with nearest-neighbour interpolation and binarised (the default bicubic filter created grey edge values).
- IoU and Dice are computed against the binarised ground truth (the thesis IoU counted any mask value above 0 as zona pellucida; the thesis Dice counted grey edge pixels fractionally, by their grey value).
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
