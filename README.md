# Zona pellucida segmentation (U-Net-style FCN)

Semantic segmentation of the zona pellucida in time-lapse images of human embryos, with a U-Net-style fully convolutional network. Developed for a thesis; implemented in **PyTorch Lightning** (default) and **TensorFlow/Keras** (the thesis version), which share the data pipeline, class weighting, metrics and figures.

### Features

- **Two frameworks, one pipeline**: a framework-neutral core (data, splits, class weights, metrics, figures, experiments) and two backends with the same interface (`zona_pellucida/backends/{pytorch,keras}`). The PyTorch network is a layer-for-layer port of the Keras model; a test copies the Keras weights into it and checks that both give the same output.
- **Class imbalance**: the zona pellucida is ~5% of the pixels. Frequency-normalised class weights (inverse frequency or SegNet median frequency) and a per-pixel weighted binary cross-entropy; foreground IoU, Dice, precision and recall instead of pixel accuracy alone.
- **Sound evaluation**: train / validation / test split (131 / 33 / 41, stratified by video); best epoch and decision threshold chosen on validation only; a leave-one-embryo-out check, since frames of one video are highly correlated.
- **Reproducibility**: one seed for everything, deterministic GPU ops (bit-exact reruns tested on CPU and on a CUDA GPU with PyTorch), saved split, and the config and environment (library versions, GPU, CUDA/cuDNN) stored next to every model.

### Model

- Input: 512x512 grayscale image. Output: one zona pellucida probability per pixel (the PyTorch model outputs logits, with the sigmoid inside the loss).
- Contracting path of five blocks of two 3x3 convolutions (32 to 512 filters, ReLU, batch normalisation, max pooling); expanding path of four 2x2 transposed convolutions with skip connections; dropout 0.5; 1x1 output convolution.
- Loss: binary cross-entropy (baseline) or frequency-weighted binary cross-entropy. Adam (learning rate 0.001), batch size 16, 100 epochs; the weights of the epoch with the best validation IoU are kept (optional early stopping in the config).

### Code structure

```
fcn.ipynb                 # Colab notebook: the whole pipeline step by step, FRAMEWORK = 'pytorch' | 'keras'
fcn_pipeline_pytorch.py   # single-file exports of notebook + package, generated
fcn_pipeline_keras.py     # (the Keras one is the thesis appendix)
CHANGELOG.md              # changes since the thesis version
zona_pellucida/
├── configs/config.yaml   # all settings (framework, seed, paths, split, training, weighting, evaluation)
├── config.py             # YAML -> typed config
├── reproducibility.py    # seeding, determinism errors, environment report
├── data/                 # loading & pairing, splits (incl. leave-one-embryo-out), class balance & weights
├── evaluation/           # test metrics, validation threshold selection
├── experiments.py        # unweighted vs weighted runs, leave-one-embryo-out (framework-neutral)
├── plots.py              # figures
├── backends/
│   ├── pytorch/          # UNet nn.Module, weighted BCE-with-logits, LightningModule (torchmetrics), Trainer + callbacks
│   └── keras/            # fcn_8, weighted BCE, Dice/IoU metrics, fit + callbacks
└── bin/                  # run_experiments.py (whole pipeline), export_pipeline.py
tests/                    # pytest on synthetic data: shared, pytorch/, keras/
```

### Running

**Colab:** put the data in `MyDrive/images` and `MyDrive/masks`, and the code in `MyDrive/ZonaPellucida`: either copy the repository folder there, or let the notebook clone it (this needs the branch named in `REPO_BRANCH` pushed to GitHub, with the `zona_pellucida` package committed). Open `fcn.ipynb`, choose `FRAMEWORK` in the first cell and run it from the top. PyTorch and TensorFlow are preinstalled on Colab; the notebook installs PyTorch Lightning and torchmetrics. Settings are in the first cell (`SEED`, paths, `SHUFFLE`, `CLASS_WEIGHTING`, ...); everything else is in `configs/config.yaml` and can be overridden with `CONFIG_OVERRIDES`. Results go to `MyDrive/ZonaPellucida_outputs/<framework>`, outside the code folder. Restart the runtime after changing `FRAMEWORK`.

**Locally:** copy `zona_pellucida/configs/config.yaml` to `my_config.yaml` and set `framework`, `data.image_dir`, `data.mask_dir` and `output_dir` (the defaults are the Colab Drive paths), then

```
uv sync --python 3.12 --extra pytorch     # or --extra keras (or both)
uv run python -m zona_pellucida.bin.run_experiments --config my_config.yaml
```

On Windows, PyPI only has CPU builds of PyTorch. For an NVIDIA GPU, replace it after `uv sync` (a later `uv sync` switches back to the CPU build):

```
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Outputs (in `<output_dir>/<framework>`): `config_used.yaml`, `environment.yaml` (with `models/<run>.{config,environment}.yaml` per model), `split.csv`, `splits/loeo_test_sample*.csv`, `class_balance.{png,csv}`, `class_balance_per_image.csv`, `results_comparison.csv`, `leave_one_embryo_out.csv`, `models/<run>.ckpt` (PyTorch) or `models/<run>.keras`, `histories/*.{pkl,csv}`, `thresholds/*.csv`, `logs/` (TensorBoard). The notebook also writes the thesis figures to `figures/` (`fig5_*.png` are named after the figures of Chapter 5).

### Reproducibility

- One `SEED` (default 42) seeds Python, NumPy and the framework. Deterministic ops are enabled (PyTorch: `torch.use_deterministic_algorithms` and Lightning's `deterministic=True`; Keras: `enable_op_determinism`). If an op has no deterministic GPU kernel, training stops with a `DeterminismError` naming the op.
- Every run starts from a fresh, re-seeded session, so the unweighted and weighted models start from the same weights. The figure cells draw their images from generators seeded with `SEED`.
- The split (with the training order) is saved to `split.csv`. Library versions, the GPU, its driver and the CUDA/cuDNN versions are printed and saved with the results. Bit-exact reruns need the same versions and the same GPU type (Colab assigns A100, L4 or T4 per session; A100 and L4 use TF32). The two frameworks give different numbers for the same seed.

### Tests

```
uv run pytest            # fast unit tests
uv run pytest -m slow    # tiny trainings: determinism, Keras/PyTorch equivalence, notebook and exports end to end
```

Tests of a framework that is not installed are skipped. After changing the notebook or the package, regenerate the single-file exports with `uv run python -m zona_pellucida.bin.export_pipeline` (a test fails while they are stale). New package modules must be listed in the `export_modules` metadata of the first notebook cell that uses them.

### Results

`fcn.ipynb` writes the test-set metrics of both models (unweighted and weighted loss; accuracy, precision, recall, F1, IoU, Dice, at the 0.5 and the validation-selected threshold) to `results_comparison.csv`, the leave-one-embryo-out numbers to `leave_one_embryo_out.csv`, and the figures to `figures/`. The thesis reports the Keras results (Chapter 5).

### Overfitting Issues

Due to the limited size of the dataset, the model faces overfitting issues. The dataset consists of 205 images with corresponding masks, sourced from two videos:

- 105 images from a video of one embryo
- 100 images from a video of another embryo

The relatively small size and limited diversity of the dataset contribute to the model's tendency to overfit. Some solutions are provided in the Future work section.

### Future Work

- Collecting a larger and more diverse dataset to improve the model's robustness and generalization capabilities.
- Additional Labels: The model will be extended to segment additional structures of the embryo, requiring the use of multi-class segmentation techniques.
- Model Improvements: Exploring advanced architectures like DeepLab for improved performance.
- Data Augmentation: Implementing advanced augmentation techniques to increase dataset variability.
- Regularization: L2 weight decay and stronger augmentation (the model already uses dropout).
- Cross-Validation: Implement cross-validation to better evaluate model performance and generalizability. A leave-one-embryo-out check (train on one video, test on the other) is included.

### Dataset

The dataset used for training and evaluation consists of time-lapse images of human embryos, with manual annotations for the zona pellucida. Due to privacy and ethical considerations, the dataset cannot be shared publicly. However, instructions for replicating the dataset preparation process can be provided upon request.

### Contributing

If you would like to contribute to the project, please fork the repository and create a pull request with your proposed changes. Any contributions towards improving the model, adding new features, or fixing bugs are welcome!

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contact

For questions or inquiries, please contact John Dimitriadis at thef4llen1994@gmail.com.
