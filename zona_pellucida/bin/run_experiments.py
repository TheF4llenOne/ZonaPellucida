"""Run the full pipeline outside the notebook.

    python -m zona_pellucida.bin.run_experiments --config my_config.yaml

`my_config.yaml` is a copy of zona_pellucida/configs/config.yaml with
`framework`, `data.image_dir` and `data.mask_dir` set (keep `output_dir:
outputs`, which git ignores: the results list the dataset's file names). Loads
and splits the data, writes the class-balance report, trains the
unweighted and weighted models, and runs the cross-validation by embryo.
Everything is written to <output_dir>/<framework>.

    python -m zona_pellucida.bin.run_experiments --cv-fold 3 \
        --config <output_dir>/<framework>/config_used.yaml

runs only cross-validation fold 3 (repeat --cv-fold for more) and rebuilds
the cross-validation summaries from all folds saved so far; nothing else is
trained or written. One fold per process gives the same files as one run of
all folds, and keeps Keras within the memory of a 16 GB GPU, which it
outgrows after a few trainings in one process.
"""

import argparse
import logging
from pathlib import Path

import matplotlib

from zona_pellucida.backends import load_backend
from zona_pellucida.config import load_config, save_config
from zona_pellucida.data.class_balance import (
    compute_class_weights,
    save_class_balance,
)
from zona_pellucida.data.loading import load_dataset_from_config
from zona_pellucida.data.splitting import save_split, train_val_test_split
from zona_pellucida.experiments import (
    run_comparison,
    run_cross_validation,
    summarize_cross_validation,
)
from zona_pellucida.reproducibility import log_versions

logger = logging.getLogger(__name__)


def main(config_path=None, cv_folds=None):
    cfg = load_config(config_path)
    # Each framework gets its own folder, so results are never mixed
    # (configs saved with the results already end in it).
    if Path(cfg.output_dir).name != cfg.framework:
        cfg.output_dir = str(Path(cfg.output_dir) / cfg.framework)
    backend = load_backend(cfg.framework)
    backend.prepare_device()
    backend.start_run(cfg.seed, cfg.deterministic_ops)
    if cv_folds:
        # Only these folds; every model is saved with its config and
        # environment, the files of the main run are left alone.
        dataset = load_dataset_from_config(cfg.data)
        _, cv_table = run_cross_validation(dataset, cfg, backend,
                                           folds=cv_folds)
        logger.info("Cross-validation by embryo (folds saved so far):\n%s",
                    summarize_cross_validation(cv_table).to_string())
        return
    output_dir = Path(cfg.output_dir)
    log_versions(backend, output_dir / "environment.yaml")
    save_config(cfg, output_dir / "config_used.yaml")

    dataset = load_dataset_from_config(cfg.data)
    split = train_val_test_split(
        dataset.samples, cfg.split.test_size, cfg.split.val_size, cfg.seed,
        cfg.split.stratify_by_sample, cfg.split.group_by_sample,
    )
    save_split(dataset, split, output_dir / "split.csv")

    train = dataset.subset(split.train)
    save_class_balance(train.masks, train.image_files, output_dir)
    compute_class_weights(train.masks, cfg.class_weighting.method)

    _, table = run_comparison(dataset, split, cfg, backend)
    logger.info("Results:\n%s", table.to_string())

    if cfg.cross_validation.enabled:
        _, cv_table = run_cross_validation(dataset, cfg, backend)
        logger.info("Cross-validation by embryo:\n%s",
                    summarize_cross_validation(cv_table).to_string())


if __name__ == "__main__":
    # Figures are only saved (no window) when running from the command line.
    matplotlib.use("Agg")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="YAML config (default: packaged config.yaml)")
    parser.add_argument("--cv-fold", type=int, action="append",
                        dest="cv_folds", metavar="K",
                        help="run only cross-validation fold K (repeatable)")
    args = parser.parse_args()
    main(args.config, args.cv_folds)
