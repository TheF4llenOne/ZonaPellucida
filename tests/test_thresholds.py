import numpy as np

from zona_pellucida.config import EvaluationConfig
from zona_pellucida.evaluation.thresholds import (
    select_threshold,
    threshold_grid,
)


def test_grid_contains_exactly_half():
    grid = threshold_grid(EvaluationConfig())
    assert len(grid) == 19
    assert 0.5 in grid.tolist()


def test_selects_best_threshold():
    y_true = np.array([0, 0, 1, 1], dtype=float).reshape(1, 2, 2, 1)
    # Foreground probabilities are 0.3 and 0.35: only thresholds < 0.3
    # separate the classes perfectly.
    y_prob = np.array([0.1, 0.2, 0.3, 0.35]).reshape(1, 2, 2, 1)
    threshold, table = select_threshold(
        y_true, y_prob, [0.1, 0.25, 0.5], metric="dice")
    assert threshold == 0.25
    assert table.loc[table["threshold"] == 0.25, "dice"].item() == 1.0


def test_ties_go_to_the_threshold_closest_to_half():
    y_true = np.array([0, 1], dtype=float).reshape(1, 1, 2, 1)
    y_prob = np.array([0.05, 0.95]).reshape(1, 1, 2, 1)
    threshold, _ = select_threshold(y_true, y_prob, [0.1, 0.45, 0.8])
    assert threshold == 0.45
