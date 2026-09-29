import os
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from tests.synthetic import set_parameters, synthetic_parameters
from zona_pellucida.bin.export_pipeline import NOTEBOOK, REPO_ROOT


def _notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def test_notebook_is_valid_and_has_no_stale_outputs():
    nb = _notebook()
    nbformat.validate(nb)
    for cell in _code_cells(nb):
        assert cell.outputs == []
        assert cell.execution_count is None


def test_seed_cell_comes_first():
    first = _code_cells(_notebook())[0]
    assert "parameters" in first.metadata.get("tags", [])
    assert first.source.splitlines()[1].startswith("SEED = 42")
    assert first.source.splitlines()[2].startswith("FRAMEWORK = 'pytorch'")
    for call in ("os.environ['PYTHONHASHSEED']", "random.seed(SEED)",
                 "np.random.seed(SEED)",
                 # PyTorch
                 "torch.manual_seed(SEED)",
                 "torch.use_deterministic_algorithms(DETERMINISTIC_OPS)",
                 "CUBLAS_WORKSPACE_CONFIG",
                 # Keras
                 "set_random_seed(SEED)", "enable_op_determinism()"):
        assert call in first.source


def test_code_cells_compile():
    for cell in _code_cells(_notebook()):
        code = "\n".join(line for line in cell.source.splitlines()
                         if not line.lstrip().startswith(("%", "!")))
        compile(code, "<cell>", "exec")


def test_no_hard_coded_random_state():
    for cell in _code_cells(_notebook()):
        assert "random_state=0" not in cell.source
        assert "randint(0, len(X_train))" not in cell.source
        assert "num_test_images" not in cell.source


@pytest.mark.slow
def test_notebook_runs_end_to_end(framework, synthetic_dirs, tmp_path):
    from nbclient import NotebookClient

    nb = _notebook()
    nb.cells = [c for c in nb.cells
                if "colab" not in c.metadata.get("tags", [])]
    first = _code_cells(nb)[0]
    first.source = set_parameters(
        first.source,
        FRAMEWORK=framework,
        **synthetic_parameters(*synthetic_dirs, tmp_path / "outputs"),
    )
    os.environ["MPLBACKEND"] = "Agg"
    NotebookClient(
        nb, timeout=1800, kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    ).execute()
    outputs = Path(tmp_path / "outputs")
    table = pd.read_csv(outputs / "results_comparison.csv")
    assert set(table["run"]) == {"unweighted", "weighted"}
    for name in ("leave_one_embryo_out.csv", "environment.yaml",
                 "class_balance.png", "split.csv"):
        assert (outputs / name).exists(), name
    suffix = {"pytorch": ".ckpt", "keras": ".keras"}[framework]
    assert (outputs / "models" / f"weighted{suffix}").exists()
    thesis_figures = [
        "fig5_1_train_sample", "fig5_3_accuracy_weighted",
        "fig5_4_loss_weighted", "fig5_5_confusion_matrix",
        "fig5_6_lowest_dice", "fig5_7_median_dice", "fig5_8_highest_dice",
        "fig5_9_random_test_1", "fig5_9_random_test_4",
        "fig5_10_prediction_overlay",
    ]
    for name in thesis_figures:
        assert (outputs / "figures" / f"{name}.png").exists(), name
