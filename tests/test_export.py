import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.synthetic import set_parameters, synthetic_parameters
from zona_pellucida.backends import FRAMEWORKS
from zona_pellucida.bin.export_pipeline import (
    NOTEBOOK,
    OUTPUTS,
    REPO_ROOT,
    _for_framework,
    _resolve_framework,
    build_pipeline,
)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_export_is_up_to_date(framework):
    assert OUTPUTS[framework].read_text(encoding="utf-8") == build_pipeline(
        framework), "Run: python -m zona_pellucida.bin.export_pipeline"


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_package_module_is_exported(framework):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    listed = {m for c in cells
              for m in c.get("metadata", {}).get("export_modules", [])}
    package = REPO_ROOT / "zona_pellucida"
    modules = {
        ".".join(p.relative_to(REPO_ROOT).with_suffix("").parts)
        for p in package.rglob("*.py")
        if p.name != "__init__.py" and "bin" not in p.parts
    }
    assert modules == listed
    code = build_pipeline(framework)
    other = next(f for f in FRAMEWORKS if f != framework)
    assert f"zona_pellucida/backends/{framework}/" in code
    assert f"zona_pellucida/backends/{other}/" not in code


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_only_the_chosen_framework_is_used(framework):
    code = build_pipeline(framework)
    assert f"FRAMEWORK = {framework!r}" in code
    assert "if FRAMEWORK ==" not in code
    imports = "\n".join(line for line in code.splitlines()
                        if line.startswith(("import ", "from ")))
    if framework == "pytorch":
        assert "import torch" in imports and "tensorflow" not in imports
    else:
        assert "import tensorflow" in imports and "torch" not in imports


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_export_runs_without_file_variable(framework):
    # Pasted into a notebook cell there is no __file__.
    code = build_pipeline(framework)
    assert re.search(r"(?<![\"'])__file__(?![\"'])", code) is None


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_export_has_no_package_imports(framework):
    assert "zona_pellucida." not in "\n".join(
        line for line in build_pipeline(framework).splitlines()
        if line.startswith(("import ", "from "))
    )


@pytest.mark.slow
def test_export_runs_end_to_end(framework, synthetic_dirs, tmp_path):
    source = set_parameters(
        OUTPUTS[framework].read_text(encoding="utf-8"),
        **synthetic_parameters(*synthetic_dirs, tmp_path / "outputs"),
    )
    script = tmp_path / f"fcn_pipeline_{framework}.py"
    script.write_text(source, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    outputs = Path(tmp_path / "outputs")
    table = pd.read_csv(outputs / "results_comparison.csv")
    assert set(table["run"]) == {"unweighted", "weighted"}
    assert (outputs / "leave_one_embryo_out.csv").exists()
    assert (outputs / "class_balance.png").exists()
    assert (outputs / "split.csv").exists()


BRANCHES = '''x = 0
if FRAMEWORK == 'pytorch':
    # comment kept
    x = 1
else:  # TensorFlow/Keras
    x = 2
y = x
'''


@pytest.mark.parametrize("framework, kept, dropped",
                         [("pytorch", "x = 1", "x = 2"),
                          ("keras", "x = 2", "x = 1")])
def test_framework_branches_are_resolved(framework, kept, dropped):
    code = _resolve_framework(BRANCHES, framework)
    assert kept in code and dropped not in code
    assert "if FRAMEWORK" not in code and "else" not in code
    compile(code, "<test>", "exec")


def test_multi_line_framework_test():
    source = ("if (FRAMEWORK\n        == 'pytorch'):\n    x = 1\n"
              "else:\n    x = 2\n")
    assert _resolve_framework(source, "pytorch").strip() == "x = 1"


def test_not_equal_and_missing_else():
    source = "if FRAMEWORK != 'keras':\n    x = 1\n"
    assert _resolve_framework(source, "pytorch").strip() == "x = 1"
    assert _resolve_framework(source, "keras").strip() == ""


def test_framework_parameter_is_fixed():
    source = "FRAMEWORK = 'pytorch'  # or 'keras'\n"
    assert _resolve_framework(source, "keras").strip() == "FRAMEWORK = 'keras'"


@pytest.mark.parametrize("source", [
    "if FRAMEWORK == 'pytorch': x = 1\n",
    "if FRAMEWORK == 'pytorch':\n    x = 1\nelif FRAMEWORK == 'keras':\n"
    "    x = 2\n",
    "if FRAMEWORK == 'pytorch':\n    x = 1\nelse: x = 2\n",
])
def test_unsupported_layouts_raise_clear_errors(source):
    with pytest.raises(ValueError, match="line 1"):
        _resolve_framework(source, "keras")


def test_backend_modules_are_filtered():
    assert _for_framework("zona_pellucida.backends.pytorch.model", "pytorch")
    assert not _for_framework("zona_pellucida.backends.keras.model",
                              "pytorch")
    assert _for_framework("zona_pellucida.backends", "keras")
    assert _for_framework("zona_pellucida.experiments", "keras")


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_export_records_this_projects_code_version(framework):
    from zona_pellucida.reproducibility import code_version

    source = build_pipeline(framework)
    namespace = {"__file__": str(OUTPUTS[framework])}
    # The imports at the top of the export, then its code_version().
    exec(source.split('logger = logging.getLogger("zona_pellucida")')[0],
         namespace)
    start = source.index("def package_version(")
    end = source.index("def environment_versions(")
    exec(source[start:end], namespace)
    assert namespace["code_version"]() == code_version()
