"""Write fcn_pipeline_<framework>.py: notebook and package in one file.

    python -m zona_pellucida.bin.export_pipeline          # (re)write both
    python -m zona_pellucida.bin.export_pipeline --check  # fail if stale

One file per framework (pytorch, keras). Each package module is inserted
before the first notebook cell that lists it in its `export_modules`
metadata, so the file follows the order of the notebook (the Keras file
is the thesis appendix); modules of the other framework's backend are left
out, and so
is the other branch of every `if FRAMEWORK == ...:` block. Colab-only cells
(tag `colab`) and IPython magics are left out, and all imports are
collected at the top.
"""

import argparse
import ast
import json
import logging
import re
import sys
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "zona_pellucida"
NOTEBOOK = REPO_ROOT / "fcn.ipynb"
FRAMEWORK_NAMES = {"pytorch": "PyTorch Lightning", "keras": "TensorFlow/Keras"}
OUTPUTS = {name: REPO_ROOT / f"fcn_pipeline_{name}.py"
           for name in FRAMEWORK_NAMES}
LINE_LENGTH = 80
RULE = "# " + "=" * (LINE_LENGTH - 2)
ELSE_LINE = re.compile(r"^\s*else\s*:\s*(#.*)?$")

HEADER = '''"""Zona pellucida segmentation with a U-Net-style FCN ({name}).

Single-file export of fcn.ipynb (FRAMEWORK = '{framework}') and the
zona_pellucida package, in the order of the notebook. Generated with
`python -m zona_pellucida.bin.export_pipeline`; do not edit by hand.
"""
'''


def _split_source(source):
    """Return (top-level import nodes, docstring, remaining source)."""
    tree = ast.parse(source)
    drop = set()
    imports = []
    docstring = ast.get_docstring(tree)
    body = tree.body
    if docstring is not None:
        drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
            drop.update(range(node.lineno, node.end_lineno + 1))
    lines = source.splitlines()
    for node in imports:
        # A comment that only heads imports (e.g. "# General libraries")
        # would be left empty: drop it unless code follows the imports.
        end = node.end_lineno
        while end + 1 in drop:
            end += 1
        if end < len(lines) and lines[end].strip():
            continue
        number = node.lineno - 1
        while number >= 1 and lines[number - 1].lstrip().startswith("#") \
                and number not in drop:
            drop.add(number)
            number -= 1
    kept = [
        line for number, line in enumerate(lines, start=1)
        if number not in drop
        and line.strip() != "logger = logging.getLogger(__name__)"
    ]
    return imports, docstring, "\n".join(kept).strip("\n")


def _collapse_blank_lines(code, keep):
    """At most `keep` consecutive blank lines (removed imports leave gaps)."""
    return re.sub(r"\n(?:[ \t]*\n){%d,}" % (keep + 1), "\n" * (keep + 1),
                  code)


class _Imports:
    """Collects and renders de-duplicated import statements."""

    def __init__(self):
        self.plain = set()  # (module, alias)
        self.from_ = {}  # module -> {(name, alias)}

    def add(self, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith(PACKAGE):
                    self.plain.add((alias.name, alias.asname))
        elif node.module and not node.module.startswith(PACKAGE):
            names = self.from_.setdefault(node.module, set())
            names.update((a.name, a.asname) for a in node.names)

    @staticmethod
    def _is_stdlib(module):
        return module.split(".")[0] in sys.stdlib_module_names

    @staticmethod
    def _alias(name, asname):
        return f"{name} as {asname}" if asname else name

    def _from_line(self, module):
        names = sorted(self._alias(*n) for n in self.from_[module])
        line = f"from {module} import {', '.join(names)}"
        if len(line) <= LINE_LENGTH:
            return line
        inner = "".join(f"    {name},\n" for name in names)
        return f"from {module} import (\n{inner})"

    def render(self):
        groups = []
        for stdlib in (True, False):
            lines = sorted(
                (f"import {self._alias(m, a)}" for m, a in self.plain
                 if self._is_stdlib(m) == stdlib), key=str.lower,
            )
            lines += [
                self._from_line(m) for m in sorted(self.from_, key=str.lower)
                if self._is_stdlib(m) == stdlib
            ]
            if lines:
                groups.append("\n".join(lines))
        return "\n\n".join(groups)


def _module_block(module, imports):
    path = REPO_ROOT / (module.replace(".", "/") + ".py")
    if not path.exists():  # a package
        path = REPO_ROOT / module.replace(".", "/") / "__init__.py"
    nodes, docstring, code = _split_source(path.read_text(encoding="utf-8"))
    for node in nodes:
        imports.add(node)
    title = f"# {path.relative_to(REPO_ROOT).as_posix()}"
    if docstring:
        summary = docstring.splitlines()[0]
        joined = f"{title}: {summary}"
        title = joined if len(joined) <= LINE_LENGTH else (
            f"{title}\n# {summary}")
    # Linter markers mean nothing to a reader of the appendix.
    code = re.sub(r"[ \t]+# noqa: [A-Z]+\d+", "", code)
    code = _collapse_blank_lines(code, keep=2)
    return f"{RULE}\n{title}\n{RULE}\n\n\n{code}"


def _cell_code(source):
    """Notebook cell without IPython magics / shell commands."""
    lines = [
        line for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    ]
    return "\n".join(lines)


def _framework_test(test):
    """(value, negated) for `FRAMEWORK == 'value'` / `!=`, else None."""
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "FRAMEWORK" and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(test.comparators[0], ast.Constant)):
        return test.comparators[0].value, isinstance(test.ops[0], ast.NotEq)
    return None


def _resolve_framework(source, framework):
    """Keep only the applicable branch of top-level `if FRAMEWORK == ...:`
    blocks and fix the value of the `FRAMEWORK = ...` parameter.

    Supported layout: `if FRAMEWORK == '...':` and an optional `else:`
    (comments allowed), each followed by an indented block.
    """
    lines = source.splitlines()
    for node in reversed(ast.parse(source).body):
        if not isinstance(node, ast.If):
            continue
        test = _framework_test(node.test)
        if test is None:
            continue
        value, negated = test
        where = f"line {node.lineno}"
        if node.body[0].lineno == node.test.end_lineno:
            raise ValueError(f"{where}: put the `if FRAMEWORK` block on "
                             "its own lines")
        if node.orelse and lines[node.orelse[0].lineno - 1].lstrip() \
                .startswith("elif"):
            raise ValueError(f"{where}: `elif` on FRAMEWORK is not supported")
        if (value == framework) != negated:
            # The block starts after the (possibly multi-line) test.
            start, end = node.test.end_lineno + 1, node.body[-1].end_lineno
        elif node.orelse:
            else_line = next(
                (number for number in range(node.body[-1].end_lineno + 1,
                                            node.orelse[0].lineno)
                 if ELSE_LINE.match(lines[number - 1])), None)
            if else_line is None:
                raise ValueError(f"{where}: put `else:` on its own line")
            start, end = else_line + 1, node.orelse[-1].end_lineno
        else:
            start, end = 1, 0  # nothing applies
        branch = textwrap.dedent("\n".join(lines[start - 1:end]))
        try:
            compile(branch, "<FRAMEWORK branch>", "exec")
        except SyntaxError as err:
            raise ValueError(f"{where}: the {framework} block cannot be "
                             f"de-indented ({err.msg})") from err
        lines[node.lineno - 1:node.end_lineno] = branch.splitlines()
    source = "\n".join(lines)
    return re.sub(r"^FRAMEWORK = .*$", f"FRAMEWORK = {framework!r}", source,
                  flags=re.MULTILINE)


def _for_framework(module, framework):
    """Shared modules and the chosen framework's backend modules."""
    parts = module.split(".")
    return (parts[:2] != [PACKAGE, "backends"] or len(parts) < 3
            or parts[2] == framework)


def _defined_names(code):
    tree = ast.parse(code)
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _assigned_names(code):
    return {
        node.id for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def build_pipeline(framework, notebook_path=NOTEBOOK):
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    imports = _Imports()
    blocks = []
    exported = []
    package_defs = set()
    notebook_names = set()
    for number, cell in enumerate(notebook["cells"], start=1):
        source = "".join(cell["source"])
        metadata = cell.get("metadata", {})
        if cell["cell_type"] == "markdown":
            headings = [
                line.lstrip("#").strip() for line in source.splitlines()
                if line.startswith("#")
            ]
            blocks.extend(f"# {RULE[2:6]} {h}" for h in headings)
            continue
        if cell["cell_type"] != "code" or "colab" in metadata.get("tags", []):
            continue
        for module in metadata.get("export_modules", []):
            if module in exported or not _for_framework(module, framework):
                continue
            block = _module_block(module, imports)
            package_defs |= _defined_names(block)
            blocks.append(block)
            exported.append(module)
        nodes, _, code = _split_source(
            _resolve_framework(_cell_code(source), framework))
        code = _collapse_blank_lines(code, keep=1)
        for node in nodes:
            imports.add(node)
        notebook_names |= _assigned_names(code)
        blocks.append(f"# %% Notebook cell {number}\n{code}".rstrip())

    shadowed = package_defs & notebook_names
    if shadowed:
        raise ValueError(f"Notebook variables shadow functions: {shadowed}")
    header = HEADER.format(name=FRAMEWORK_NAMES[framework],
                           framework=framework)
    text = "\n\n".join([
        header.rstrip(),
        imports.render(),
        'logger = logging.getLogger("zona_pellucida")',
        *blocks,
    ]) + "\n"
    compile(text, str(OUTPUTS[framework]), "exec")
    return text


def main(check=False):
    stale = 0
    for framework, output in OUTPUTS.items():
        text = build_pipeline(framework)
        if check:
            current = (output.read_text(encoding="utf-8")
                       if output.exists() else "")
            if current != text:
                logger.error("%s is stale: re-run the export", output.name)
                stale = 1
            continue
        output.write_text(text, encoding="utf-8")
        logger.info("Wrote %s (%d lines)", output, text.count("\n"))
    return stale


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if an exported file is not up to date")
    sys.exit(main(parser.parse_args().check))
