"""The package layout, held as tests.

A main module holds one class and its exported entry functions, and defines
nothing private; its helpers live under ``_helpers``. Helpers and records sit
below the main modules and never import one, and the package has no import
cycle. ``test_package`` pins the public names.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cayleygw

PACKAGE = Path(cayleygw.__file__).resolve().parent

# Each main module, its one class and its exported entry functions.
MAIN_MODULES = {
    "screening.py": ("ProjectedRPAResolvent", set()),
    "moments.py": ("G0W0CayleyMoments", {"build_cayley_moments"}),
    "contour.py": ("EllipseContour", set()),
    "upfold.py": (
        "UpfoldedDysonHamiltonian",
        {"build_upfolded_hamiltonian", "diagonalize_upfolded", "reconstruction_test"},
    ),
    "kernel.py": ("CayleyGW", set()),
    "realization/sector.py": ("SectorSelfEnergyRealization", set()),
    "realization/base.py": ("UnitaryMomentRealization", set()),
    "realization/block_cmv.py": ("BlockCMVRealization", set()),
    "realization/toeplitz.py": ("ToeplitzRealization", set()),
}
LOWER_LAYERS = ("_helpers", "realization/_helpers", "tools/records")


def module_name(path: Path) -> str:
    """Return the dotted module name of ``path``."""

    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = {module_name(path): path for path in PACKAGE.rglob("*.py")}
MAIN_NAMES = {module_name(PACKAGE / relative) for relative in MAIN_MODULES}


def imported_modules(path: Path) -> set[str]:
    """Return the package modules ``path`` imports anywhere but under ``TYPE_CHECKING``."""

    name = module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and ast.unparse(node.test).endswith("TYPE_CHECKING"):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: package.count(".") + 2 - node.level]
                base = ".".join(anchor + ([node.module] if node.module else []))
            for alias in node.names:
                submodule = f"{base}.{alias.name}"
                found.add(submodule if submodule in MODULES else base)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return {module for module in found if module in MODULES}


def private(name: str) -> bool:
    """Return whether ``name`` is private: underscored and not a dunder."""

    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def test_each_main_module_is_one_class_and_its_entry_functions() -> None:
    found = {}
    for relative in MAIN_MODULES:
        tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        found[relative] = (classes, functions)
    assert found == {
        relative: ([cls], entries) for relative, (cls, entries) in MAIN_MODULES.items()
    }


def test_main_modules_define_nothing_private() -> None:
    defined = []
    for relative in MAIN_MODULES:
        tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            defined += [
                f"{relative}: {target.id}"
                for target in targets
                if isinstance(target, ast.Name) and private(target.id)
            ]
        defined += [
            f"{relative}:{node.lineno}: {node.name}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and private(node.name)
        ]
    assert defined == []


def test_helpers_and_records_never_import_a_main_module() -> None:
    lower = [path for layer in LOWER_LAYERS for path in sorted((PACKAGE / layer).rglob("*.py"))]
    assert lower
    upward = [
        f"{module_name(path)} -> {target}"
        for path in lower
        for target in sorted(imported_modules(path) & MAIN_NAMES)
    ]
    assert upward == []


def test_the_package_has_no_import_cycle() -> None:
    graph = {name: imported_modules(path) - {name} for name, path in MODULES.items()}
    cycles: list[list[str]] = []
    state: dict[str, str] = {}

    def visit(name: str, trail: list[str]) -> None:
        state[name] = "open"
        for target in sorted(graph[name]):
            if state.get(target) == "open":
                cycles.append(trail[trail.index(target) :] + [target])
            elif target not in state:
                visit(target, trail + [target])
        state[name] = "done"

    for name in sorted(graph):
        if name not in state:
            visit(name, [name])
    assert cycles == []
