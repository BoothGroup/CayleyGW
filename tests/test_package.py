"""The public names, the version, the Sector contract and docstring coverage."""

from __future__ import annotations

import inspect

import cayleygw
from cayleygw import Sector

PUBLIC_NAMES = {
    "CayleyGW",
    "build_cayley_moments",
    "build_upfolded_hamiltonian",
    "diagonalize_upfolded",
    "extract_ip_ea",
    "reconstruction_test",
    "calculate_spectrum",
    "plot_spectrum",
    "enable_logging",
    "ExactG0W0SelfEnergy",
    "UpfoldedDysonHamiltonian",
    "Screening",
    "Sector",
    "Tolerances",
    "CayleyGWError",
    "ValidationError",
    "RefusalError",
    "__version__",
}


def test_version_and_public_exports() -> None:
    assert cayleygw.__version__ == "1.0.0"
    assert len(cayleygw.__all__) == len(set(cayleygw.__all__))
    assert set(cayleygw.__all__) == PUBLIC_NAMES
    for name in cayleygw.__all__:
        assert hasattr(cayleygw, name)


def test_sector_contract() -> None:
    assert Sector.HOLE.value == "<"
    assert Sector.PARTICLE.value == ">"
    assert Sector.HOLE.arc_sign == -1
    assert Sector.PARTICLE.arc_sign == 1


def test_public_objects_have_substantive_docstrings() -> None:
    undocumented: list[str] = []
    for name in cayleygw.__all__:
        value = getattr(cayleygw, name)
        if inspect.isclass(value) or inspect.isfunction(value):
            docstring = inspect.getdoc(value)
            if docstring is None or len(docstring.split()) < 8:
                undocumented.append(name)
    assert undocumented == []


def test_public_methods_and_properties_have_substantive_docstrings() -> None:
    undocumented: list[str] = []
    for exported_name in cayleygw.__all__:
        exported = getattr(cayleygw, exported_name)
        if not inspect.isclass(exported):
            continue
        for member_name, member in inspect.getmembers(exported):
            if member_name.startswith("_"):
                continue
            if isinstance(member, property):
                candidate = member.fget
            elif inspect.isfunction(member) or inspect.ismethod(member):
                candidate = member
            else:
                continue
            docstring = inspect.getdoc(candidate)
            if docstring is None or len(docstring.split()) < 8:
                undocumented.append(f"{exported_name}.{member_name}")
    assert undocumented == []
