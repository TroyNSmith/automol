"""Identity tests."""

import pytest

from automol import (
    AlgorithmRegistry,
    Geometry,
    IdentityKind,
    hill_formula,
    rdkit_inchi,
    rdkit_smiles,
)
from automol.utils.exc import AlgorithmAlreadyRegisteredError, UnknownAlgorithmError


@pytest.fixture
def water_inchi() -> str:
    """Water identity fixture."""
    return "InChI=1S/H2O/h1H2"


@pytest.fixture
def water_smiles() -> str:
    """Water smiles fixture."""
    return "O"


def test__inchi_roundtrip(water_inchi: str) -> None:
    """Test inchi to Geometry roundtrip."""
    water = rdkit_inchi.geometry_fn(water_inchi)
    water_inchi_rt = rdkit_inchi.identity_fn(water)

    assert water_inchi == water_inchi_rt


def test__smiles_roundtrip(water_smiles: str) -> None:
    """Test smiles to Geometry roundtrip."""
    water = rdkit_smiles.geometry_fn(water_smiles)
    water_smiles_rt = rdkit_smiles.identity_fn(water)

    assert water_smiles == water_smiles_rt


def test__duplicate_registration_raises() -> None:
    """Test that re-registering an algorithm is rejected."""
    with pytest.raises(AlgorithmAlreadyRegisteredError):
        AlgorithmRegistry.register(
            name="rdkit inchi",
            kind=IdentityKind.STEREOISOMER,
            identity_fn=rdkit_inchi.identity_fn,
        )


def test__unknown_algorithm_raises() -> None:
    """Test that looking up an unregistered algorithm is rejected."""
    with pytest.raises(UnknownAlgorithmError):
        AlgorithmRegistry.get("not-a-real-algorithm")


def test__hill_formula(water: Geometry) -> None:
    """Test Geometry to Hill-ordered formula."""
    ident = hill_formula.identity_fn(water)
    assert ident == "H2O"


def test__hill_formula_with_carbon() -> None:
    """Test Hill formula with carbon present."""
    methane = Geometry(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        charge=0,
        spin=0,
    )
    ident = hill_formula.identity_fn(methane)
    assert ident == "CH4"


def test__hill_formula_no_hydrogen() -> None:
    """Test Hill formula with no hydrogen present."""
    dichlorine = Geometry(
        symbols=["Cl", "Cl"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    ident = hill_formula.identity_fn(dichlorine)
    assert ident == "Cl2"


def test__smiles_parent_algorithm() -> None:
    """Test that RDKitSMILES uses RDKitInChI as its parent algorithm."""
    canon_smiles = "CCCCC"
    weird_smiles = "C(C)CCC"
    smiles = [weird_smiles, "CCC", "CC(C)C"]

    geo = rdkit_smiles.geometry_fn(canon_smiles)
    other_geos = {s: rdkit_smiles.geometry_fn(s) for s in smiles}

    canon_ident = rdkit_smiles.identity_fn(geo)
    weird_ident = rdkit_smiles.identity_fn(geo, other_geos=other_geos)

    assert canon_ident == canon_smiles
    assert weird_ident == weird_smiles
