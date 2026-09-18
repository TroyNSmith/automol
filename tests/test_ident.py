"""Identity tests."""

import pytest

from automol import Geometry, HillFormula, Identity, RDKitInChI, RDKitSMILES
from automol.ident import AlgorithmRegistry
from automol.utils.exc import AlgorithmAlreadyRegisteredError, UnknownAlgorithmError


@pytest.fixture
def water_inchi() -> Identity:
    """Water identity fixture."""
    return Identity(algorithm=RDKitInChI, value="InChI=1S/H2O/h1H2")


@pytest.fixture
def water_smiles() -> Identity:
    """Water smiles fixture."""
    return Identity(algorithm=RDKitSMILES, value="O")


def test__inchi_roundtrip(water_inchi: Identity) -> None:
    """Test inchi to Geometry roundtrip."""
    water = water_inchi.geometry()
    water_inchi_rt = Identity.from_geometry(water, algorithm=RDKitInChI)

    assert water_inchi.value == water_inchi_rt.value


def test__smiles_roundtrip(water_smiles: Identity) -> None:
    """Test smiles to Geometry roundtrip."""
    water = water_smiles.geometry()
    water_smiles_rt = Identity.from_geometry(water, algorithm=RDKitSMILES)

    assert water_smiles.value == water_smiles_rt.value


def test__duplicate_registration_raises() -> None:
    """Test that re-registering an algorithm is rejected."""
    with pytest.raises(AlgorithmAlreadyRegisteredError):
        AlgorithmRegistry.register(RDKitInChI)


def test__unknown_algorithm_raises() -> None:
    """Test that looking up an unregistered algorithm is rejected."""
    with pytest.raises(UnknownAlgorithmError):
        AlgorithmRegistry.get("not-a-real-algorithm")


def test__hill_formula(water: Geometry) -> None:
    """Test Geometry to Hill-ordered formula."""
    ident = Identity.from_geometry(water, algorithm=HillFormula)
    assert ident.value == "H2O"


def test__hill_formula_with_carbon() -> None:
    """Test Hill formula with carbon present."""
    methane = Geometry(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        charge=0,
        spin=0,
    )
    ident = Identity.from_geometry(methane, algorithm=HillFormula)
    assert ident.value == "CH4"


def test__hill_formula_no_hydrogen() -> None:
    """Test Hill formula with no hydrogen present."""
    dichlorine = Geometry(
        symbols=["Cl", "Cl"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    ident = Identity.from_geometry(dichlorine, algorithm=HillFormula)
    assert ident.value == "Cl2"


def test__smiles_parent_algorithm() -> None:
    """Test that RDKitSMILES uses RDKitInChI as its parent algorithm."""
    canon_smiles = "CCCCC"
    weird_smiles = "C(C)CCC"
    smiles = [weird_smiles, "CCC", "CC(C)C"]

    geo = RDKitSMILES.geometry_fn(canon_smiles)
    other_geos = {s: RDKitSMILES.geometry_fn(s) for s in smiles}

    canon_ident = Identity.from_geometry(geo, algorithm=RDKitSMILES)
    weird_ident = Identity.from_geometry(
        geo, algorithm=RDKitSMILES, other_geos=other_geos
    )

    assert canon_ident.value == canon_smiles
    assert weird_ident.value == weird_smiles
