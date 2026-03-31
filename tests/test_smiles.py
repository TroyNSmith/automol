"""SMILES tests."""

import pytest

from automol import rd


@pytest.mark.parametrize(
    ("smi", "symbols", "charge", "spin"),
    [
        ("[CH3]", ["C", "H", "H", "H"], 0, 1),
        ("[CH3+]", ["C", "H", "H", "H"], 1, 0),
    ],
)
def test__geometry(smi: str, symbols: list[str], charge: int, spin: int) -> None:
    """Test geometry from SMILES."""
    mol = rd.mol.from_smiles(smi)
    geo = rd.mol.to_geometry(mol)
    assert geo.symbols == symbols
    assert geo.charge == charge
    assert geo.spin == spin
