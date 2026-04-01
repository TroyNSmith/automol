"""QCIO interface tests."""

import numpy as np
import pytest

from automol.geom import Geometry
from automol.qc import structure


@pytest.mark.parametrize(
    ("symbols", "coords", "charge", "spin"),
    [
        (["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], 0, 0),
        (["O", "H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.7, 0.5], [0.0, -0.7, 0.5]], 0, 0),
        (["C"], [[0.0, 0.0, 0.0]], 0, 2),  # Carbon atom (triplet)
    ],
)
def test_structure_conversion(
    symbols: list[str], coords: list[list[float]], charge: int, spin: int
) -> None:
    """Test that Geometry -> QCIO Structure -> Geometry preserves all data."""
    org_geo = Geometry(
        symbols=symbols, coordinates=np.array(coords), charge=charge, spin=spin
    )
    struc = structure.from_geometry(org_geo)
    fnl_geo = structure.geometry(struc)

    assert fnl_geo.symbols == org_geo.symbols
    assert fnl_geo.charge == org_geo.charge
    assert fnl_geo.spin == org_geo.spin
    # Approximately equal due to unit conversions
    assert fnl_geo.coordinates == pytest.approx(org_geo.coordinates, abs=1e-8)
