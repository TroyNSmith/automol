"""QCIO Structure interface."""

import pint
from qcio import Structure

from automol.geom import Geometry


def struc_from_geometry(geo: Geometry) -> Structure:
    """
    Generate QCIO Structure from Geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        QCIO Structure.
    """
    return Structure(
        symbols=geo.symbols,
        geometry=geo.coordinates * pint.Quantity("angstrom").m_as("bohr"),
        charge=geo.charge,
        multiplicity=geo.spin + 1,
    )


def struc_to_geometry(struc: Structure) -> Geometry:
    """
    Generate Geometry from QCIO Structure.

    Parameters
    ----------
    struc
        QCIO Structure.

    Returns
    -------
        Geometry.
    """
    return Geometry(
        symbols=struc.symbols,
        coordinates=struc.geometry * pint.Quantity("bohr").m_as("angstrom"),
        charge=struc.charge,
        spin=struc.multiplicity - 1,
    )
