"""Molecular geometries."""

import hashlib

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds
import scipy

from . import element, rd
from .types import CoordinatesField, FloatArray


class Geometry(BaseModel):
    """
    Molecular geometry.

    Parameters
    ----------
    symbols
        Atomic symbols in order (e.g., ``["H", "O", "H"]``).
        The length of ``symbols`` must match the number of atoms.
    coordinates
        Cartesian coordinates of the atoms in Angstroms.
        Shape is ``(len(symbols), 3)`` and the ordering corresponds to ``symbols``.
    charge
        Total molecular charge.
    spin
        Number of unpaired electrons, i.e. two times the spin quantum number (``2S``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int = 0
    spin: int = 0

    hash: str | None = Field(default=None)

    @property
    def masses(self) -> list[float]:
        """Get isotopic masses."""
        return list(map(element.mass, self.symbols))

    @property
    def atomic_numbers(self) -> list[float]:
        """Get atomic numbers."""
        return list(map(element.number, self.symbols))

    @model_validator(mode="after")
    def populate_hash(self) -> "Geometry":
        """Populate hash immediately after the model is created."""
        # Only populate if hash wasn't explicitly provided
        if self.hash is None:
            self.hash = geometry_hash(self)
        return self


# Importers / Exporters
def xyz_block(geom: Geometry) -> str:
    """
    Return geometry as formatted xyz block.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    lines = [f"{len(geom.symbols)}", ""]
    for sym, (x, y, z) in zip(geom.symbols, geom.coordinates, strict=True):
        lines.append(f"{sym:<2} {x:12.8f} {y:12.8f} {z:12.8f}")

    return "\n".join(lines)


def rdkit_mol(geom: Geometry) -> Mol:
    """
    Instantiate an rdkit Mol from a Geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    Mol
        rdkit Mol instance.
    """
    raw_mol = Chem.MolFromXYZBlock(xyz_block(geom))
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    return conn_mol


def from_rdkit_mol(mol: Mol) -> Geometry:
    """
    Generate geometry from RDKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
        Geometry.
    """
    if not rd.mol.has_coordinates(mol):
        mol = rd.mol.add_coordinates(mol)

    return Geometry(
        symbols=rd.mol.symbols(mol),
        coordinates=rd.mol.coordinates(mol),
        charge=rd.mol.charge(mol),
        spin=rd.mol.spin(mol),
    )


def from_smiles(smi: str) -> Geometry:
    """
    Instantiate Geometry from SMILES string.

    Parameters
    ----------
    smi
        SMILES formatted string.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    mol = rd.mol.from_smiles(smi)
    return from_rdkit_mol(mol)


def inchi(geom: Geometry) -> str:
    """
    Provide InChI string from Geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    mol = rdkit_mol(geom)
    return rd.mol.inchi(mol)


# Properties
def geometry_hash(geom: Geometry, decimals: int = 6) -> str:
    """
    Generate geometry hash string.

    Parameters
    ----------
    decimals
        Number of decimal places to round the coordinates before hashing.

    Returns
    -------
        Geometry hash string.
    """
    # 1. Convert symbols and coordinates to integers
    numbers = geom.atomic_numbers
    icoords = np.rint(geom.coordinates * 10**decimals)
    # 2. Generate bytes representation of each field
    numbers_bytes = np.asarray(numbers, dtype=np.dtype("<i8")).tobytes("C")
    icoords_bytes = icoords.astype(np.dtype("<i8")).tobytes("C")
    charge_bytes = geom.charge.to_bytes(1, byteorder="little", signed=True)
    spin_bytes = geom.spin.to_bytes(1, byteorder="little", signed=True)
    # 3. Combine all bytes and generate hash
    geo_bytes = b"|".join([numbers_bytes, icoords_bytes, charge_bytes, spin_bytes])
    return hashlib.sha256(geo_bytes).hexdigest()


def center_of_mass(geom: Geometry) -> FloatArray:
    """
    Calculate geometry center of mass.

    Parameters
    ----------
        Geometry.

    Returns
    -------
        Center of mass coordinates.
    """
    masses = list(map(element.mass, geom.symbols))
    coords = geom.coordinates
    return np.sum(np.reshape(masses, (-1, 1)) * coords, axis=0) / np.sum(masses)


def inertia_moments(geo: Geometry) -> FloatArray:
    """Compute inertia moments of a Geometry."""
    coords = geo.coordinates - center_of_mass(geo)

    # Compute inertia tensor
    norms_sq = np.einsum("ni,ni->n", coords, coords)
    total = np.sum(geo.masses * norms_sq)
    i_matrix = total * np.eye(3) - np.einsum("n,ni,nj->ij", geo.masses, coords, coords)

    # Principal moments via symmetric eigendecomposition
    moments, _ = np.linalg.eigh(i_matrix)

    return np.sort(moments)


def kabsch(
    geo_1: Geometry, geo_2: Geometry, *, heavy_only: bool = False
) -> tuple[FloatArray, FloatArray, float]:
    """
    Compute the optimal rotation / translation to align two Geometries and their RMSD.

    For more information on the numerical method, see https://hunterheidenreich.com/posts/kabsch-algorithm/

    Parameters
    ----------
    geo_1
        Geometry object.
    geo_2
        Geometry object.
    heavy_only
        If True, only consider heavy atoms.

    Returns
    -------
    FloatArray
        Optimal rotation
    FloatArray
        Optimal translation
    float
        RMSD
    """
    p = np.array(geo_1.coordinates)
    q = np.array(geo_2.coordinates)
    p_masses = geo_1.masses
    q_masses = geo_2.masses

    if heavy_only:
        mask_p = np.array([s != "H" for s in geo_1.symbols])
        mask_q = np.array([s != "H" for s in geo_2.symbols])
        # Contrapositive of "If no heavy atoms exist (e.g., H2, H), skip masking"
        if np.any(mask_p):
            p, q = p[mask_p], q[mask_q]

            p_masses = np.asanyarray(p_masses)
            q_masses = np.asanyarray(q_masses)

            p_masses, q_masses = p_masses[mask_p], q_masses[mask_q]

    if p.shape != q.shape:
        msg = f"""
        Input arrays must have same number of dimensions.\n
        {p.shape = }\n
        {q.shape = }\n
        """
        raise ValueError(msg)

    # --- Optimal translation -------------------
    centroid_p = center_of_mass(geo_1)
    centroid_q = center_of_mass(geo_2)
    t = centroid_p - centroid_q  # Optimal translation
    # Center the coordinates
    p = p - centroid_p
    q = q - centroid_q

    # --- Optimal rotation ----------------------
    H = np.dot(p.T, q)  # Covariance matrix  # noqa: N806
    U, _, Vt = np.linalg.svd(H)  # noqa: N806

    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:  # Validate right-handed coordinates
        Vt[-1, :] *= -1.0

    R = np.dot(Vt.T, U.T)  # Optimal rotation  # noqa: N806

    # --- RMSD ----------------------------------
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / p.shape[0])

    return R, t, rmsd


def is_similar(
    geom_1: Geometry,
    geom_2: Geometry,
    *,
    moi_tol: float = 1e-3,
    rmsd_tol: float = 1e-1,
) -> bool:
    """Check if geometries are similar."""
    # --- Symbols  ---
    if geom_1.symbols.sort() != geom_2.symbols.sort():
        return False

    # --- Geometry Hash ---
    if geometry_hash(geom_1) == geometry_hash(geom_2):
        return True

    # --- Moments of Inertia ---
    moments_1 = np.sort(inertia_moments(geom_1))
    moments_2 = np.sort(inertia_moments(geom_2))

    eps = 1e-6  # Avoid division by zero in linear molecules
    moi_diff = np.abs(moments_1 - moments_2) / (moments_2 + eps)

    if np.any(moi_diff > moi_tol):
        return False

    # --- Heavy Atom RMSD ---
    _, _, rmsd = kabsch(geom_1, geom_2, heavy_only=True)

    return rmsd < rmsd_tol


def distance_matrix(geom: Geometry) -> np.ndarray:
    """Compute the distance matrix for a geometry."""
    return scipy.spatial.distance_matrix(geom.coordinates, geom.coordinates)


def set_dist(
    geom: Geometry,
    *,
    atom_indices: tuple[int, int],
    dist: float,
    max_dr: float = 0.25,
    in_place: bool = False,
) -> Geometry:
    """Set distance between two atoms."""
    geom = geom if in_place else geom.model_copy(deep=True)
    i, j = atom_indices

    # Compute current distance and unit vector
    vec = geom.coordinates[j] - geom.coordinates[i]
    r = np.linalg.norm(vec)
    unit_vec = vec / r

    # Ensure that change does not exceed max allowable
    dr = abs(r - dist)
    if dr > max_dr:
        msg = f"{dr = } exceeds {max_dr = }."
        raise ValueError(msg)

    # Atom j coordinates relevant to atom i
    geom.coordinates[j] = geom.coordinates[i] + (unit_vec * dist)

    return geom
