"""Molecular identities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict
from rdkit import Chem

from . import geom
from .utils.exc import AlgorithmAlreadyRegisteredError, UnknownAlgorithmError

if TYPE_CHECKING:
    from .geom import Geometry

# `Mapping` (rather than `dict`) is covariant in its value type, so a mapping
# of `Geometry` subclasses (e.g. `dict[str, SubGeometry]`) is accepted too.
OTHER_GEOS = Mapping[str, "Geometry"] | None


@runtime_checkable
class IdentityProtocol(Protocol):
    """Protocol for identity functions."""

    def __call__(self, geo: Geometry, other_geos: OTHER_GEOS = None) -> str:
        """Identity function not implemented."""
        msg = "Identity function not implemented."
        raise NotImplementedError(msg)


@runtime_checkable
class GeometryProtocol(Protocol):
    """Protocol for geometry functions."""

    def __call__(self, value: str) -> Geometry:
        """Geometry function not implemented."""
        msg = "Geometry function not implemented."
        raise NotImplementedError(msg)


class IdentityKind(StrEnum):
    """Category of molecular identity."""

    FORMULA = "formula"
    STEREOISOMER = "stereoisomer"
    CONFORMER = "conformer"
    ISOMER = "isomer"


def default_geometry_fn(value: str) -> Geometry:
    """Default geometry function that raises an error."""
    msg = "Geometry function not implemented."
    raise NotImplementedError(msg)


class Algorithm(BaseModel):
    """Boilerplate for Algorithm instances."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    name: str
    kind: IdentityKind
    parent_algorithm: Algorithm | None = None
    deterministic: bool = True  # Indicates if this algorithm is deterministic

    identity_fn: IdentityProtocol
    geometry_fn: GeometryProtocol = default_geometry_fn


class AlgorithmRegistry:
    """Central registry of all known identity algorithms."""

    algorithms: ClassVar[list[Algorithm]] = []

    @classmethod
    def register(  # noqa: PLR0913
        cls,
        name: str,
        kind: IdentityKind,
        identity_fn: IdentityProtocol,
        geometry_fn: GeometryProtocol = default_geometry_fn,
        parent_algorithm: Algorithm | None = None,
        *,
        deterministic: bool = True,
    ) -> Algorithm:
        """Register an algorithm instance."""
        if any(name == a.name for a in cls.algorithms):
            msg = f"Algorithm {name!r} is already registered."
            raise AlgorithmAlreadyRegisteredError(msg)
        algorithm = Algorithm.model_validate(
            {
                "name": name,
                "kind": kind,
                "parent_algorithm": parent_algorithm,
                "deterministic": deterministic,
                "identity_fn": identity_fn,
                "geometry_fn": geometry_fn,
            }
        )
        cls.algorithms.append(algorithm)
        return algorithm

    @classmethod
    def get(cls, name: str) -> Algorithm:
        """Get an algorithm from registry."""
        try:
            return next(a for a in cls.algorithms if a.name == name)
        except StopIteration:
            available = ", ".join(sorted(a.name for a in cls.algorithms))
            msg = f"Unknown algorithm {name!r}. Available: {available}"
            raise UnknownAlgorithmError(msg) from None

    @classmethod
    def all_algorithms(cls) -> list[str]:
        """Return all registered algorithms."""
        return sorted(a.name for a in cls.algorithms)

    @classmethod
    def algorithms_for_kind(cls, kind: str) -> list[str]:
        """Return all registered algorithms for a kind."""
        return sorted(a.name for a in cls.algorithms if a.kind == kind)


def rdkit_inchi_geometry_fn(value: str) -> Geometry:
    """Generate Geometry from InChI with RDKit."""
    mol = Chem.MolFromInchi(value, sanitize=True, removeHs=False)
    mol = Chem.AddHs(mol)
    return geom.from_rdkit_mol(mol)


def rdkit_inchi_identity_fn(
    geo: Geometry,
    other_geos: OTHER_GEOS = None,  # noqa: ARG001
) -> str:
    """Generate InChI from Geometry with RDKit."""
    mol = geom.rdkit_mol(geo)
    mol_block = Chem.rdmolfiles.MolToMolBlock(mol)
    return Chem.inchi.MolBlockToInchi(mol_block)


rdkit_inchi = AlgorithmRegistry.register(
    name="rdkit inchi",
    kind=IdentityKind.STEREOISOMER,
    identity_fn=rdkit_inchi_identity_fn,
    geometry_fn=rdkit_inchi_geometry_fn,
)


def rdkit_smiles_geometry_fn(value: str) -> Geometry:
    """Generate Geometry from SMILES with RDKit."""
    mol = Chem.MolFromSmiles(value)
    mol = Chem.AddHs(mol)
    return geom.from_rdkit_mol(mol)


def rdkit_smiles_identity_fn(geo: Geometry, other_geos: OTHER_GEOS = None) -> str:
    """Generate SMILES from Geometry with RDKit."""
    inchi = rdkit_inchi.identity_fn(geo, other_geos)
    if other_geos:
        smiles = next(
            s
            for s, g in other_geos.items()
            if rdkit_inchi.identity_fn(g, None) == inchi
        )
        if smiles:
            return smiles
    return Chem.MolToSmiles(Chem.RemoveAllHs(geom.rdkit_mol(geo)))


rdkit_smiles = AlgorithmRegistry.register(
    name="rdkit smiles",
    kind=IdentityKind.STEREOISOMER,
    deterministic=False,
    parent_algorithm=rdkit_inchi,
    identity_fn=rdkit_smiles_identity_fn,
    geometry_fn=rdkit_smiles_geometry_fn,
)


def hill_formula_identity_fn(
    geo: Geometry,
    other_geos: OTHER_GEOS = None,  # noqa: ARG001
) -> str:
    """Render the molecular formula in Hill order."""
    counts = Counter(s.capitalize() for s in geo.symbols)

    ordered = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    ordered.extend(sorted(counts.items(), key=lambda x: x[0]))

    return "".join(s if n == 1 else f"{s}{n}" for s, n in ordered)


hill_formula = AlgorithmRegistry.register(
    name="hill formula",
    kind=IdentityKind.FORMULA,
    deterministic=False,
    parent_algorithm=rdkit_inchi,
    identity_fn=hill_formula_identity_fn,
)
