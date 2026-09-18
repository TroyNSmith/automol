"""Molecular identities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import BaseModel
from rdkit import Chem

from . import geom
from .utils.exc import AlgorithmAlreadyRegisteredError, UnknownAlgorithmError

if TYPE_CHECKING:
    from .geom import Geometry


class IdentityKind(StrEnum):
    """Category of molecular identity."""

    FORMULA = "formula"
    STEREOISOMER = "stereoisomer"
    CONFORMER = "conformer"
    ISOMER = "isomer"


# Identifiers for the built-in algorithms. Higher-level packages are free to
# register additional algorithms under their own string identifiers; these
# are just the ones shipped with this package.
RDKIT_INCHI = "rdkit inchi"
RDKIT_SMILES = "rdkit smiles"
HILL_FORMULA = "hill formula"


class Algorithm(ABC):
    """Boilerplate for Algorithm functions class."""

    name: ClassVar[str]
    kind: ClassVar[IdentityKind]
    parent_algorithm: ClassVar[type[Algorithm] | None] = None

    @classmethod
    @abstractmethod
    def identity_fn(
        cls, geo: Geometry, other_geos: dict[str, Geometry] | None = None
    ) -> str:
        """Generate an identifier string from a Geometry."""

    @classmethod
    def geometry_fn(cls, value: str) -> Geometry:
        """Instantiate a Geometry from an identifier string."""
        msg = f"Conversion of {value} to Geometry not implemented."
        raise NotImplementedError(msg)


class AlgorithmRegistry:
    """Central registry of all known identity algorithms."""

    _algorithms: ClassVar[list[type[Algorithm]]] = []

    @classmethod
    def register[T: type[Algorithm]](cls, cls_: T) -> T:
        """Register identity_fn and geometry_fn as an AlgorithmDef."""
        if any(a.name == cls_.name for a in cls._algorithms):
            msg = f"Algorithm {cls_.name!r} is already registered."
            raise AlgorithmAlreadyRegisteredError(msg)
        cls._algorithms.append(cls_)
        return cls_

    @classmethod
    def get(cls, name: str) -> type[Algorithm]:
        """Get an algorithm from registry."""
        try:
            return next(a for a in cls._algorithms if a.name == name)
        except StopIteration:
            available = ", ".join(sorted(a.name for a in cls._algorithms))
            msg = f"Unknown algorithm {name!r}. Available: {available}"
            raise UnknownAlgorithmError(msg) from None

    @classmethod
    def all_algorithms(cls) -> list[str]:
        """Return all registered algorithms."""
        return sorted(a.name for a in cls._algorithms)

    @classmethod
    def algorithms_for_kind(cls, kind: str) -> list[str]:
        """Return all registered algorithms for a kind."""
        return sorted(a.name for a in cls._algorithms if a.kind == kind)


class Identity(BaseModel):
    """
    Molecular identity record.

    Parameters
    ----------
    algorithm
        Registered algorithm that produced this identity.
    value
        Resulting string identifier.
    kind
        Category of identity (e.g., "stereoisomer", "conformer"). Must match
        the registered algorithm's kind; prefer `from_geometry` or
        `from_value` over setting this directly.
    """

    algorithm: type[Algorithm]
    value: str

    @classmethod
    def from_geometry(
        cls,
        geo: Geometry,
        *,
        algorithm: type[Algorithm],
        other_geos: dict[str, Geometry] | None = None,
    ) -> Self:
        """Return an Identity from a Geometry, by algorithm alone."""
        value = algorithm.identity_fn(geo, other_geos)
        return cls(algorithm=algorithm, value=value)

    def geometry(self) -> Geometry:
        """Return a Geometry from Identity instance."""
        return self.algorithm.geometry_fn(self.value)


@AlgorithmRegistry.register
class RDKitInChI(Algorithm):
    """Identify geometry with InChI using RDKit."""

    name: ClassVar[str] = "RDKitInChI"
    kind: ClassVar[IdentityKind] = IdentityKind.STEREOISOMER

    @classmethod
    def identity_fn(
        cls,
        geo: Geometry,
        other_geos: dict[str, Geometry] | None = None,  # noqa: ARG003
    ) -> str:
        """Generate InChI from Geometry with RDKit."""
        mol = geom.rdkit_mol(geo)
        mol_block = Chem.rdmolfiles.MolToMolBlock(mol)
        return Chem.inchi.MolBlockToInchi(mol_block)

    @classmethod
    def geometry_fn(cls, value: str) -> Geometry:
        """Generate Geometry from InChI with RDKit."""
        mol = Chem.MolFromInchi(value, sanitize=True, removeHs=False)
        mol = Chem.AddHs(mol)
        return geom.from_rdkit_mol(mol)


@AlgorithmRegistry.register
class RDKitSMILES(Algorithm):
    """Identify geometry with SMILES using RDKit."""

    name: ClassVar[str] = "RDKitSMILES"
    kind: ClassVar[IdentityKind] = IdentityKind.STEREOISOMER
    parent_algorithm: ClassVar[type[Algorithm]] = RDKitInChI

    @classmethod
    def identity_fn(
        cls, geo: Geometry, other_geos: dict[str, Geometry] | None = None
    ) -> str:
        """Generate SMILES from Geometry with RDKit."""
        inchi = cls.parent_algorithm.identity_fn(geo)
        if other_geos:
            smiles = next(
                s
                for s, g in other_geos.items()
                if cls.parent_algorithm.identity_fn(g) == inchi
            )
            if smiles:
                return smiles
        return Chem.MolToSmiles(Chem.RemoveAllHs(geom.rdkit_mol(geo)))

    @classmethod
    def geometry_fn(cls, value: str) -> Geometry:
        """Generate Geometry from SMILES with RDKit."""
        mol = Chem.MolFromSmiles(value)
        mol = Chem.AddHs(mol)
        return geom.from_rdkit_mol(mol)


@AlgorithmRegistry.register
class HillFormula(Algorithm):
    """Identify geometry with its molecular formula in Hill order."""

    name: ClassVar[str] = "HillFormula"
    kind: ClassVar[IdentityKind] = IdentityKind.FORMULA

    @classmethod
    def identity_fn(
        cls,
        geo: Geometry,
        other_geos: dict[str, Geometry] | None = None,  # noqa: ARG003
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
