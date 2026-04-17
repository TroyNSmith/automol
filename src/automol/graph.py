"""Core functions."""

import networkx as nx
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import Mol, RWMol

from . import rd


class _CustomBaseModelMeta(ModelMetaclass):
    def __getattr__(self, item: str):  # noqa: ANN204
        try:
            super().__getattr__(item)  # ty:ignore[unresolved-attribute]
        except AttributeError:
            if item in self.__dict__.get("__pydantic_fields__", ()):
                return item
            raise


class CustomBaseModel(BaseModel, metaclass=_CustomBaseModelMeta):
    """A custom base model that allows accessing field names as class attributes."""


class Atom(CustomBaseModel):
    """Represents an atom in a molecule."""

    symbol: str


class Bond(CustomBaseModel):
    """Represents a bond between two atoms in a molecule."""

    order: int


def validate(graph: nx.Graph) -> nx.Graph:
    """Validate the graph structure."""
    for key, data in graph.nodes(data=True):
        if not Atom.model_validate(data):
            msg = f"Node {key} does not have a valid Atom instance."
            raise ValueError(msg)
    for key1, key2, data in graph.edges(data=True):
        if not Bond.model_validate(data):
            msg = f"Edge ({key1}, {key2}) does not have a valid Bond instance."
            raise ValueError(msg)

    return graph


def from_smiles(smiles: str) -> nx.Graph:
    """
    Instantiate Graph from SMILES string.

    Parameters
    ----------
    smiles
        SMILES formatted string.

    Returns
    -------
    graph
        Graph.
    """
    mol = rd.mol.from_smiles(smiles, with_coords=False)
    return from_rdkit_mol(mol)


def from_inchi(inchi: str) -> nx.Graph:
    """
    Instantiate Graph from InChI string.

    Parameters
    ----------
    inchi
        InChI string.

    Returns
    -------
    graph
        Graph.
    """
    mol = rd.mol.from_inchi(inchi, with_coords=False)
    return from_rdkit_mol(mol)


def from_rdkit_mol(mol: Mol) -> nx.Graph:
    """
    Instantiate Graph from RKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
    graph
        Graph.
    """
    graph = nx.Graph()

    for mol_atom in mol.GetAtoms():
        atom = Atom(symbol=mol_atom.GetSymbol())
        graph.add_node(mol_atom.GetIdx(), **atom.model_dump())

    for mol_bond in mol.GetBonds():
        bond = Bond(order=mol_bond.GetBondTypeAsDouble())
        graph.add_edge(
            mol_bond.GetBeginAtomIdx(), mol_bond.GetEndAtomIdx(), **bond.model_dump()
        )

    return validate(graph)


def inchi(graph: nx.Graph) -> str:
    """
    Provide InChI string from Graph.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    mol = rdkit_mol(graph)
    return rd.mol.inchi(mol)


def rdkit_mol_with_index_map(graph: nx.Graph) -> tuple[Mol, dict[int, int]]:
    """Convert a graph back to an RDKit molecule."""
    rw_mol = RWMol()
    to_idx: dict[int, int] = {}

    for key in sorted(graph.nodes()):
        atom = Atom(**graph.nodes[key])
        idx = rw_mol.AddAtom(rdchem.Atom(atom.symbol))
        to_idx[key] = idx

    for key1, key2 in graph.edges():
        bond = Bond(**graph.edges[key1, key2])
        rw_mol.AddBond(to_idx[key1], to_idx[key2], rdchem.BondType(bond.order))

    to_key = dict(map(reversed, to_idx.items()))
    return rw_mol.GetMol(), to_key


def rdkit_mol(graph: nx.Graph, *, label: bool = False) -> Mol:
    """Convert a graph back to an RDKit molecule."""
    mol, to_key = rdkit_mol_with_index_map(graph)
    if label:
        mol = rd.mol.add_atom_numbers(mol, to_number=to_key)
    return mol
