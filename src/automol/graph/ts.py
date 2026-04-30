"""Core molecular graph functions.

Uses NetworkX for graph representation, with Atom and Bond data validation.

Does not include bond order information.
"""

import copy
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Any

import networkx as nx
from rdkit.Chem import rdchem

from .core import Atom, Bond, Graph, isomorphisms, remove_bonds


class Change(StrEnum):
    """Changes."""

    FORMED = "formed"
    BROKEN = "broken"
    FLEETING = "fleeting"


class TransBond(Bond):
    """Represents a bond between two atoms in a molecule."""

    change: Change | None

    def to_rdkit_bond_type(self) -> rdchem.BondType:
        """Convert to an RDKit Bond Type."""
        if self.change is not None:
            return rdchem.BondType.HYDROGEN
        return rdchem.BondType.SINGLE


# From
def all_from_reactants_and_products(
    rct_gra: Graph[Atom, Bond],
    prd_gra: Graph[Atom, Bond],
) -> list[Graph[Atom, TransBond]]:
    """Fewest-bonds-first constructive count vector mappings."""
    gras, _ = all_from_reactants_and_products_with_mappings(rct_gra, prd_gra)
    return gras


# Algorithms
def is_isomorphic[AtomT: Atom, BondT: Bond](
    gra1: Graph[AtomT, BondT], gra2: Graph[AtomT, BondT]
) -> bool:
    """Check if two graphs are isomorphic."""
    atom_fields = gra1.atom_type.model_fields.keys()
    bond_fields = gra1.bond_type.model_fields.keys()

    def atom_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
        return all(n1[field] == n2[field] for field in atom_fields)

    def bond_match(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        return all(e1[field] == e2[field] for field in bond_fields)

    return nx.is_isomorphic(gra1, gra2, node_match=atom_match, edge_match=bond_match)


# Transition state graphs
BondKey = tuple[int, int]
BondSymbol = tuple[str, str]
FORMED_BOND = TransBond(change=Change.FORMED)
BROKEN_BOND = TransBond(change=Change.BROKEN)


def from_bond_changes(
    gra: Graph[Atom, Bond], bond_changes: dict[BondKey, Change]
) -> Graph[Atom, TransBond]:
    """Construct a transition graph from a graph and bond changes."""
    ts_gra = Graph(atom_type=Atom, bond_type=TransBond)
    ts_gra.add_nodes_from(gra.nodes(data=True))
    ts_gra.add_edges_from(gra.edges(), change=None)
    formed_bonds = {k for k, c in bond_changes.items() if c == Change.FORMED}
    broken_bonds = {k for k, c in bond_changes.items() if c == Change.BROKEN}
    ts_gra.add_edges_from(formed_bonds, change=Change.FORMED)
    ts_gra.add_edges_from(broken_bonds, change=Change.BROKEN)
    ts_gra.validate()
    return ts_gra


def bond_changes(
    gra: Graph[Atom, TransBond],
) -> dict[BondKey, Change]:
    """Extract the formed and broken bonds from a transition graph."""
    change = nx.get_edge_attributes(gra, TransBond.change)
    return {k: v for k, v in change.items() if v is not None}


def formed_bonds(gra: Graph[Atom, TransBond]) -> set[BondKey]:
    """Extract the formed bonds from a transition graph."""
    changes = bond_changes(gra)
    return {k for k, v in changes.items() if v == Change.FORMED}


def broken_bonds(gra: Graph[Atom, TransBond]) -> set[BondKey]:
    """Extract the broken bonds from a transition graph."""
    changes = bond_changes(gra)
    return {k for k, v in changes.items() if v == Change.BROKEN}


def reverse(gra: Graph[Atom, TransBond]) -> Graph[Atom, TransBond]:
    """Reverse the direction of a transition graph."""
    changes = bond_changes(gra)
    changes = {
        k: Change.FORMED if v == Change.BROKEN else Change.BROKEN
        for k, v in changes.items()
    }
    return from_bond_changes(gra, changes)


def reactants_graph(gra: Graph[Atom, TransBond]) -> Graph[Atom, Bond]:
    """Extract the reactant graph from a transition graph."""
    rct_gra = Graph(atom_type=Atom, bond_type=Bond)
    rct_gra.add_nodes_from(gra.nodes(data=True))
    rct_gra.add_edges_from(gra.edges(data=True))
    rct_gra.remove_edges_from(formed_bonds(gra))
    return rct_gra


def products_graph(gra: Graph[Atom, TransBond]) -> Graph[Atom, Bond]:
    """Extract the product graph from a transition graph."""
    prd_gra = Graph(atom_type=Atom, bond_type=Bond)
    prd_gra.add_nodes_from(gra.nodes(data=True))
    prd_gra.add_edges_from(gra.edges(data=True))
    prd_gra.remove_edges_from(broken_bonds(gra))
    return prd_gra


# Reaction mapping
def all_from_reactants_and_products_with_mappings(
    rct_gra: Graph[Atom, Bond],
    prd_gra: Graph[Atom, Bond],
) -> tuple[list[Graph[Atom, TransBond]], list[dict[int, int]]]:
    """Fewest-bonds-first constructive count vector mappings.

    Note: The mappings are from products to reactants!
    """
    bond_symbs1 = _bond_symbols(rct_gra)
    bond_symbs2 = _bond_symbols(prd_gra)
    counter1 = Counter(bond_symbs1.values())
    counter2 = Counter(bond_symbs2.values())
    diff_counter = copy.deepcopy(counter2)
    diff_counter.subtract(counter1)

    all_bond_symbs = sorted(diff_counter.keys(), key=lambda x: (-x.count("H"), x))

    break_counts1 = {k: -v for k, v in diff_counter.items() if v < 0}
    break_counts2 = {k: v for k, v in diff_counter.items() if v > 0}

    gras = []
    mappings = []
    bnd_changes_lst = []

    for extra_count in range(2):
        iter1 = itertools.combinations(all_bond_symbs, extra_count)
        for extra_symbs in iter1:
            iter2 = _iterate_break_bond_sets(
                rct_gra, prd_gra, break_counts1, break_counts2, extra_symbs=extra_symbs
            )
            for break_bonds1, break_bonds2 in iter2:
                gra1 = remove_bonds(rct_gra, break_bonds1)
                gra2 = remove_bonds(prd_gra, break_bonds2)

                iter3 = _iterate_reverse_isomorphisms_with_distinct_bond_changes(
                    gra1,
                    gra2,
                    break_bonds1,
                    break_bonds2,
                    bnd_changes_lst=bnd_changes_lst,
                )
                for bnd_changes, mapping in iter3:
                    gra = from_bond_changes(rct_gra, bnd_changes)
                    rct_gra_ = reactants_graph(gra)
                    prd_gra_ = products_graph(gra)

                    # Continue if reactant does not match
                    if not is_isomorphic(rct_gra, rct_gra_):
                        continue

                    # Continue if product does not match
                    if not is_isomorphic(prd_gra, prd_gra_):
                        continue

                    # Continue if not unique
                    if any(is_isomorphic(gra, g) for g in gras):
                        continue

                    gras.append(gra)
                    mappings.append(mapping)
                    bnd_changes_lst.append(bnd_changes)

        # If we found somthing, break
        if gras:
            break

    return gras, mappings


def _bond_symbols(G: Graph) -> dict[BondKey, BondSymbol]:  # noqa: N803
    """Extract the bond symbols from a transition graph."""
    return {
        (key1, key2): tuple(
            sorted([G.nodes[key1][Atom.symbol], G.nodes[key2][Atom.symbol]])
        )
        for key1, key2 in G.edges()
    }


def _bonds_by_symbol(G: Graph) -> dict[BondSymbol, set[BondKey]]:  # noqa: N803
    """Group bonds by their symbols."""
    bond_symbs = _bond_symbols(G)
    bonds_by_symb = defaultdict(set)
    for bond_key, bond_symb in bond_symbs.items():
        bonds_by_symb[bond_symb].add(bond_key)
    return dict(bonds_by_symb)


def _iterate_break_bond_sets(
    gra1: Graph[Atom, Bond],
    gra2: Graph[Atom, Bond],
    break_counts1: dict[BondSymbol, int],
    break_counts2: dict[BondSymbol, int],
    extra_symbs: Sequence[BondSymbol] = (),
) -> Iterator[tuple[tuple[BondKey, ...], tuple[BondKey, ...]]]:
    """Fewest-bonds-first constructive count vector mappings.

    Note: The mappings are from products to reactants!
    """
    break_counts1 = defaultdict(int, break_counts1)
    break_counts2 = defaultdict(int, break_counts2)
    for bond_symb in extra_symbs:
        break_counts1[bond_symb] += 1
        break_counts2[bond_symb] += 1

    for break_bonds1 in _iterate_bond_sets(gra1, break_counts1):
        for break_bonds2 in _iterate_bond_sets(gra2, break_counts2):
            yield break_bonds1, break_bonds2


def _iterate_bond_sets(
    G: Graph,  # noqa: N803
    counts: dict[BondSymbol, int],
) -> Iterator[tuple[BondKey, ...]]:
    """Iterate over all combinations of bonds to form or break."""
    bonds_by_symb = _bonds_by_symbol(G)
    combo_iters = [
        itertools.combinations(bonds_by_symb[symb], count)
        for symb, count in counts.items()
    ]

    for combos in itertools.product(*combo_iters):
        yield tuple(itertools.chain.from_iterable(combos))


def _iterate_reverse_isomorphisms_with_distinct_bond_changes(
    gra1: Graph[Atom, Bond],
    gra2: Graph[Atom, Bond],
    break_bonds1: Sequence[BondKey],
    break_bonds2: Sequence[BondKey],
    bnd_changes_lst: list[dict[BondKey, Change]],
) -> Iterator[tuple[dict[BondKey, Change], dict[int, int]]]:
    """Iterate over reverse_isomorphisms with distinct bond changes."""
    bnd_changes_lst = copy.copy(bnd_changes_lst)
    mappings = isomorphisms(gra2, gra1)
    for mapping in mappings:
        break_bonds = break_bonds1
        form_bonds = {tuple(sorted(map(mapping.get, b))) for b in break_bonds2}
        bnd_changes = {
            **dict.fromkeys(break_bonds, Change.BROKEN),
            **dict.fromkeys(form_bonds, Change.FORMED),
        }
        if bnd_changes not in bnd_changes_lst:
            bnd_changes_lst.append(bnd_changes)
            yield bnd_changes, mapping
