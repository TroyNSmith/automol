# Molecular identity

`automol.Identity` wraps a string identifier (InChI, SMILES, ...) together
with the algorithm that produced it, so identifiers from different
algorithms are never accidentally compared or mixed up.

## Generating an identity from a `Geometry`

```python
from automol import RDKIT_INCHI, RDKIT_SMILES, Geometry, Identity

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

inchi = Identity.from_geometry(water, algorithm=RDKIT_INCHI)
smiles = Identity.from_geometry(water, algorithm=RDKIT_SMILES)

inchi.value       # "InChI=1S/H2O/h1H2"
inchi.algorithm   # "rdkit inchi"
inchi.kind        # "stereoisomer"
```

If you already have a string identifier from elsewhere, wrap it directly
with `from_value` instead of recomputing it:

```python
inchi = Identity.from_value("InChI=1S/H2O/h1H2", algorithm=RDKIT_INCHI)
```

## Going back to a `Geometry`

Algorithms that support the inverse direction can reconstruct a `Geometry`
from the identifier:

```python
water_rt = inchi.geometry()
```

Calling `.geometry()` on an identity produced by an algorithm with no known
inverse raises `NotImplementedError`.

## `kind`

Every registered algorithm is tagged with a `kind` — a category describing
what sort of identity it produces (currently `"stereoisomer"` for both
built-in RDKit algorithms). `Identity.kind` is set automatically by
`from_geometry` and `from_value`, and is validated against the registered
algorithm's `kind` on construction — an explicit mismatch raises a
`ValueError`. This lets code group or dispatch on `kind` without hardcoding
a specific algorithm.

## How algorithms are implemented

An algorithm is just a plain string identifier — there's no closed enum, so
higher-level packages can register their own algorithms without touching
`automol` itself. Behavior is registered via `automol.ident.AlgorithmRegistry`,
by subclassing `AlgorithmFns` and decorating it with the algorithm's
identifier and `kind`:

```python
from automol.ident import AlgorithmFns, AlgorithmRegistry

@AlgorithmRegistry.register("rdkit inchi", "stereoisomer")
class RDKitInChI(AlgorithmFns):
    @staticmethod
    def identity_fn(geo: Geometry) -> str:
        ...  # Geometry -> InChI

    @staticmethod
    def geometry_fn(value: str) -> Geometry:
        ...  # InChI -> Geometry
```

`geometry_fn` is optional — omit it (or fall back to `AlgorithmFns`'s
default) for an algorithm that only supports the forward direction; calling
`.geometry()` on such an identity raises `NotImplementedError`, as above.

`automol.ident` exposes its built-in algorithm identifiers as module-level
constants (`RDKIT_INCHI`, `RDKIT_SMILES`, `HILL_FORMULA`) so callers don't
need to hardcode the raw strings; a new package can follow the same pattern
for its own algorithms.

Registering an algorithm twice raises
`automol.utils.exc.AlgorithmAlreadyRegisteredError`; looking up one that was
never registered raises `automol.utils.exc.UnknownAlgorithmError`.
