# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.0.25] - 2026-09-19
### Added
- `Algorithm` `BaseModel` representing a registered algorithm as a standalone instance (`name`, `kind`, `identity_fn`, `geometry_fn`, `parent_algorithm`, `deterministic`), replacing the `AlgorithmDef` dataclass / `AlgorithmFns` ABC pair.
- `IdentityProtocol` / `GeometryProtocol` (`@runtime_checkable` `Protocol`s) describing the `identity_fn` / `geometry_fn` callable shapes, replacing the `Callable[...]` type aliases used by `AlgorithmDef`.
- `parent_algorithm` field on `Algorithm` / parameter on `AlgorithmRegistry.register(...)` so a non-canonical algorithm (e.g. `rdkit_smiles`, `hill_formula`) can disambiguate `other_geos` via a canonical parent (e.g. `rdkit_inchi`); `rdkit_smiles`'s `identity_fn` now returns the original (possibly non-canonical) SMILES from `other_geos` when its InChI matches, instead of always returning RDKit's canonical form.
- `deterministic` field on `Algorithm` / parameter on `AlgorithmRegistry.register(...)` (default `True`) flagging whether an algorithm produces deterministic strings; `rdkit_smiles` and `hill_formula` are registered as non-deterministic.
- Module-level `rdkit_inchi`, `rdkit_smiles`, `hill_formula` `Algorithm` instances, exported from the top-level `automol` namespace and called directly (`rdkit_inchi.identity_fn(...)`, `.geometry_fn(...)`) instead of through i`Identity`.
- `OTHER_GEOS` type alias (`Mapping[str, Geometry] | None`) for the `other_geos` parameter.

### Changed
- `AlgorithmRegistry.register(...)` is now a plain classmethod that registers an `Algorithm` instance directly (`AlgorithmRegistry.register(name=..., kind=..., identity_fn=..., geometry_fn=...)`), instead of a decorator applied to an `AlgorithmFns` subclass.
- `AlgorithmRegistry` stores algorithms in a public `algorithms: ClassVar[list[Algorithm]]` instead of a private `_algorithms: ClassVar[dict[str, AlgorithmDef]]`.

### Removed
- `Identity` `BaseModel` (`.from_geometry()`, `.from_value()`, `.geometry()`, and the `kind`/`algorithm` consistency validator) — replaced by calling `identity_fn` / `geometry_fn` directly on the registered `Algorithm` instances.
- `AlgorithmDef` dataclass and `AlgorithmFns` ABC — superseded by the `Algorithm` model and `IdentityProtocol` / `GeometryProtocol`.
- `AlgorithmRegistry.register_def()` — folded into `AlgorithmRegistry.register()`.
- `RDKIT_INCHI`, `RDKIT_SMILES`, `HILL_FORMULA` string constants — replaced by the `rdkit_inchi`, `rdkit_smiles`, `hill_formula` `Algorithm` instances.

## [0.0.24] - 2026-09-11
### Added
- `IdentityKind` `StrEnum` for categorizing identity types (`FORMULA`, `STEREOISOMER`, `CONFORMER`, `ISOMER`).
- `AlgorithmDef`, `AlgorithmFns`, `AlgorithmRegistry` exported from top-level `automol` namespace.
- `other_geos` parameter type (`dict[str, Geometry] | None`) in `AlgorithmFns.identity_fn` to support named reference geometries for conformer identity generation.

### Changed
- `Algorithm` `StrEnum` removed; algorithms are now plain string identifiers, so higher-level packages can register their own via `AlgorithmRegistry.register(algorithm, kind)` without modifying `automol.ident`. Built-in algorithms are exposed as module-level constants (`RDKIT_INCHI`, `RDKIT_SMILES`, `HILL_FORMULA`) instead of enum members.


## [0.0.23] - 2026-09-01
### Removed
- `geom.is_duplicate_conformer()` and `irmsd` dependency. (Broken `numpy` reference in solved `irmsd` version).
- `Algorithm.IRMSD` conformer-group identity algorithm.

### Fixed
- Dependencies listed in `pixi.toml` instead of `pyproject.toml`

## [0.0.22] - 2026-08-28
### Added
- `Geometry.relabel_atoms()` for reordering atoms by index.
- `geom.analysis.bond_graph()` / `orbit_classes()` for pynauty-based graph construction and automorphism-orbit detection.
- `geom.analysis.kabsch_align()`, `hungarian_correspondence()`, `assignment_rmsd()` for atom-correspondence-aware structural alignment and RMSD.
- `Algorithm.HILL_FORMULA` / `HillFormula` for Hill-ordered molecular formula identity via the `ident` registry.
- `pynauty` dependency.

### Changed
- `geom.{comparison,inertia,internal,properties,transform}` consolidated into `geom.analysis` (distance/inertia/vibration/comparison/graph analysis); `geom.view` renamed to `geom.io` and merged with xyz block/file I/O.
- `geom.hill_formula()` -> `Algorithm.HILL_FORMULA` / `HillFormula`, matching the pattern used by InChI/SMILES/SMG_HASH.
- `geom.transform.{translate,reflect,rotate,transition}` and `geom.internal.set_bond()` moved to `geom.core`.
- xyz parsing (`from_xyz_block`) reimplemented without `pyparsing`.
- `.gitignore`: drop `experimental`, add `.scratch`.

### Removed
- `automol.view` top-level re-export (use `automol.geom.io` / `automol.View`).

## [0.0.21] - 2026-07-26
### Added
- `geom.transform.transition()` for determining the transition-state geometry between two geometries via `StereoCondensedReactionGraph`.
- `Algorithm.SMG_HASH` / `StereoMolGraphHash` for enantiomer-invariant conformer identity via `StereoMolGraph` hashing.

### Changed
- `geom.internal.set_distance()` -> `set_bond()` for clarity.
- `geom.comparison.is_duplicate_conformer()` now returns a list the same length as `geos`, appending `False` for symbol-count mismatches instead of skipping them.


## [0.0.19] - 2026-07-17
### Added
- `Algorithm.IRMSD` for tagging conformer-group identities (unregistered algorithm; built directly via `Identity.from_value` rather than `from_geometry`).
- `geom.comparison.is_duplicate_conformer()` for iRMSD-based conformer matching.
- `geom.adjacency_matrix(..., flood_fill=True)` option for connectivity-based flood filling.
- `rd.mol.set_coordinates()` for replacing an RDKit mol's conformer coordinates.
- `geom.inertia`, `geom.internal`, `geom.vibration` modules (experimental): moments of inertia, internal coordinates, and vibrational analysis.
- `docs/source/*` pages (geometry, identity, interoperability, visualization, installation) and expanded README.

### Changed
- `automol.element` -> `automol.utils.element` to fit module layering.
- `automol.view` -> `automol.geom.view` since it consumes `Geometry` directly.
- `tests/test_geom.py` -> `tests/{test_core,test_properties,test_transform,test_comparison}.py` + `tests/conftest.py` to organize growing test suite.

### Removed
- `automol.geom.canon`, `automol.geoms`, `automol.graph` (including `graph.ts`) — superseded by current `geom`/`ident` design.

## [0.0.18] - 2026-07-02
### Added

### Changed
- `Geometry.canonical_form(self, *, in_place=True)` -> `.canonical_form(self, *, delta ...)` to support method chaining and discourage in-place operations on SQLModel subclasses.
- `_float_array_validator(...)` returns `np.array(obj, dtype...)` instead of `np.asarray(obj, dtype...)` due to instantiation concerns when hashing.

### Fixed
- Bug with `... for targets in nx.all_pairs_shortest_path...` exposed when operating on purely cyclic molecules.
- Premature raise on `Geometry.validate_coordinates_shape(...)` model validator exposed when validating SQLModel subclasses.
- Premature hash setting on `Geometry.set_hash()` model validator exposed when validating SQLModel subclasses.
- Instance building on `Geometry.canonical_form()` exposed when canonicalizing SQLModel subclasses.
- `test__deterministic_canonical_order(...)` to reflect updates.

### Removed
- `Geometry.sort()`.

## [0.0.17] - 2026-06-29
### Added
- `elements`, `constants`, and `ident` from `automatics` (package discontinued).
- `canonical_frame` to `geom` module for expanding `eckart_frame` logic to include sign choices.
- `canonical_sorting` to `geom` module for initial implementation of standardized atom sorting.

### Changed
- `geom.py` -> `geom/*` to better organize growing codebase.

### Fixed
- `tests` to reflect changes in update.

### Removed
- `is_similar` due to canonical framing and sorting.

## [0.0.16] - 2026-06-18
### Added
- `automatics.geom` module exports within `automol.geom` to avoid namespace clash.
- `harmonic_zpv` (harmonic zero point vibrational energy) method.
- `xyzrender` as a developer / optional dependency.

### Changed
- Propyl oxirane test fixtures to read objects from data files.
- Bump `automatics` to v0.0.6.

## [0.0.15] - 2026-06-17
### Added
- `geom.vibrational_analysis()` and corresponding functions to calculate frequencies from a `Geometry` and its Hessian.

### Changed
- Unit conversions import from `automatics`.
- `kabsch()` and `is_similar()` relocated from `geom` to `geoms`.

### Fixed
- Bump `automatics` to v0.0.5.
- Layering to incorporate new `geoms` module.

### Removed
- Minor comments in src files.

## [0.0.14] - 2026-06-12
### Added
- Dependency on automatics (0.0.4).

### Changed
- Update tests to reflect refactors.

### Removed
- Geometry, Identity, and View relocated to automatics for consistent source of truth in autosuite.
- Miscellaneous utility scripts pertaining to Geometry, Identity, and View.


## [0.0.13] - 2026-06-03
- Implemented Identity class with boilerplate for handling conversions between Geometry and chemical identifiers such as InChI / SMILES.
- Converted qcdata to an optional dependency with conversion methods placed in geom.py.
- Implemented a decorator for optional qc data dependency.
- Dropped qccompute from pixi.toml pypi-dependency list.

## [0.0.12] - 2026-05-20
- geom.is_similar() checks InChI first, no longer considers moment of inertia deviation, and ensures that geometry symbols are identically ordered between geo1 and geo2 (kabsch implentation is order dependent).
- Added nvalence, covalent radius, and group to elements-data.
- geom.determine_neighbors() as a first attempt at defining connectivity from geometries.

## [0.0.11] - 2026-05-04
- Added missing pyparsing dependency

## [0.0.10] - 2026-04-30
- Renames functions and arguments for clarity and consistency

## [0.0.9] - 2026-04-18
- Overhaul graph API with better design and better typing as Graph[Atom, Bond]
- Implement graph.ts submodule with brute-force reaction mapping algorithm

## [0.0.8] - 2026-04-16
- Added view submodule for building view objects
- Added geometry functions (translation, rotation, reflection, dihedral angles, etc.)
- Added graph submodule with conversion to/from RDKit Mol and SMILES/InChI

## [0.0.7] - 2026-04-08
- Added inertia moments, kabsch alignment, and center of mass algebraic methods to geom.py
- Added similarity analysis to geom.py (mirroring first two steps of prism_pruner)
- Added distance setting to geom.py

## [0.0.6] - 2026-04-01

## [0.0.5] - 2026-01-29
### Added
- Geometry hash function to root namespace

## [0.0.4] - 2026-01-29
### Changed
- Renamed geometry hash function to `geometry_hash()` to avoid shadowing built-in `hash()`

## [0.0.3] - 2026-01-28
### Added
- Geometry hash function

## [0.0.2] - 2026-01-28
### Fixed
- Fix Geometry.coordinate type annotation

## [0.0.1] - 2026-01-26
### Added
- Generate Geometry from SMILES
- Calculate Geometry center of mass
