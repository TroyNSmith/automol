# Contributing to automol

Thank you for your interest in contributing to **automol**!
Contributions of all kinds are welcome, including bug reports,
documentation improvements, and new features.

This document outlines the basic development workflow and coding
conventions used in the project.

## Development workflow

To get set up:
1. Install [Pixi](https://pixi.prefix.dev/latest/installation/)
2. Fork the repository
3. Clone the repository and run `pixi run init` inside it
To contribute code, submit pull requests with clear descriptions of the changes.
For larger contributions, create an issue first to propose your idea.

## Coding standards

Coding standards are largely enforced by the pre-commit hooks, which perform
formatting and linting ([Ruff](https://github.com/charliermarsh/ruff)),
import linting ([Lint-Imports](https://import-linter.readthedocs.io/en/stable/)),
static type-checking ([Ty](https://github.com/astral-sh/ty)),
and testing ([PyTest](https://docs.pytest.org/en/latest/))
with code coverage reports [CodeCov](https://docs.codecov.com/docs).

Docstrings follow the
[NumPy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

---

## Naming conventions

This project follows consistent naming conventions to clearly
distinguish **modules**, **types**, and **data-valued variables**. The
goal is to keep scientific code concise, readable, and free of name
collisions.

### Modules

Submodules are named after molecular data *domains* using short,
singular nouns:
```
automol.geom
automol.graph
automol.smiles
```
Modules act as **namespaces for algorithms and utilities**, not as
variable names.

**Rule:** Module names must not be used for data-valued variables.

---

### Types (data models)

Data structures are defined as singular, capitalized class names:

``` python
Geometry
Graph
Smiles
```

These classes represent molecular data objects and define their schema
and validation.

---

### Variables (instances of data types)

Variables holding instances of molecular data types use **short,
unambiguous abbreviations**, rather than full words or module names.

  Type              |Variable name
  ------------------|---------------
  `Geometry`        |`geo`
  `Graph`           |`gra`
  `Smiles`          |`smi`

Example:

``` python
from automol import geom, Geometry

geo = Geometry(["O", "H", "H"], coordinates)
com = geom.center_of_mass(geo)
```

**Rule:**
- `geom` refers to the **module** - `geo` refers to a **geometry
instance**

This distinction is used consistently throughout the codebase.

---

### Functions vs methods

Algorithms operating on molecular data are implemented as **module-level
functions**, not instance methods:

``` python
def center_of_mass(geo: Geometry) -> FloatArray:
    ...
```

This keeps data models lightweight and separates data representation
from algorithms, following standard scientific Python practice.

---

## Domain Ownership & Conversions

To maintain a decoupled suite, we follow a **Domain-Driven Design** approach. Each package in the suite is the owner of its specific objects and is the sole authority on how to translate those objects to/from external standards (like qcio).

### The Core Philosophy
> **"If you own the data, you own the interface."**
>
> Contributors should implement conversion logic within the package that defines the internal model. This prevents packages from needing to know the implementation details of every other tool in the suite.

### Example Ownership
| Package | Owned Object | Responsibility | Key Conversion Methods |
|---------|--------------|----------------|------------------------|
| **AutoMol** | `Geometry` | Coordinates, charge, spin, ... | ```from_geometry()``` ```to_geometry()``` |
| **AutoStore** | `Calculation` | Calculation arguments, metadata, provenance, ... | ```from_calculation()``` ```to_calculation()``` |

### Implementation Guidelines

#### 1. Decoupled Conversions

Conversion logic should be implemented as *standalone functions* rather than methods on the class. This keeps our core Pydantic/SQLModel objects lightweight and prevents external dependencies (like qcio or pint) from being required to instantiate a base model.

#### 2. Using the Shared Interface

When bridging objects between packages, always use the standalone conversion functions. For example, if AutoStore needs to generate a qcio `Structure` from an automol `Geometry`, it calls automol's geometry converter rather than manual dictionary mapping:
```python
# Autostore leverages AutoMol's ownership of Geometry
structure = automol.qc.structure.from_geometry(geo)
```

#### 3. Directory Structure & Abstraction Levels
To keep the core of the packages stable and independent, we follow a **Provider Pattern**. Core objects (like `Geometry` and `Calculation`) must remain "pure"--they should have zero knowledge of sub-packages or external libraries like RDKit, qcio, ...

Instead, the **sub-packages** provide bridges. They "import up" from the core and provide necessary translations.

Dependencies should always flow from the specific (sub-packages) to the general (core models).

* **Incorrect**: Putting `from_rdkit()` inside `geometry.py`. This forces the core to be dependent on RDKit.

* **Correct**: Putting `to_geometry()` inside `src/autopilot/rd/mol.py`.

**Rationale**: This allows the core packages to provide the framework for methods developed in this suite without bias towards specific software. Contributors can add support for new software by adding a new-subfolder without risking conflicts in the core model files or creating overly large core scripts.

---

## Questions

If you have questions about contributing or design decisions, feel free
to open an issue for discussion.
