# Contributing

aromir turns a molecule into numbers people will compare with measurements. The rules
below exist so that every one of those numbers can be trusted and traced.

## Set up

```console
$ pip install -e ".[dev]"
$ pre-commit install
```

## Before a pull request

```console
$ ruff check . && ruff format --check .
$ mypy
$ pytest --cov
```

CI runs the same commands on Linux and macOS. A pull request is merged when they pass
and a reviewer has read the change.

## What the code is held to

**One job per function.** ruff enforces the limits (50 statements, 12 branches, 6
arguments). A function that outgrows them is split; settings that outgrow an argument
list become one validated object.

**Settings are checked before anything runs.** A settings object is a frozen pydantic
model that forbids unknown fields. Options that do not combine are refused when the
object is built, never after the first hour of computation. The level of theory is not
a setting: the network is trained on one level, and the code has one.

**Fail loudly.** Raise one of the exceptions in `aromir.errors` with the reason. No
`assert` for a runtime check, no silent fallback, no default that hides a missing input.

**One implementation.** Every constant and every formula is defined once and imported.
`aromir.constants` holds the physical constants; `aromir.vibrations` holds the
vibrational arithmetic.

**No dead code.** Unused variables, commented-out code and stale docstrings leave in the
change that orphans them. Code for a later version arrives with that version.

**Results are written whole.** A record goes to a temporary file and is moved over its
target. A result folder is never overwritten.

**Types and names.** Every function is annotated and mypy runs in strict mode. Names
say what a thing is. The matrices keep the literature's letters (H, B, F, P).

**Comments say why.** What the code does is the code's job; history is git's.

## Tests

Every behaviour has a test, and every switch has a test for each side. Three kinds
carry the weight:

- **A known answer**: a spring's frequency, the intensity of a stretch between two
  point charges.
- **An invariance**: a rotated molecule has the same spectrum.
- **An independent counterpart**: the polar tensor against finite differences of the
  dipole; the Hessian against pyscf's own.

Tests that run quantum chemistry are marked `quantum` and use water, which takes
seconds. `pytest -m "not quantum"` runs the rest in seconds.

## Claims

"Validated" is written only with a number and a counterpart: not "validated on water"
but "polar tensor against finite differences of the dipole on water, 1.2e-5 e". A new
call into a library names, in its docstring, the upstream example or test that makes
the same call.

## Commits

One change per commit; the subject line says what is now true. The changelog gets a
line for anything a user can see.
