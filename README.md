# aromir

Infrared spectra of aromatic molecules: a cheap DFT Hessian, a learned correction to it,
and a certificate that says how far each band can be trusted.

> **Status: pre-alpha (0.1.0.dev0).** The low level works end to end and is tested. The
> network's code is in the package and reproduces its training code's output, but its
> trained weights are not published yet, so no corrected spectrum can be computed. The
> anharmonic step and the certificate are not in this version; see
> [What works today](#what-works-today) and the [roadmap](#roadmap).

## The idea

A harmonic infrared spectrum needs a Hessian (band positions) and dipole derivatives
(band heights). An accurate Hessian is expensive, and a cheap one puts bands tens of
cm⁻¹ away from where they belong. aromir computes the cheap one, B3LYP/6-31G*, and
corrects it with a neural network trained on the difference to a better functional:

```
SMILES or xyz
  → canonical atom order, start geometry        (RDKit)
  → geometry optimisation                       (pyscf + geomeTRIC)
  → analytic Hessian and dipole derivatives     (pyscf)
  → ΔH, the learned correction to the Hessian   (code here; weights not published yet)
  → band positions, heights, spectrum
  → certificate: what each band rests on        (not in this version)
```

The network is trained on neutral, closed-shell aromatic molecules of H, C, N, O, F, S
and Cl. A molecule outside that domain is refused with the reasons, not answered badly.

## What works today

`aromir baseline` runs the pipeline up to and including the low level and writes the
B3LYP harmonic spectrum, with the conventional scaled-B3LYP positions beside it. That
scaled spectrum is the baseline the learned correction will be compared with.

```console
$ aromir check "c1ccc2ccccc2c1"
admitted: c1ccc2ccccc2c1 (C10H8, 18 atoms)

$ aromir baseline "c1ccccc1" --out results/benzene --threads 4
```

The result folder holds `geometry.json` (the optimised geometry), `lowlevel.npz` (the
Hessian and the polar tensor), `baseline.json` (bands, checks and provenance) and
`pyscf.log` (the quantum-chemistry log, written as the calculation runs).

Two checks guard every result:

- **The geometry is a minimum.** A geometry with an imaginary mode has no harmonic
  spectrum, and none is written.
- **The dipole derivatives satisfy their own equations.** pyscf's response solver can
  stop early ([pyscf#2702](https://github.com/pyscf/pyscf/issues/2702)), which a Hessian
  barely feels and a band height does. aromir checks the response against the
  coupled-perturbed equations, corrects it, and refuses the result if the residual stays
  above 1e-6 or the translation sum rule is violated by more than 5e-4 e.

## The network

`aromir.network` holds the ΔH network: the internal coordinates and the pair pattern it
predicts on, the equivariant encoder, the pair head, and weights stored as safetensors
with a JSON recipe. It is a rewrite of the code the weights are trained with, and
regression pins hold the two to the same output: 16 molecule-and-configuration pairs
agree to 3e-7 relative in ΔH, with every integer input identical.

The correction is symmetric and free of translations and rotations by construction, and
it turns with the molecule; the tests check each of those on random networks. With the
`canonical` angle choice and `cos`-encoded dihedrals it also gives one answer for every
atom order; that is tested on a molecule without symmetry-equivalent atoms.

A checkpoint of the training code becomes weight files with

```console
$ python -m aromir.network.convert checkpoint.pt weights/ v1.3-seed0 --network-version 1.3
```

## Install

Linux and macOS; on Windows, use WSL (pyscf does not build natively there). pyscf's
macOS wheels run on one thread, whatever `--threads` says.

```console
$ pip install git+https://github.com/thebreadishard/aromir
```

## Develop

```console
$ git clone https://github.com/thebreadishard/aromir && cd aromir
$ pip install -e ".[dev]"
$ pre-commit install
$ pytest                      # everything, including seconds of quantum chemistry on water
$ pytest -m "not quantum"     # the linear algebra and the chemistry-free checks only
$ ruff check . && ruff format --check . && mypy
```

[CONTRIBUTING.md](CONTRIBUTING.md) states the rules the code is held to.

## Roadmap

| version | what it adds |
|---|---|
| 0.1 | the network's trained weights, the anharmonic band positions, the certificate, `aromir spectrum` |
| 1.0 | an answer that does not depend on the atom order, the training code, a model card generated from the evaluation records, releases on PyPI and conda-forge |

## Licence

Code: [MIT](LICENSE). Weights and data, once published: CC BY 4.0.
