# aromir

Infrared spectra of aromatic molecules: a cheap DFT Hessian, a learned correction to it,
and a certificate that says how far each band can be trusted.

> **Status: pre-alpha (0.1.0.dev0).** The whole chain is in the package: the low level,
> the network, the corrected spectrum and its certificate. Each step is tested on its
> own; no test yet runs the full chain with a release on a real molecule. The trained
> weights of network v1.3 ship inside the package. Positions are harmonic: the
> anharmonic step is not in this version; see
> [What works today](#what-works-today) and the [roadmap](#roadmap).

## The idea

A harmonic infrared spectrum needs a Hessian (band positions) and dipole derivatives
(band heights). An accurate Hessian is expensive, and a cheap one puts bands tens of
cm⁻¹ away from where they belong. aromir computes the cheap one, B3LYP/6-31G*, and
corrects it with a neural network trained on the difference to a better functional:

```
SMILES or xyz
  → atom order from the bond graph, start geometry (RDKit)
  → geometry optimisation                       (pyscf + geomeTRIC)
  → analytic Hessian and dipole derivatives     (pyscf)
  → ΔH, the learned correction to the Hessian   (network v1.3, inside the package)
  → band positions and heights
  → certificate: what each band rests on
```

The network is trained on neutral, closed-shell aromatic molecules of H, C, N, O, F, S
and Cl. A molecule outside that domain is refused with the reasons, not answered badly:
so is an isotope label, a SMILES with anything after it, and, for `aromir spectrum`, a
molecule smaller or larger than the training molecules unless
`--outside-training-sizes` asks for it. The command's exit code is 0 for a result, 2
for a refusal, 1 for a failure, and 3 for a corrected spectrum that was written but
holds an imaginary band (a band named a flat coordinate, see below, is no band position
and does not count there). A failure is one line; with `AROMIR_TRACEBACK=1` in the
environment an unforeseen one keeps its traceback, for a bug report.

## What works today

`aromir baseline` runs the pipeline up to and including the low level and writes the
B3LYP harmonic spectrum, with the conventional scaled-B3LYP positions beside it (the
three scale factors of Bauschlicher et al. 2018 for B3LYP/6-31G*). That
scaled spectrum is the baseline the certificate prints beside every corrected band.

```console
$ aromir check "c1ccc2ccccc2c1"
admitted: c1ccc2ccccc2c1 (C10H8, 18 atoms)

$ aromir baseline "c1ccccc1" --out results/benzene --threads 4
```

The result folder holds `geometry.json` (the optimised geometry), `lowlevel.npz` (the
Hessian and the polar tensor), `baseline.json` (bands, checks and provenance) and
`pyscf.log` (the quantum-chemistry log, written as the calculation runs).

`aromir spectrum` adds the learned correction and the certificate, with the network
release that ships inside the package (v1.3, in `src/aromir/releases/v1.3`, with its
[model card](src/aromir/releases/v1.3/MODEL_CARD.md)):

```console
$ aromir spectrum "c1ccccc1" --out results/benzene
```

`--release <folder>` names another release. A release is a folder with one weight file
and recipe per seed, `evaluation.json` (the
version's measured error per mode family and hold-out), `training.json` (the elements,
sizes and molecules it was trained on) and, where it was measured, `invariance.json`
(how far the answer moves when the same molecule is turned, mirrored or listed in
another atom order). The result folder then also holds
`correction.npz` (each seed's ΔH and their mean) and the certificate, as
`certificate.json` and `certificate.md`. The certificate lists every band with

- its corrected position and height, and how far the seeds disagree about the position;
- the uncorrected B3LYP band that moves most like it, plain and scaled;
- its family (CH-stretch, CH-oop, ring-ip or other: the evaluation's names for rules on a
  mode's position and motion, which the certificate spells out) and the error measured
  for that family on molecules the network never saw;
- whether the band is licensed: the family's error is within the licence line on every
  hold-out, the molecule lies inside the training sizes, its low-level mode is real, the
  band is real for the seeds'
  mean and for every seed, and a CH-stretch or CH-oop band moves carbon-bound hydrogens
  (an O–H stretch
  falls under the CH-stretch rule and is not licensed), with the reason where it is not;

and says whether the molecule lies inside the sizes the network was trained on, whether
the network has seen it, and what the measured error does not cover: how far the
positions move when the same molecule is turned, mirrored or listed in another atom
order. Nothing it states about the network is typed into the
package: every number comes from the release's records.

The error printed for a family is the error of what aromir delivers: the corrected band
positions of the seeds' mean against the reference, on molecules the network never
trained on. `aromir evaluate <folder>` measures it for a release on a folder of hold-out
molecules with stored Hessians (the format is in `aromir.evaluation`), together with the
first-order error of each seed that a training run reports, which is a different and
larger number. For v1.3 all four families are within the licence line; on the smaller
hold-out the family 'other' rests on eight molecules, and the card and the certificate
say what that is worth.

`aromir card` writes the release's model card the same way: what
the network is, what it was trained on, its measured error per family, and where it
stops being known.

Two checks and a repair guard every result:

- **The geometry is a minimum.** A geometry with an imaginary mode has no harmonic
  spectrum. aromir steps along the mode once and optimises again, which costs the
  optimisation and the Hessian a second time; if an imaginary mode remains, no spectrum
  is written, and the geometries and the Hessians stay in the folder.
- **The dipole derivatives satisfy their own equations.** pyscf's response solver can
  stop early ([pyscf#2702](https://github.com/pyscf/pyscf/issues/2702)), which a Hessian
  barely feels and a band height does. aromir checks the response against the
  coupled-perturbed equations, corrects it, and refuses the result if the residual stays
  above 1e-6 or the translation sum rule is violated by more than 5e-4 e.
- **Soft modes are measured.** pyscf's analytic DFT Hessian leaves out the response of
  the integration grid, which can move a mode below 20 cm⁻¹ by several cm⁻¹ or make it
  imaginary. For those modes aromir replaces the Hessian's columns by central
  differences of gradients that include the grid's response. Whether the geometry is a
  minimum is read from the measured Hessian, and the positions are printed from it. A
  network must be given what it was trained on, so a release trained on analytic
  Hessians (v1.3) reads the analytic one; its correction is added to the measured one.

Each step is saved when it ends: `geometry.json`, then `lowlevel.npz`, each naming the
run that made it, with `run.json` recording what was asked. After a step off a saddle
point the saddle's two files are kept as `saddle_point_geometry.json` and
`saddle_point_lowlevel.npz`. A run that fails after hours of quantum
chemistry, or is stopped, is taken up by the same command into the same folder; an
optimisation is taken up at its last gradient (`optimisation_progress.json`).

A molecule with a nearly free torsion can have no minimum the optimiser's displacement
criterion accepts, or none at all at this level. aromir then says what it did instead
of ending without a spectrum: a geometry that meets geomeTRIC's tight thresholds for the
energy and the forces but not, in 100 steps, the one for the displacement is named so
on the certificate with the last step's numbers, and a band whose low-level mode is
still imaginary and softer than 20 cm⁻¹ after one step off the saddle point is printed
as a flat coordinate — no band position, never licensed.

## The network

`aromir.network` holds the ΔH network: the internal coordinates and the pair pattern it
predicts on, the equivariant encoder, the pair head, and weights stored as safetensors
with a JSON recipe. It is a rewrite of the code the weights are trained with, and
regression pins hold the two to the same output: 16 molecule-and-configuration pairs
agree to 3e-7 relative in ΔH, with every integer input identical.

The correction is symmetric and free of translations by construction. It is free of
rotations and turns with the molecule where the internal coordinates hold no linear
angle; the tests check that on biphenyl with random networks. A nitrile or an alkyne
has linear angles, which are measured against a fixed axis in space: release v1.3 moved
the bands of three such molecules by up to 5.34 cm⁻¹ when they were turned, and its
model card and every certificate state the measured spreads. With the `canonical` angle choice and
`cos`-encoded dihedrals the network also gives one answer for every atom order; that
is tested on a molecule without symmetry-equivalent atoms, and v1.3 was not trained
with those settings.

A checkpoint of the training code becomes weight files with

```console
$ python -m aromir.network.convert checkpoint.pt weights/ v1.3-seed0 --network-version 1.3
```

## Install

Linux and macOS; on Windows, use WSL (pyscf does not build natively there). pyscf's
macOS wheels run on one thread, whatever `--threads` says.

```console
$ pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux: see below
$ pip install git+https://github.com/thebreadishard/aromir
```

The network runs on the CPU. On Linux, PyPI's default torch wheel bundles CUDA (a few
GB); installing the CPU wheel first, as above, avoids it. The second command needs git
and builds the package with hatchling, which pip fetches from PyPI; everything aromir
imports is declared and installed with it, and the weights of the bundled release are
inside the package.

The tests run on Linux and macOS with RDKit 2026.9, geomeTRIC 1.1, pyscf 2.15.0 (2.15.1
has no wheel for Apple silicon yet, so the bound is 2.15.0 itself) and torch
2.15 (2.14 on the development machine); on Python 3.12 and 3.13 with numpy 2.5 and scipy
1.18, on Python 3.11 with numpy 2.4 and scipy 1.17, the last series that support it. The
upper bounds in `pyproject.toml` are those series; the lower bounds are not tested. RDKit
decides the atom order and geomeTRIC the internal coordinates, and release v1.3's answer
depends on both, so a newer series of either is allowed only after the pins in
`tests/test_order_pins.py` pass with it. Those pins hold seven molecules' internal
coordinates atom by atom; they are no proof for a molecule that is not among them. The
versions are what pip chose in CI on 16 November 2026, not pins; each CI job lists its own.

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
| 0.1 (this version) | the network's trained weights as a release, and its model card; positions are harmonic, and the certificate says so |
| 0.2 | anharmonic band positions as an optional extra: a learned shift per band from a GFN2-xTB force field, with its measured error |
| 1.0 | an answer that does not depend on the atom order, the training code, releases on PyPI and conda-forge |

## Licence

Code: [MIT](LICENSE). The weights and records of a network release
(`src/aromir/releases/`): [CC BY 4.0](src/aromir/releases/LICENSE.md).
