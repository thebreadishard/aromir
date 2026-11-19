# aromir

Infrared spectra of aromatic molecules: a cheap DFT Hessian, a learned correction to it,
and a certificate that says how far each band can be trusted.

> **Status: pre-alpha (0.1.0.dev0).** The whole chain is in the package: the low level,
> the network, the corrected spectrum and its certificate. Each step is tested on its
> own; the test suite does not run the full chain (a real molecule takes minutes of
> quantum chemistry). The full chain has been run by hand with network v1.3 on benzene,
> benzonitrile, a methoxypicoline and 3-vinylfuran (16 November 2026; the records are
> in `probes/results_m1` of thebreadishard/udacity-capstone-plan). The trained
> weights of network v1.4 ship inside the package. Positions are harmonic: the
> anharmonic step is not in this version; see
> [What works today](#what-works-today) and the [roadmap](#roadmap).

## The idea

A harmonic infrared spectrum needs a Hessian (band positions) and dipole derivatives
(band heights). An accurate Hessian is expensive, and a cheap one puts bands tens of
cm⁻¹ away from where they belong. aromir computes the cheap one, B3LYP/6-31G*, and
corrects it with a neural network trained on the difference to ωB97X/6-31G*, a more
expensive functional (the "DFT proxy" of the model card: the level the network is taught
to reach, standing in for a still better one that is too expensive to train on):

```
SMILES or xyz
  → atom order from the bond graph, start geometry (RDKit)
  → geometry optimisation                       (pyscf + geomeTRIC)
  → analytic Hessian and dipole derivatives     (pyscf)
  → ΔH, the learned correction to the Hessian   (network v1.4, inside the package)
  → band positions and heights
  → certificate: what each band rests on
```

**What the numbers mean.** A corrected position aims at the harmonic position
ωB97X/6-31G* would give, not at a measured spectrum, and every error aromir prints is
against that calculation. Measured bands lie, for nearly all modes, below harmonic ones,
by most for the CH stretches (the scaled baseline's factors say as much: 0.960 above
2500 cm⁻¹, 0.979 below 1111 cm⁻¹), and aromir does not yet add that anharmonic shift. The scaled B3LYP position
printed beside each band is the conventional estimate of a measured position, and is
there as the baseline; band heights are computed on the corrected modes with B3LYP's
dipole derivatives, which are not corrected, and carry no error bar. The errors are
measured on two test sets of molecules the networks the error is measured on never saw:
*test set (a): unseen molecules*, molecules set aside before any network was trained, and
*test set (b): held-out scaffolds*, the substituted molecules of two ring scaffolds,
fluorene and fluoranthene, set aside as a whole (the model card says what the training
set holds of them). In code and records these are the hold-outs: never trained on in
any form — a twin or another charge state included. A band is
*licensed* — a word for "passes the accuracy test", not the software licence — when its
family's error is at most the release's line (3 cm⁻¹ for v1.3 and v1.4) on both test sets,
and the band meets the other conditions listed under
[What works today](#what-works-today).

aromir accepts neutral, closed-shell aromatic molecules of H, C, N, O, F, S and Cl, the
molecules its networks are trained on (v1.4's also trained on the radical cations of 50 of
them). A molecule outside that domain is refused with the reasons, not answered badly:
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
release that ships inside the package (v1.4, in `src/aromir/releases/v1.4`, with its
[model card](src/aromir/releases/v1.4/MODEL_CARD.md); v1.3 stays beside it in
`src/aromir/releases/v1.3` for `--release`):

```console
$ aromir spectrum "c1ccccc1" --out results/benzene
```

`--release <folder>` names another release. A release is a folder with one weight file
and recipe per seed, `evaluation.json` (the
version's measured error per mode family and test set), `training.json` (the elements,
sizes and molecules it was trained on) and, where it was measured, `invariance.json`
(how far the answer moves when the same molecule is turned, mirrored or listed in
another atom order).

A version can be a pair of networks trained on one recipe. Its *measurement networks* are
trained without the test sets' molecules, and every error and the licence are read on them;
its *release networks* are trained on the labelled entries of their training pool, the
test sets' molecules included, and are what the folder ships. v1.4 is such a pair. Its folder holds the release networks'
seeds and `training.json`, the measurement networks' training record as
`measurement.json`, and `release.json`: the hashes of the shipped seeds' tensor files
(a folder whose seeds are other ones is refused), the SHA-256 of the content of the four
records beside it (the content as Python's JSON reader gives it, so 1 and 1.0 differ; a
record changed afterwards is refused: a guard against an accidental
change, not a signature, and `release.json`'s own fields are bound by nothing), the
checkpoints of both kinds by hash and the
checks the release networks passed before they were shipped (the same recipe, agreement
with the measurement networks on the pool, the test sets learnt, early stopping off the
cap, and benzene by name). The certificate and the card say in one sentence that the
spectrum is made by the release networks and that its error bars are the measurement
networks', and a molecule the release networks trained on is said to be in their
training data, where the error bar is not its out-of-sample error.

The result folder then also holds
`correction.npz` (each seed's ΔH and their mean) and the certificate, as
`certificate.json` and `certificate.md`. The certificate lists every band with

- its corrected position and height, and how far the seeds disagree about the position;
- the uncorrected B3LYP band that moves most like it, plain and scaled;
- its family (CH-stretch; CH-oop, a CH out-of-plane bend; ring-ip, a ring in-plane mode;
  or other: the evaluation's names for rules on a
  mode's position and motion, which the certificate spells out) and the error measured
  for that family on molecules the network never saw (for release networks: that the
  measurement networks never saw);
- whether the band is licensed: the family's error is within the licence line on both
  test sets, the molecule lies inside the training sizes, its low-level mode is real, the
  band is real for the seeds'
  mean and for every seed, molecules of the molecule's group (with or without a linear
  group: a nitrile, an alkyne) moved by no more than the licence line under a rotation, a
  mirror image or another xyz listing in the release's `invariance.json`, and a CH-stretch
  or CH-oop band moves carbon-bound
  hydrogens (an O–H stretch
  falls under the CH-stretch rule and is not licensed), with the reason where it is not;

and says whether the molecule lies inside the sizes the network was trained on, whether
the network has seen it, and what the measured error does not cover: how far the
positions move when the same molecule is turned, mirrored or listed in another atom
order. Nothing it states about the network is typed into the
package: every number comes from the release's records.

The error printed for a family is the error of the corrected band positions of the seeds'
mean against the reference, on molecules the networks never trained on: for v1.3 the error
of what aromir delivers, for v1.4 that of its measurement networks, the error bar of what
its release networks deliver. `aromir evaluate <folder>` measures it for a release on a folder of test-set
molecules with stored Hessians (the format is in `aromir.evaluation`), together with the
first-order error of each seed that a training run reports, which is a different and
larger number. It refuses release networks, which trained on the test sets: v1.4's errors
were measured with it on the version's measurement networks (whose weights are not
shipped), and agree to 0.005 cm⁻¹ with a second read by aromir's code on the training
project's stored Hessians in the same atom order; the training project's own read, in its
corpus atom order, differs by up to 0.05 cm⁻¹, because the network is not order-free. For
v1.4 all four families are within the licence line; on test set (a), of eight
molecules, the resampling intervals of CH-oop and 'other' cross it, and the card and the
certificate say what that is worth.

The licence also reads how far the same molecule's positions move when it is turned,
mirrored or listed in another order, which a release's `invariance.json` records for
molecules with a linear group (a nitrile, an alkyne) and without one. While the largest of
those spreads for a molecule's group is above the licence line, no band of the molecule
is licensed; its corrected positions are printed all the same, with the reason. A
molecule with a linear group is not licensed either where its group's spread was not
measured; one without keeps its licence there. A linear angle is measured against a fixed
axis in space, which is why such molecules move most: v1.4's record gives up to 11.20
cm⁻¹ with a linear group and 2.85 without, so molecules with one are not licensed and
the others are; v1.3's gives 5.34 and 3.07, so none is. A release whose record puts the
spread at or below the line gets the licence back with no change to the code.

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
run that made it, with `run.json` recording what was asked and the aromir version that
started the run. After a step off a saddle point the saddle's two files are kept as
`saddle_point_geometry.json` and `saddle_point_lowlevel.npz`. A run that fails after
hours of quantum chemistry, or is stopped, is taken up by the same command into the same
folder; an optimisation is taken up at its last gradient (`optimisation_progress.json`).
Only the version that started a run takes it up: another one refuses the folder and
names both.

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
has linear angles, which are measured against a fixed axis in space: when three such
molecules were turned, release v1.3 moved their bands by up to 5.34 cm⁻¹ and v1.4 by up
to 11.20 cm⁻¹ (diphenylacetylene, measured by `scripts/measure_invariance.py`), and each
release's model card and every certificate state the measured spreads. With the
`canonical` angle choice and `cos`-encoded dihedrals the network also gives one answer
for every atom order; that is tested on a molecule without symmetry-equivalent atoms, and
neither v1.3 nor v1.4 was trained with those settings.

A checkpoint of the training code becomes weight files with

```console
$ python -m aromir.network.convert checkpoint.pt weights/ v1.3-seed0 --network-version 1.3
```

## Install

Linux and macOS are tested in CI. Native Windows is not: there the two commands below
install aromir with pyscf's Windows wheel (pyscf 2.15.0 on Python 3.14, 18 November 2026,
where water's B3LYP/6-31G* energy and frequencies equal Linux's in every printed digit), but
no full `aromir spectrum` has been checked there, so WSL 2 (Ubuntu 26.04 was used) is the
tested route on Windows. On Windows, make the environment in a short folder (for example
`C:\Users\<you>\aromir`): torch's files have paths longer than Windows' default limit of 260
characters, and in a deep folder its installation fails. pyscf's macOS wheels run on one
thread, whatever `--threads` says.

```console
$ pip install torch --index-url https://download.pytorch.org/whl/cpu   # every platform; see below
$ pip install git+https://github.com/thebreadishard/aromir
```

The network runs on the CPU. On Linux, PyPI's default torch wheel bundles CUDA (a few
GB); installing the CPU wheel first, as above, avoids it. The second command needs git
and builds the package with hatchling, which pip fetches from PyPI; everything aromir
imports is declared and installed with it, and the weights of the bundled release are
inside the package.

The tests run on Linux and macOS with RDKit 2026.9, geomeTRIC 1.1, pyscf 2.15.0 (2.15.1
cannot be installed on Apple silicon yet, so the bound is 2.15.0 itself) and torch
2.15 (2.14 on the development machine); on Python 3.12 and 3.13 with numpy 2.5 and scipy
1.18, on Python 3.11 with numpy 2.4 and scipy 1.17, the last series that support it. The
upper bounds in `pyproject.toml` are those series; the lower bounds are not tested, except
RDKit's: a job in CI loads both releases, admits molecules and checks the atom-order
pins with RDKit 2026.9.2, the lowest allowed, and with the newest 2026.9. The first
release of each series from 2024.9 to 2026.9 left phenyl azide a double bond open there.
RDKit
decides the atom order and geomeTRIC the internal coordinates, and the releases' answers
depend on both, so a newer series of either is allowed only after the pins in
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

Open, not scheduled: the weights on the Hugging Face Hub as well, with a revision-pinned
download for machines that install aromir without the weights (parked on 18 November
2026). Until then the weights ship only inside the package, and nothing in it refers to
the Hub.

## Licence

Code: [MIT](LICENSE). The weights and records of a network release
(`src/aromir/releases/`): [CC BY 4.0](src/aromir/releases/LICENSE.md).
