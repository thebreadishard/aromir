# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Admission: a molecule is parsed from a SMILES or an xyz file and refused, with every
  reason, unless it is one neutral, closed-shell aromatic molecule of H, C, N, O, F, S
  and Cl.
- A start geometry in one canonical atom order for every writing of a molecule.
- Geometry optimisation at B3LYP/6-31G* with pyscf and geomeTRIC.
- The analytic B3LYP Hessian and the atomic polar tensor from one orbital response, with
  the response checked against its own equations and the polar tensor against the
  translation sum rule.
- Harmonic bands with double-harmonic intensities, the scaled-B3LYP baseline, and
  Lorentzian broadening.
- The repair of soft modes: a Hessian's modes below 20 cm⁻¹ measured from gradients that
  include the grid's response, for the baseline and for a release trained that way.
- `aromir check` and `aromir baseline`.
- The ΔH network (`aromir.network`): its inputs, the model, weights as safetensors with
  a recipe, and the conversion of a training checkpoint.
- The family of every mode (CH stretch, CH out-of-plane, ring in-plane, other).
- A network release (seeds, measured errors, training set) and `aromir spectrum`: the
  corrected spectrum with each seed's ΔH and a certificate, in JSON and Markdown, that
  states each band's family, licence, measured error and spread over the seeds.
- `aromir card`: a release's model card, written from the release's own files.
- The trained weights of network v1.3 (three seeds) with their evaluation and training
  records and model card, inside the package under CC BY 4.0; `aromir spectrum` and
  `aromir card` use that release unless `--release` names another.

### Fixed

- The network's correction no longer depends on the LAPACK build. The pseudo-inverse of
  the Wilson B matrix was `np.linalg.pinv` at its default cutoff, which lies inside the
  rounding residue of B's six null singular values: on numpy 2.5.4's Linux wheels one of
  them came back above the cutoff for 72 of the 1,240 molecules of the training corpus
  (4-cyanobiphenyl among them), B⁺ reached 1e13 and the corrected spectrum was
  meaningless, while another build gave the right answer for the same input.
  `aromir.network.primitives.pseudo_inverse` now keeps exactly the 3N − 6 largest singular
  values and refuses, with an `InternalCoordinatesError`, a B whose values on either side
  of that count are not where they belong (a linear molecule, primitives that miss a
  direction). With the v1.3 release the Linux build's ΔH for 76 corpus molecules, 71 of
  those 72 among them, now agrees with the Windows build's to 2.4e-7 hartree/bohr² (bands
  to 2e-4 cm⁻¹); on Windows 75 of the 76 are unchanged to the bit.
- An alkyne between two rings or an allene no longer breaks the pseudo-inverse on every
  build. geomeTRIC measures a linear angle against a fixed axis, so where two such groups
  share an axis that is bent by θ rad, a rotation enters B at about 0.01 θ of its largest
  singular value: 3.5e-8 for diphenylacetylene at its minimum (the 76th molecule above:
  ΔH 6e17 before, 0.056 hartree/bohr² now), 1e-4 for start geometries of unsymmetrical
  alkynes. The count drops that direction, and B⁺ then differs from the B⁺ of a B without
  rigid motions by about 0.3 θ of its largest element (measured on molecules of up to 60
  atoms); a bend above about 0.1 rad (6°) is refused, and so is a molecule in which that
  rotation comes within a tenth of the smallest real singular value — rods of several
  alkynes and some 90 atoms or more.
