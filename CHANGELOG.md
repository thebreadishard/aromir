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
- `aromir check` and `aromir baseline`.
