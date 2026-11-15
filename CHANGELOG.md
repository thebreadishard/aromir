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
  a recipe, and the conversion of a training checkpoint. Trained weights are not
  included yet.
- The family of every mode (CH stretch, CH out-of-plane, ring in-plane, other).
- A network release (seeds, measured errors, training set) and `aromir spectrum`: the
  corrected spectrum with each seed's ΔH and a certificate, in JSON and Markdown, that
  states each band's family, licence, measured error and spread over the seeds.
- `aromir card`: a release's model card, written from the release's own files.
