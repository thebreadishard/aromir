# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Admission: a molecule is parsed from a SMILES or an xyz file and refused, with every
  reason, unless it is one neutral, closed-shell aromatic molecule of H, C, N, O, F, S
  and Cl.
- A start geometry with the atoms ordered by the bond graph: one order for every SMILES
  of a molecule; atoms the graph cannot tell apart keep the order of an xyz file.
- Geometry optimisation at B3LYP/6-31G* with pyscf and geomeTRIC.
- The analytic B3LYP Hessian and the atomic polar tensor from one orbital response, with
  the response checked against its own equations and the polar tensor against the
  translation sum rule.
- Harmonic bands with double-harmonic intensities, the scaled-B3LYP baseline, and a
  Lorentzian broadening function (no command writes a curve yet).
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

### Changed

- A band is licensed for the molecule it belongs to, not for its family alone: the
  family's error must be within the licence line on every hold-out, the molecule inside
  the training sizes, the band real for the seeds' mean and for every seed, and a
  CH-stretch or CH-oop band must move carbon-bound hydrogens (at most 10 % of its
  hydrogen motion on N–H, O–H or S–H). An O–H or N–H stretch, which the family rule
  counts as CH-stretch, a band of a molecule outside the training sizes and an
  imaginary band are printed as `no` with the reason (`licence_note` in
  `certificate.json`). A ring-ip band is licensed as the rule takes it.
- The certificate states what the measured error does not cover: how far the positions
  move for another atom order, a rotation, the mirror image and another listing of an
  xyz file, each marked where it exceeds the licence line; that the heights are
  B3LYP's; that one conformer was computed, and which stereo the SMILES left open; and
  each family's rule in words. A stereoisomer of a training molecule is no longer
  reported as never seen.
- A release can hold `invariance.json`, the measured spreads under a rotation, a
  reflection and another atom listing; v1.3's is measured and shipped, the largest seen
  on three molecules with a nitrile or an alkyne and on four without: 5.34 / 0.00 cm⁻¹
  under a rotation, 5.22 / 2.60 cm⁻¹ for the mirror image, 5.27 / 3.07 cm⁻¹ between
  listings of one xyz geometry. The model card and the README no longer say that the correction turns with
  the molecule by construction, or that every input gets one canonical atom order.

- Refused with a reason, before anything is computed: an isotope label (the masses are
  those of the most abundant isotopes), a SMILES followed by a blank, a name or a
  non-ASCII character (RDKit reads only up to the blank), an input without atoms, and,
  for `aromir spectrum`, a molecule outside the release's training sizes unless
  `--outside-training-sizes` is given.
- No failure a user can cause ends in a traceback: no 3D start geometry, `--out` naming
  a file, a release record that is not UTF-8, a missing, truncated or damaged weight
  file, weights that do not fit their recipe and a recipe under another seed's name
  are refusals (exit 2); any other file or numerical error is one line with its kind
  (exit 1), or its traceback when `AROMIR_TRACEBACK` is set. An xyz file whose comment
  line is not UTF-8 is read, and so is the xyz file of an imine or an amidine, whose
  N–H hydrogen RDKit keeps as a stereo mark.
- `aromir card --out` refuses an existing file and a missing folder.
- A run no longer loses its quantum chemistry to a failure that comes after it. The
  optimised geometry, the analytic Hessian and the measured soft modes are each saved
  when they end, with `run.json` saying what was asked; the same command into the same
  folder takes the run up. Each saved step is one file that names its own attempt and
  geometry, so a run killed between two writes resumes from a step that is whole or
  absent; a damaged file is refused in words. A Hessian with an imaginary mode is
  kept, not discarded.
- A folder left by a failed run no longer blocks the next one: it is resumed by the
  same request and refused, with the request it holds, for another.
- `aromir spectrum` with a release trained on analytic Hessians (v1.3) no longer skips
  the soft-mode repair and then refuses the molecule for a mode the repair would have
  measured (an audit counted 41 of 1,154 molecules of the training corpus with an
  imaginary lowest mode in the analytic Hessian). The soft modes are always measured; the minimum is decided
  and the positions are printed from the measured Hessian; the network reads the kind
  it was trained on; the certificate says so. From a real saddle point aromir steps
  0.1 bohr along the mode and optimises once more (tested with the quantum chemistry
  itself on planar hydrogen peroxide). That runs the optimisation, the Hessian and the
  repair a second time and can still end without a minimum; the certificate says when
  the structure is one off a saddle point.
- What the network cannot read is found on the start geometry, before the quantum
  chemistry, not after it.
- The scaled-B3LYP baseline uses the scale factors of the basis set the Hessian is
  computed with. Bauschlicher et al. (2018, ApJS 234, 32) fit them per basis set
  (Table 2): 0.979, 0.969 and 0.960 for 6-31G*. aromir used the row of 4-31G (0.956,
  0.952, 0.960), which put every baseline band below 1111 cm⁻¹ 2.3 % too low (20.7 cm⁻¹
  at 900 cm⁻¹) and those up to 2500 cm⁻¹ 1.7 % too low. The region edge is 1111.1 cm⁻¹
  (9 µm), not 1111.0.
- A band's printed position lies inside its printed range: the range is over the seeds
  and their mean, whose position is an eigenvalue of its own and can lie outside the
  seeds'.
- `--threads` also bounds the network's threads (torch took every core); the help says
  what `--max-memory-mb` is.
- `aromir spectrum` exits with 3, not 0, when a corrected band is imaginary for the
  seeds' mean; the result is still written. A band imaginary for one seed only is
  not licensed and named in a warning.
- A release is refused when its records disagree or cannot be true: an error that is
  negative or not finite, an element the package does not know, a smallest molecule
  larger than the largest, hold-outs named differently in the two records, or a
  hold-out count that the training record's list does not explain. A record's
  validation error names its file.

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
