# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Release networks beside measurement networks: a release folder may ship networks trained
  on their whole training pool, hold-outs included, with `release.json` (the shipped
  seeds' tensor hashes, the checkpoints of both kinds by hash and the release checks, all
  passed) and `measurement.json` (the training record of the measurement networks its
  errors belong to). Such a folder is refused when one record comes without the other, a
  check failed, the seeds are not the ones `release.json` names, the release networks lack
  a molecule the measurement networks trained on or a hold-out molecule, or they list
  hold-outs; or when the two training records give another domain (elements, sizes, the
  soft-mode repair, the cation test, cations the release lacks), `release.json` counts
  fewer entries than the molecules and cations it trained on, names other checks than the
  five, or names measurement checkpoints the evaluation's sources do not, or a record is
  not the one `release.json` binds by the SHA-256 of its content (a guard against an
  accidental change; `release.json`'s own fields are not bound). The certificate and the card
  say which networks made the spectrum and which ones the error bars are of; a molecule in
  the release networks' training data is said to be there; a neutral molecule whose
  radical cation the networks trained on, but not the molecule itself, gets the measured
  cation test as a clause. `aromir evaluate` refuses release networks.
- Network v1.4 ships inside the package as its release networks (1132 labelled entries:
  1078 distinct neutral molecules and the radical cations of 50), with the errors and
  licence of its measurement networks, measured with `aromir evaluate`; v1.3's folder stays
  in the package for `--release`.
- The evaluation record's format 3: each family's band error counts the hold-out molecules
  that have a band of the family, and a caveat says "on 5 of the hold-out's 8 molecules"
  where it said "on 8". A record of format 2 is refused with the reason; v1.3's and
  v1.4's records are measured again with `aromir evaluate` (every number unchanged).
- The loader refuses an evaluation record whose rms lies outside its own resampling
  interval, whose share of resamples within the line contradicts the interval, that counts
  more molecules with a band than bands, or whose tables or hold-out sizes differ from the
  `measured.json` beside it.
- v1.4 ships `invariance.json`, measured on its release networks by
  `scripts/measure_invariance.py` with v1.3's molecules and trials (the script gives v1.3's
  record to the last digit): with / without a nitrile or an alkyne, 11.20 / 0.00 cm⁻¹ under
  a rotation, 4.46 / 2.67 cm⁻¹ for the mirror image, 11.03 / 2.85 cm⁻¹ between listings of
  one xyz geometry; the largest with a linear group is diphenylacetylene's. The certificate
  and the card state these numbers where they said "not measured for this release"; the
  card names the records `release.json` binds, which do not include `invariance.json`. The
  strict invariance xfails cover v1.4 beside v1.3, and each release's largest change on the
  stored molecules is pinned.

- Admission: a molecule is parsed from a SMILES or an xyz file and refused, with every
  reason, unless it is one neutral, closed-shell aromatic molecule of H, C, N, O, F, S
  and Cl.
- A start geometry with the atoms ordered by the bond graph: one order for every SMILES
  of a molecule; atoms the graph cannot tell apart keep the order of an xyz file.
- Geometry optimisation at B3LYP/6-31G* with pyscf and geomeTRIC.
- The analytic B3LYP Hessian and the atomic polar tensor from one orbital response, with
  the response checked against its own equations and the polar tensor against the
  translation sum rule.
- Harmonic bands with double-harmonic intensities and the scaled-B3LYP baseline.
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

- The certificate names the weights and the release it was made with: `certificate.json`
  records each seed's tensor SHA-256 by seed name (as `release.json` binds them) and the
  release, named without the machine's paths: `aromir/releases/v1.4` for a release inside
  the package (`inside_package`), the folder's name for any other; `certificate.md` says
  the same in one line under the band table.
- `invariance.json` names the tensor SHA-256 of each seed it was measured on
  (`tensor_sha256`, written by `scripts/measure_invariance.py`), and the loader refuses a
  record without it or of other weights than the seeds beside it, and a folder whose
  model card names `invariance.json` without the file. v1.3's and v1.4's records carry
  their seeds' hashes, added by the script's own code without measuring again (the spreads
  and sources are unchanged; the seeds have not changed since either was measured), and
  both cards say so. CONTRIBUTING says how the next release binds the record in
  `release.json`.
- RDKit's lower bound is 2024.9 (was 2023.9): older series are built against numpy 1
  and do not import under numpy 2, which the numpy bound allows. A job in CI runs the
  release loading, the admission tests and the atom-order pins on 2024.9.1 and on the
  newest 2026.9.
- A double bond that can have no E/Z configuration — one with a terminal atom, or with
  a carbon or a nitrogen that carries a second double bond, as in an azide, an allene or
  a ketene (a sulfur with two double bonds is bent, and its bonds are kept) — is never
  named as an unassigned double bond. RDKit before 2026.9.2 offers phenyl
  azide's N=N bonds as potential stereo bonds, and the certificate would have named
  one; with 2026.9.2 nothing changes for any molecule of the releases' lists.
- A release record's SMILES lists are compared by molecule, not by spelling: each entry
  is read in the standard form the running RDKit writes, which must be that RDKit's own
  canonical SMILES of the entry (an entry admission would write otherwise, such as one
  with atom-map numbers, is not in the standard form), and every
  comparison with the lists — whether a molecule was seen, the counts, the checks of a
  release pair — uses those forms. RDKit 2024.9 spells stilbene's bond directions
  otherwise than 2026.9, which wrote v1.3's list, and a comparison of strings refused v1.3
  there. An entry that is empty or holds more than a SMILES, that does not parse, that
  is not in the standard form, that is listed twice, or that is one molecule with
  another entry of its list is refused; where RDKit decides, with the version that
  read it.
- `run.json` records the aromir version that started a run, and only that version takes
  the run up: another one, or a `run.json` without a version, is refused with both
  versions named and what to do (finish with the version that started it, or delete the
  folder).
- No band of a molecule is licensed while the release's `invariance.json` gives the
  molecules of its group — with a linear group (the certificate's linear angle: a nitrile,
  an alkyne) or without — a largest spread above the licence line under a rotation, a
  mirror image or another xyz listing; nor of a molecule with a linear group while that
  spread is not measured (a molecule without one keeps its licence there). The reason is
  stated in the certificate's statement and in every band's licence column, and the
  corrected positions are printed unchanged. v1.4's record gives 11.20 cm⁻¹ with a linear
  group and 2.85 without, so diphenylacetylene and benzonitrile are no longer licensed
  (benzonitrile by its group: its own turns move it less) and the others keep the licence;
  v1.3's gives 5.34 and 3.07, so with `--release` v1.3 licenses no band. A release whose
  record puts a group's spread at or below the line licenses it again. The card states the
  rule with its release's numbers.
- 'Seen' follows one stereo rule, the training project's: a row without stereo labels
  stands for both mirror images of a molecule whose one stereo element is one tetrahedral
  centre (an xyz sulfoxide that names its configuration is in the training set when the
  row leaves it open), and for nothing else; any other stereoisomer of a trained or a
  test-set molecule, and a request that leaves open what a row names, is said to be one,
  in a sentence of its own for a test set, never unseen. Before, a labelled request
  against a stereo-free row was called a stereoisomer, and a stereoisomer of a test-set
  molecule was called unseen.
- A molecule charged only in nitro groups is not enumerated for resonance structures: its
  own structure is the preferred one, and the enumeration found writing-dependent counts
  on both sides of the 256 cap (1,5-dinitronaphthalene 208 to 295 over ten writings;
  1,3,6-trinitropyrene up to 9897). The 719 di- and tri-nitro isomers of twelve PAH
  parents up to 40 atoms now get one name from ten writings each, in milliseconds.
- Readers are told of test sets: the README, the model cards, the certificate and
  `aromir evaluate --help` say "test set (a): unseen molecules" and "test set (b): held-out
  scaffolds" at first use and "test set (a)" / "(b)" after, where they said "hold-out".
  Code, record fields and the `evaluate` folder layout keep the hold-outs; record text
  (a licence basis, the sources) is quoted as recorded. Both shipped cards are written
  again by `aromir card`; `release.json` binds the records and the seeds, not the card.
- The certificate's statement says that the errors are against the higher level, not
  against experiment, and that a measured spectrum is compared with the scaled B3LYP
  positions; it said only that those were a baseline.
- The certificate and the card say what a band height is: computed on the corrected modes
  with B3LYP's dipole derivatives, which are not corrected. They said the heights were
  B3LYP's, but the correction moves them through the modes.

- **The licence is read on the error of what the release delivers** (decided on 16
  November 2026, after both quantities were measured on both hold-outs). That is the
  band error of the seeds' mean: the mean ΔH added to the Hessian, the sum
  diagonalised, each corrected band paired with its reference band. Until now the
  record held, unnamed, the first-order error of each seed averaged over the seeds,
  the number a training run reports. It is the larger number here for two reasons:
  averaging the seeds removes part of each seed's error, and a first-order shift is
  not the shift of the diagonalised Hessian (for 'other' the first-order error of the
  seeds' mean is still 3.00 / 3.02 cm⁻¹, measured outside the package). For v1.3 the
  band error is 1.43 / 0.95 / 1.64 / 2.78 cm⁻¹
  on hold-out (a) and 1.91 / 2.19 / 1.88 / 2.77 on (b) (ring-ip, CH-stretch, CH-oop,
  other), where the record printed 1.77 / 1.19 / 2.17 / 3.62 and 2.12 / 2.50 / 2.70 /
  3.89 from the training runs' own records. All four families are now within the 3 cm⁻¹ line, 'other' included. On
  hold-out (a) 'other' rests on eight molecules: resampling them puts its error
  between 1.68 and 3.76 cm⁻¹, and the card and every certificate say so in their
  first paragraph.
- `evaluation.json` has a second format (`record_version: 2`) that names both
  quantities: `band_error_seed_mean_cm`, with the number of bands and the resampling
  interval per family and hold-out, and `first_order_error_per_seed_cm` with its
  mean. v1.3's record is rewritten in it, every number from `aromir evaluate`; its
  first-order numbers differ from the training runs' by up to 0.08 cm⁻¹ (CH-oop on
  hold-out (b): 2.46 / 2.93 / 2.80 against 2.44 / 2.95 / 2.72). Its weights and its
  network version are unchanged. A record of the first format is refused with the reason: read silently,
  its numbers would be printed under the name of a quantity they are not.
- `aromir evaluate <folder>` measures a release on hold-out molecules with stored
  Hessians (`aromir.evaluation` gives the file format) and writes both quantities:
  the one implementation of the numbers a record of format 2 holds, to two decimals,
  on which the licence is read. On v1.3's two hold-outs its band errors reproduce the
  measurement made outside the package to the second decimal.

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
  (exit 1), or its traceback when `AROMIR_TRACEBACK` is set to anything but 0. An xyz file whose comment
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
  along the mode and optimises once more (tested with the quantum chemistry
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

- Resuming a run reads a step file inside the guard and builds the step outside it: an
  error of the code, such as a field another version wrote, is raised as itself and no
  longer reported as "the folder holds a damaged run; delete it to start over".
- An absolute intensity is pinned: a stretch between two point charges of ±1 e on C and
  O gives 974.88 km/mol / μ = 142.19 km/mol, derived by hand in the test.

- `aromir.dft` said the training Hessians were psi4's at psi4's minima. They are pyscf's
  analytic B3LYP/6-31G* Hessians at grid (99, 590), computed at geometries psi4 optimised
  (density-fitted, grid (75, 302)).
- Phenyl azide read from an xyz file is no longer said to leave a double bond unassigned:
  bond perception gave some listings the quinoid structure `N#[N+]N=c1cccc[cH-]1`, whose
  exocyclic N=C is a potential E/Z bond; the resonance step now gives every listing the
  azide's own structure, and a test holds it in eight listings.
- A molecule's name and atom order no longer depend on atom-map numbers in a SMILES, or
  on whether a group is written hypervalent or charge-separated or in another resonance
  structure; ties between resonance structures are resolved by the canonical SMILES.
  Before, atom-map numbers entered the canonical SMILES, so `[CH3:5]c1ccc(O)cc1Cl` was
  numbered as another molecule than `Cc1ccc(O)cc1Cl` (the network is not order-free:
  bands moved by up to 2.85 cm⁻¹ between orders) and `[cH:1]1ccccc1` was called unseen
  although benzene is in the training data; bond perception from an xyz file placed the
  charges of an azide or a sulfoxide otherwise than the SMILES did. Both routes now read
  the input into one standard form (`admission.read`): map numbers are cleared and the
  certificate and `aromir check` say so; RDKit's Normalizer writes a hypervalent
  sulfoxide or nitro group charge-separated; a molecule that still carries charges takes
  its preferred resonance structure (`[N-]=[N+]=N` for an azide, `[CH2-][n+]1ccccc1` for
  a pyridinium ylide, one writing for both of a sydnone's), keeping its own where it is
  one of the preferred (a nitro group keeps its charge where it was given). No step may
  change the formula, an atom's hydrogens or its aromatic count. A molecule with more
  than 8 charged atoms, or more than 256 resonance structures, keeps the structure it was
  given. The 'seen' look-up matches the standard SMILES and then InChI's fixed-hydrogen
  key, the same for every resonance structure. A release whose training lists hold an
  entry admission would write otherwise is refused; every SMILES the shipped releases
  list is already in that form and no two share a key, so their records and hashes are
  unchanged. A run started by an earlier version from such a request is refused with
  that reason. Not yet the same from an xyz file: a betaine, which bond perception reads
  as its uncharged quinoid form and admission refuses for want of an aromatic ring, and
  nitro compounds, for about a third of whose atom listings RDKit finds no bond orders.
- A molecule with a flat coordinate — a nearly free torsion — gets its spectrum. On
  3-vinylfuran the run found a saddle point at -12 cm⁻¹, stepped 0.1 bohr off it, and
  the second optimisation ended after 100 steps with the energy and the forces
  converged and the displacement not: no spectrum after 67 minutes, and nothing of the
  second attempt kept. Four changes, which hold for every molecule:
  - an optimisation saves where it stands after every gradient
    (`optimisation_progress.json`), and the same command takes it up there;
  - one that does not converge on energy, forces and displacement in 100 steps goes on
    for at most 20 under geomeTRIC's same test with the same thresholds for the energy
    change and the rms and largest force, and the two displacement thresholds set so
    that any step meets them (geomeTRIC's own `convergence_drms` / `convergence_dmax`).
    The certificate then says 'converged on energy and forces; the displacement
    criterion was not met on a flat coordinate', with the last step's numbers, when
    one step without the criterion was enough; when more were needed it says how many
    and does not call the coordinate flat;
  - the step off a saddle point is sized from the imaginary mode's measured curvature,
    to lower the energy by 1e-5 hartree in the harmonic picture, and moves no atom by
    more than 0.3 bohr; it was 0.1 bohr for every saddle point. Both numbers are
    chosen;
  - a geometry that after that one step and a second optimisation still has an
    imaginary mode softer than 20 cm⁻¹ is printed, with that band named a flat
    coordinate, never licensed and named in the statement. An imaginary mode beyond
    20 cm⁻¹ still ends the run as not a minimum. `aromir spectrum` says on its last line
    how many bands lie on a flat coordinate, and `baseline.json` marks each such band.
- The tests of the atom order hold what it promises: every listing of benzene gets one
  bond graph position by position, and every listing of a molecule without symmetry
  one geometry exactly (a walk of any ring passed the earlier test).
- The dependencies have upper bounds, the series the package is tested with; the
  internal coordinates of seven molecules are pinned atom by atom (a digest of
  geomeTRIC's own description of every primitive), so a new RDKit or geomeTRIC that
  numbers or chooses differently for one of them fails a test.
- pyscf is held at 2.15.0: 2.15.1, released on 17 November 2026, passes the tests on
  Linux but has no wheel for macOS on Apple silicon, and its source distribution lacks
  `pyscf/lib/libxcfun.patch`, so pip's build of it fails; an install that admitted it
  failed on such a Mac.
- A step file that does not name the run that made it, a `lowlevel.npz` whose arrays
  have another molecule's shape, whose record contradicts itself or that does not
  decompress: each is a damaged run, said in words before anything is computed on it.
- Three tests that are expected to fail, strictly, state what release v1.3 does not do
  yet: a turned nitrile, a mirror image and another listing of one xyz geometry each
  get another spectrum. The day a release answers alike, they pass and must be unmarked.
- One molecule of hold-out (a), benzene, is evaluated in the tests: the family rule, the
  network and the evaluation are tied to the numbers the record was made from.
- A release whose evaluation record gives the first-order error of another number of
  seeds than the release holds is refused: the record was measured on another set.
- Release v1.3 ships `measured.json`, what `aromir evaluate` wrote on its two hold-outs,
  and a test holds `evaluation.json`'s tables to it: the record's numbers can be followed
  to their measurement inside the package.
- Whether a molecule is in the training set is decided by its canonical SMILES; a test
  holds every SMILES of v1.3's lists to RDKit's present canonical form, so an RDKit that
  writes one of them differently fails a test instead of answering 'not seen'.
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

### Removed

- The Lorentzian broadening (`spectrum.broadened`, `grid`, `FWHM`, `GRID`), which no
  command used, with its tests.
