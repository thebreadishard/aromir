---
license: cc-by-4.0
library_name: aromir
tags:
  - chemistry
  - infrared-spectroscopy
  - hessian
  - equivariant
model_name: aromir ΔH network v1.4
---

# aromir ΔH network v1.4

A correction to the Hessian of an aromatic molecule. The network reads a cheap B3LYP/6-31G* Hessian and returns ΔH, the difference to ωB97X/6-31G* (the DFT proxy: the network's ΔH is trained on ωB97X − B3LYP Hessians). The corrected Hessian gives harmonic infrared band positions closer to that level at the cost of the cheap one.

- **Seeds:** 3 (0, 1, 2); the applied correction is their mean, and their spread is reported per band.
- **Networks:** The spectrum is made by the release networks of version 1.4, trained on the 1132 labelled entries of their training pool, the test sets' molecules included (1078 distinct neutral molecules by SMILES and the radical cations of 50 molecules): each of the 3 networks is fitted on a random 85 % of the entries and stopped on the other 15 %, a different 15 % per seed, so not every entry is fitted by every network (with 3 networks about 0.15³ ≈ 0.3 % by none). Its error bars were measured on the measurement networks of the same version — the same recipe trained without the test sets' molecules — on molecules those networks never saw; the release networks themselves have seen the test sets' molecules, so they have no out-of-sample error of their own.
- **Licensed families:** ring-ip, CH-stretch, CH-oop, other (within 3 cm⁻¹ on test set (a): unseen molecules and on test set (b): held-out scaffolds).
- **Not licensed:** none.
- **On thin ground:** 'CH-oop' is within the licence line on test set (a) at 1.88 cm⁻¹, on 5 of the test set's 8 molecules (the others have no 'CH-oop' band): resampling the test set's molecules puts the error between 0.93 and 3.24 cm⁻¹, and 4% of the resamples lie above the line.
- **On thin ground:** 'other' is within the licence line on test set (a) at 2.38 cm⁻¹, on 8 molecules: resampling the test set's molecules puts the error between 1.45 and 3.19 cm⁻¹, and 7% of the resamples lie above the line.
- **Use it through:** [aromir](https://github.com/thebreadishard/aromir), which builds the inputs, applies the correction and writes a certificate for every band.

## How well it is known

The rms error of the corrected band positions of the seeds' mean against ωB97X/6-31G*, in cm⁻¹, on molecules the measurement networks never trained on — the same recipe trained without the test sets, whose error is the error bar of what aromir prints; the release networks aromir runs trained on these molecules too. In square brackets the 95 % interval when the test set's molecules are resampled. The licence is read on the band error of the measurement networks' seed mean, the error bar of what the release networks deliver, as recorded to two decimals; chosen on 16 November 2026, after it and the first-order error per seed had both been measured on both hold-outs.

| family | test set (a), 8 molecules | test set (b), 40 molecules | licensed |
|---|---|---|---|
| ring-ip | 1.17 [0.90–1.41], 199 bands | 1.22 [1.09–1.34], 1194 bands | yes |
| CH-stretch | 0.80 [0.44–1.20], 73 bands | 1.67 [1.37–1.95], 410 bands | yes |
| CH-oop | 1.88 [0.93–3.24], 46 bands | 1.26 [0.98–1.59], 84 bands | yes |
| other | 2.38 [1.45–3.19], 156 bands | 2.10 [1.88–2.34], 1258 bands | yes |

- test set (a): unseen molecules — eight of the ten molecules set aside on 19 September 2026, before any network was trained; the licence and the promotion are decided on these eight, the set of 30 October on which v1.3 and v1.4 are both clean. For v1.4 the other two, benzonitrile and 2-naphthoic acid, are clean as well: on all ten its measurement networks read ring-ip 1.10 [0.82–1.34], CH-stretch 0.78 [0.47–1.15], CH-oop 1.75 [0.91–2.83], other 2.23 [1.34–3.02] cm⁻¹, which decides nothing
- test set (b): held-out scaffolds — the 40 substituted molecules of two ring scaffolds, fluorene (21) and fluoranthene (19), set aside on 8 October 2026, that have analytic Hessians at both levels and a minimum at the low level: of the 41 on that list, 9-nitrofluorene has an imaginary low-level mode, and 30 more fluoranthene children (set b+) have semi-numerical Hessians only and are not read here. No measurement network trained on a molecule containing fluorene's ring system; five that contain fluoranthene's, benzo[k]fluoranthene and four of its derivatives, were trained on

For comparison, the first-order error of each seed alone, the number a training run reports and the one earlier versions of this card printed as the error: its mean over the seeds, each seed's value in brackets. It is not the error of what aromir prints.

| family | test set (a) | test set (b) |
|---|---|---|
| ring-ip | 1.55 (1.77, 1.52, 1.37) | 1.64 (1.53, 1.49, 1.89) |
| CH-stretch | 1.15 (1.34, 1.19, 0.92) | 2.02 (2.32, 2.20, 1.55) |
| CH-oop | 2.24 (2.27, 2.10, 2.36) | 1.72 (1.80, 1.95, 1.42) |
| other | 2.91 (2.79, 3.10, 2.84) | 3.68 (3.20, 4.15, 3.70) |

A family is licensed when the band error of the seeds' mean, as recorded to two decimals, is within 3 cm⁻¹ on every test set. aromir's certificate marks each band with its family, that family's error and whether the band itself is licensed, which asks more: see the limits below.

Coupled cluster: not measured for this network version.

Sources: every error in this record: `aromir evaluate` with the measurement networks (E7_rungC_v14s3clean_c00_control_model_n1082_seed0.pt, E7_rungC_v14s3clean_c00_control_model_n1082_seed1.pt, E7_rungC_v14s3clean_c00_control_model_n1082_seed2.pt) on the two hold-outs as probes/results_m1/aromir_audit_fixes_2026-11-16/holdouts_for_aromir.py of thebreadishard/udacity-capstone-plan (plan 05) writes them; counterpart, within 0.01 cm⁻¹ in every cell: probes/results_m1/aromir_v14_release_2026-11-18/product_error.json (aromir's code on the stored Hessians, in aromir's atom order); the capstone's read in the corpus's atom order is modules/05_support_predictor/out/v14/E7_rungC_v14s3clean_c00_control_product_error.json; the molecules with a band of each family: the same `aromir evaluate` (format 3); counterpart, equal in every cell: probes/results_m1/aromir_266_counts_2026-11-18/count_bands.py of thebreadishard/udacity-capstone-plan (plan 05), the molecules with a low-level mode of the family; hold-out (b)'s description: modules/05_support_predictor/corpus/holdout_b_frozen_2026-10-08.txt (41 molecules) and probes/results_m1/aromir_audit_answers_2026-11-18/scaffolds.log and murcko.log of thebreadishard/udacity-capstone-plan (plan 05); 9-nitrofluorene's imaginary mode and set b+: probes/results_m1/aromir_audit_v14_2026-11-18/AUDIT.md, K2; hold-out (a) on ten, which decides nothing: probes/results_m1/aromir_audit_answers_2026-11-18/a10_measured.json (read_a10.sh: `aromir evaluate` with the measurement networks on set 'a' of E7_rungC_v14s3clean_c00_control_eval_clean_seed0.json) of thebreadishard/udacity-capstone-plan (plan 05).

## What it was trained on

- 1132 labelled entries of the training pool, each a neutral closed-shell molecule or the radical cation of one, with an aromatic ring: 1078 distinct neutral molecules by SMILES and the radical cations of 50 molecules (the 1082 neutral entries hold 1078 distinct molecules: a molecule may be an entry under more than one identifier). The measurement networks, on which every error is read, trained on 1028 of those neutral molecules; the release networks on all of the entries, the test sets' molecules included.
- Elements: C, Cl, F, H, N, O, S.
- Sizes: 8 to 40 atoms.
- Held out from the measurement networks, on which every error is read: 8 in test set (a); 40 in test set (b); the release networks trained on these molecules.
- The training Hessians were used as pyscf's analytic Hessians, without the soft-mode repair; aromir gives the network a molecule's Hessian prepared the same way. It measures the soft modes in either case, to decide whether the geometry is a minimum and for the positions it prints: the correction is added to the Hessian with its soft modes measured.

## Where it stops being known

- **The level is ωB97X/6-31G*.** The errors above are against that level, not against experiment.
- **Positions are harmonic.** No anharmonic shift is applied.
- **Neutral, closed-shell, aromatic molecules only**, of the elements C, Cl, F, H, N, O, S. aromir refuses anything else before computing.
- **Sizes outside 8 to 40 atoms are extrapolation.** `aromir spectrum` refuses them unless `--outside-training-sizes` is given; its certificate then licenses none of their bands.
- **Not licensed: no family.** Bands of an unlicensed family are printed with their measured error and marked.
- **The answer depends on how the molecule's atoms are numbered**: the internal coordinates are chosen by atom number, and a dihedral near ±180° enters by its raw value. aromir orders the atoms of every input by the bond graph first, so every SMILES of a molecule gets one order; atoms the graph cannot tell apart keep the order of an xyz file, so two listings of one geometry can still differ. Do not feed the network a molecule in another order. The spread between arbitrary atom orders (the evaluation's atom-order spread) was not measured for this release; the spread between xyz listings put into aromir's canonical order is below.
- **A rigid rotation of the molecule** moved band positions by up to 11.20 cm⁻¹ on 3 molecule(s) with a linear group and 0.00 cm⁻¹ on 4 molecule(s) without a linear group (8 trial(s) per molecule). These are the largest seen, not bounds.
- **The molecule's mirror image** moved band positions by up to 4.46 cm⁻¹ on 3 molecule(s) with a linear group and 2.67 cm⁻¹ on 4 molecule(s) without a linear group (1 trial(s) per molecule). These are the largest seen, not bounds.
- **Another listing of the same atoms in an xyz file** moved band positions by up to 11.03 cm⁻¹ on 3 molecule(s) with a linear group and 2.85 cm⁻¹ on 4 molecule(s) without a linear group (12 trial(s) per molecule). These are the largest seen, not bounds.
  A linear angle (in a nitrile, an alkyne) is measured against a fixed axis in space, and a dihedral enters with its sign; the test sets' errors hold none of this. Source: scripts/measure_invariance.py of aromir at 7dc0e12b28074eed26a598c166ff4c86782b33f6, 2026-11-19, on the release networks v1.4 ships (tensor sha256: seed-0 4e82e0f99715bb99ef050410410869e2f68cdbcf57f43398cd0d7febee655a14, seed-1 14059360ed350c1bfaa2ca51bb852fa22eccb6e2644226b2250ccb6652a831aa, seed-2 c6451ca246977bd704392d45911233b8d7ea581bfc33268212b54bc504400b0c): benzonitrile, phenylacetylene and diphenylacetylene with a linear group; benzene, biphenyl, 2-methylnaphthalene and phenol without; stored B3LYP geometries and Hessians of thebreadishard/capstone-data (modules/05_support_predictor/corpus/molecules), the trials of v1.3's record.
- **A molecule's licence follows its group's measured spread.** aromir's certificate licenses no band of a molecule while the largest spread this release's invariance record gives for molecules of its group (with or without a linear group, an angle near 180° as in a nitrile or an alkyne) under a rotation, a mirror image or another xyz listing of the atoms exceeds 3 cm⁻¹, nor of a molecule with a linear group while that spread was not measured; for this release it is 11.20 cm⁻¹ with a linear group, so their bands are not licensed and 2.85 cm⁻¹ without a linear group, so their bands keep the licence; the corrected positions are printed either way.
- **Band heights have no error bar.** They are computed on the corrected modes with the cheap level's dipole derivatives, which are not corrected.
- **A family is a rule, not a kind of bond.** ring-ip: between 950 and 1750 cm⁻¹ and in the molecular plane: ring stretches and in-plane bends, and with them every other mode there, such as a C=O stretch or a CH₃ deformation; in a non-planar molecule every mode in that window. CH-stretch: above 2800 cm⁻¹ with most of its motion on hydrogens: the C–H stretches, and by the same rule an O–H or N–H stretch. CH-oop: between 600 and 1050 cm⁻¹, out of the molecular plane and on hydrogens; a non-planar molecule has none, and its C–H wags fall under other or ring-ip. other: every mode the three rules above do not take. aromir's certificate licenses a band only when its family is within the licence line, the molecule lies inside the training sizes and holds no charge separation or aromatic ring size the training molecules lack, its low-level mode is real, the band is real for the seeds' mean and for every seed, molecules of the molecule's group (with or without a linear group) moved by no more than the licence line in the release's invariance record, and a CH-stretch or CH-oop band moves carbon-bound hydrogens. A ring-ip band is licensed as the rule takes it, C=O stretches included.

## What it is

An equivariant message-passing encoder (PaiNN type) over the atoms, and a head that predicts one number per pair of internal coordinates: an entry of ΔF. The Cartesian correction is ΔH = Bᵀ ΔF B with B the Wilson matrix, so it is symmetric and free of translations. It is free of rotations and turns with the molecule only where no linear angle is among the internal coordinates: see the limits above.

- Encoder: 3 blocks, 64 scalar and 64 vector channels, sum pooling.
- Head: hidden width 256, with the pair features, a learned share of the low level's own force constant.
- Pair pattern f; primitives by the geometric angle choice, dihedrals raw.
- 292,546 parameters per seed.

## Files

| file | seed | sha256 |
|---|---|---|
| seed-0.safetensors | 0 | `4e82e0f99715bb99ef050410410869e2f68cdbcf57f43398cd0d7febee655a14` |
| seed-1.safetensors | 1 | `14059360ed350c1bfaa2ca51bb852fa22eccb6e2644226b2250ccb6652a831aa` |
| seed-2.safetensors | 2 | `c6451ca246977bd704392d45911233b8d7ea581bfc33268212b54bc504400b0c` |

Each `seed-<n>.json` beside its tensors holds the recipe and that sha256; `evaluation.json`, `training.json`, `invariance.json`, `release.json`, `measurement.json` hold the records this card is written from, and `measured.json`, where the folder has it, is the output of `aromir evaluate` the error tables are copied from: the loader refuses tables that differ from it. `release.json` binds `evaluation.json`, `training.json`, `measurement.json` and `measured.json` by the SHA-256 of their content. `invariance.json` names the seeds' tensor files it was measured on by their SHA-256, and the loader refuses it beside other weights.

## Use

```console
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install git+https://github.com/thebreadishard/aromir
$ aromir spectrum "c1ccc2ccccc2c1" --out results/naphthalene
```

aromir uses the release inside the package; `--release <folder>` names another, such as a copy of this one.

## Licence

Weights and records: CC BY 4.0. Code: MIT, at https://github.com/thebreadishard/aromir.

This card was written by `aromir card` from release v1.4's own files.
