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
- **Networks:** The spectrum is made by the release networks of version 1.4, trained on the 1132 labelled entries of their training pool, the hold-out molecules included (1078 distinct neutral molecules by SMILES and the radical cations of 50 molecules): each of the 3 networks is fitted on a random 85 % of the entries and stopped on the other 15 %, a different 15 % per seed, so not every entry is fitted by every network (with 3 networks about 0.15³ ≈ 0.3 % by none). Its error bars were measured on the measurement networks of the same version — the same recipe trained without the hold-out molecules — on molecules those networks never saw; the release networks themselves have seen the hold-out molecules, so they have no held-out error of their own.
- **Licensed families:** ring-ip, CH-stretch, CH-oop, other.
- **Not licensed:** none.
- **On thin ground:** 'CH-oop' is within the licence line on hold-out (a) at 1.88 cm⁻¹, on 8 molecules: resampling those molecules puts the error between 0.93 and 3.24 cm⁻¹, and 4% of the resamples lie above the line.
- **On thin ground:** 'other' is within the licence line on hold-out (a) at 2.38 cm⁻¹, on 8 molecules: resampling those molecules puts the error between 1.45 and 3.19 cm⁻¹, and 7% of the resamples lie above the line.
- **Use it through:** [aromir](https://github.com/thebreadishard/aromir), which builds the inputs, applies the correction and writes a certificate for every band.

## How well it is known

The rms error of the corrected band positions of the seeds' mean against ωB97X/6-31G*, in cm⁻¹, on molecules the measurement networks never trained on — the same recipe trained without the hold-outs, whose error is the error bar of what aromir prints; the release networks aromir runs trained on these molecules too. In square brackets the 95 % interval when the hold-out's molecules are resampled. The licence is read on the band error of the seeds' mean, the error of what the release delivers, as recorded to two decimals; chosen on 16 November 2026, after it and the first-order error per seed had both been measured on both hold-outs.

| family | (a), 8 molecules | (b), 40 molecules | licensed |
|---|---|---|---|
| ring-ip | 1.17 [0.90–1.41], 199 bands | 1.22 [1.09–1.34], 1194 bands | yes |
| CH-stretch | 0.80 [0.44–1.20], 73 bands | 1.67 [1.37–1.95], 410 bands | yes |
| CH-oop | 1.88 [0.93–3.24], 46 bands | 1.26 [0.98–1.59], 84 bands | yes |
| other | 2.38 [1.45–3.19], 156 bands | 2.10 [1.88–2.34], 1258 bands | yes |

- (a): eight of the ten molecules set aside on 19 September 2026, before any network was trained; the other two have a twin in another layer of the training pool and are left out of the read (30 October 2026)
- (b): every substituted molecule of two ring scaffolds, set aside on 8 October 2026; no measurement network trained on a molecule of either scaffold

For comparison, the first-order error of each seed alone, the number a training run reports and the one earlier versions of this card printed as the error: its mean over the seeds, each seed's value in brackets. It is not the error of what aromir prints.

| family | (a) | (b) |
|---|---|---|
| ring-ip | 1.55 (1.77, 1.52, 1.37) | 1.64 (1.53, 1.49, 1.89) |
| CH-stretch | 1.15 (1.34, 1.19, 0.92) | 2.02 (2.32, 2.20, 1.55) |
| CH-oop | 2.24 (2.27, 2.10, 2.36) | 1.72 (1.80, 1.95, 1.42) |
| other | 2.91 (2.79, 3.10, 2.84) | 3.68 (3.20, 4.15, 3.70) |

A family is licensed when the band error of the seeds' mean, as recorded to two decimals, is within 3 cm⁻¹ on every hold-out. aromir's certificate marks each band with its family, that family's error and whether the band itself is licensed, which asks more: see the limits below.

Coupled cluster: not measured for this network version.

Sources: every error in this record: `aromir evaluate` with the measurement networks (E7_rungC_v14s3clean_c00_control_model_n1082_seed0.pt, E7_rungC_v14s3clean_c00_control_model_n1082_seed1.pt, E7_rungC_v14s3clean_c00_control_model_n1082_seed2.pt) on the two hold-outs as probes/results_m1/aromir_audit_fixes_2026-11-16/holdouts_for_aromir.py of thebreadishard/udacity-capstone-plan (plan 05) writes them; counterpart, within 0.01 cm⁻¹ in every cell: probes/results_m1/aromir_v14_release_2026-11-18/product_error.json (aromir's code on the stored Hessians, in aromir's atom order); the capstone's read in the corpus's atom order is modules/05_support_predictor/out/v14/E7_rungC_v14s3clean_c00_control_product_error.json.

## What it was trained on

- 1132 labelled entries of the training pool, each a closed-shell molecule or radical cation with an aromatic ring: 1078 distinct neutral molecules by SMILES and the radical cations of 50 molecules; the measurement networks, on which every error is read, trained on 1028 of those neutral molecules.
- Elements: C, Cl, F, H, N, O, S.
- Sizes: 8 to 40 atoms.
- Held out from the measurement networks, on which every error is read: 8 in (a); 40 in (b); the release networks trained on these molecules.
- The training Hessians were used as pyscf's analytic Hessians, without the soft-mode repair; aromir gives the network a molecule's Hessian prepared the same way. It measures the soft modes in either case, to decide whether the geometry is a minimum and for the positions it prints: the correction is added to the Hessian with its soft modes measured.

## Where it stops being known

- **The level is ωB97X/6-31G*.** The errors above are against that level, not against experiment.
- **Positions are harmonic.** No anharmonic shift is applied.
- **Neutral, closed-shell, aromatic molecules only**, of the elements C, Cl, F, H, N, O, S. aromir refuses anything else before computing.
- **Sizes outside 8 to 40 atoms are extrapolation.** `aromir spectrum` refuses them unless `--outside-training-sizes` is given; its certificate then licenses none of their bands.
- **Not licensed: no family.** Bands of an unlicensed family are printed with their measured error and marked.
- **The answer depends on how the molecule's atoms are numbered**: the internal coordinates are chosen by atom number, and a dihedral near ±180° enters by its raw value. aromir orders the atoms of every input by the bond graph first, so every SMILES of a molecule gets one order; atoms the graph cannot tell apart keep the order of an xyz file, so two listings of one geometry can still differ. Do not feed the network a molecule in another order. The spread between atom orders was not measured for this release.
- **The same molecule turned or mirrored** may get another answer; by how much was not measured for this release.
- **Band heights are the cheap level's.** The dipole derivatives are not corrected.
- **A family is a rule, not a kind of bond.** ring-ip: between 950 and 1750 cm⁻¹ and in the molecular plane: ring stretches and in-plane bends, and with them every other mode there, such as a C=O stretch or a CH₃ deformation; in a non-planar molecule every mode in that window. CH-stretch: above 2800 cm⁻¹ with most of its motion on hydrogens: the C–H stretches, and by the same rule an O–H or N–H stretch. CH-oop: between 600 and 1050 cm⁻¹, out of the molecular plane and on hydrogens; a non-planar molecule has none, and its C–H wags fall under other or ring-ip. other: every mode the three rules above do not take. aromir's certificate licenses a band only when its family is within the licence line, the molecule lies inside the training sizes, its low-level mode is real, the band is real for the seeds' mean and for every seed, and a CH-stretch or CH-oop band moves carbon-bound hydrogens. A ring-ip band is licensed as the rule takes it, C=O stretches included.

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

Each `seed-<n>.json` beside its tensors holds the recipe and that sha256; `evaluation.json` and `training.json` hold the records this card is written from.

## Use

```console
$ pip install git+https://github.com/thebreadishard/aromir
$ aromir spectrum "c1ccc2ccccc2c1" --out results/naphthalene
```

aromir uses the release inside the package; `--release <folder>` names another, such as a copy of this one.

## Licence

Weights and records: CC BY 4.0. Code: MIT, at https://github.com/thebreadishard/aromir.

This card was written by `aromir card` from release v1.4's own files.
