# Pins of the bundled release

- `evaluation_v1.3.json`: the evaluation record of network v1.3.
- `delta_h_biphenyl_cn_v1.3.npz`: 4-cyanobiphenyl at its B3LYP/6-31G* geometry (`symbols`,
  `coords_bohr`), its B3LYP Hessian with the rigid motions projected out (`hessian`,
  hartree/bohr²) and the three seeds' ΔH of release v1.3 for it (`delta_h_windows`,
  largest element 0.040), computed on Windows with numpy 2.5.1 and torch 2.14.0. On
  numpy 2.5.4 for Linux the same input gave 1e29 until the pseudo-inverse of the Wilson B
  matrix took its rank from the molecule. Written by
  `probes/results_m1/aromir_pinv_2026-11-16/make_release_fixture.py` of the capstone
  repository.
