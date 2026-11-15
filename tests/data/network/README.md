# Regression pins of the network

These files hold the output of the code the network's weights are trained with (the
capstone repository's inference path), so that aromir's rewrite is held to the same
numbers without importing that repository.

Written by `probes/aromir_network_golden.py` of `thebreadishard/udacity-capstone-plan`
(plan 05) with torch 2.15.0, geomeTRIC 1.1.1 and RDKit 2026.9.2.

- `<configuration>.safetensors` and `.json`: a seeded random network under the trainer's
  parameter names, with the trainer's constructor record. Every parameter and buffer is
  drawn, so no zero-initialised row hides a wiring error.
- `<configuration>_<molecule>.npz`: the molecule (an RDKit geometry), a seeded random
  symmetric matrix standing in for the low-level Hessian, the trainer's ΔH for it, and the
  integer inputs (pattern pairs, pair classes, hinge classes, parities) that say where a
  mismatch starts.

The three configurations: `carried_shape` (the carried recipe's switches: pattern f, pair
features, low-level share, sum pooling), `order_free` (canonical angle choice, cos-encoded
dihedrals) and `parity_odd` (the parity-odd factor, mean pooling, pattern d).
