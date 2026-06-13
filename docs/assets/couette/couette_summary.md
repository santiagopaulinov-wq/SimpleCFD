# Plane Couette Benchmark

## Problem

- Channel height: 1.0
- Lower wall velocity: 0.0
- Upper wall velocity: 1.0
- Dynamic viscosity: 1.0
- Nodes: 33

## Validation

- Wall shear stress: 1
- L1 profile error: 1.26151193868e-15
- L2 profile error: 1.45833737519e-15
- Linf profile error: 2.33146835171e-15
- Analytic flow rate per unit width: 0.5
- Numerical flow rate per unit width: 0.5
- Analytic kinetic energy per unit width: 0.333333333333
- Numerical kinetic energy per unit width: 0.33349609375
- Kinetic-energy relative error: 0.000488281249996

## Refinement

| n_nodes | dy | l1_error | l2_error | linf_error | observed_l2_error_order | kinetic_energy_relative_error | observed_energy_order |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.125 | 5.24272e-17 | 6.985e-17 | 1.11022e-16 |  | 0.0078125 |  |
| 17 | 0.0625 | 1.38778e-17 | 2.54117e-17 | 5.55112e-17 |  | 0.00195313 | 2 |
| 33 | 0.03125 | 1.26151e-15 | 1.45834e-15 | 2.33147e-15 |  | 0.000488281 | 2 |
| 65 | 0.015625 | 1.90406e-15 | 2.26702e-15 | 3.60822e-15 |  | 0.00012207 | 2 |

## Artifacts

- Profile CSV: `couette_profile.csv`
- Convergence CSV: `couette_convergence.csv`
- Profile figure: `couette_profile.png`
- Convergence figure: `couette_energy_convergence.png`
- Profile error figure: `couette_profile_error_convergence.png`
- Profile observed-order figure: `couette_profile_observed_orders.png`
