# Plane Poiseuille Benchmark

## Problem

- Channel height: 1.0
- Length: 1.0
- Dynamic viscosity: 1.0
- Pressure drop: 8.0
- Nodes: 33

## Validation

- Analytic centerline velocity: 1
- Numerical centerline velocity: 1
- L1 profile error: 1.60645907351e-15
- L2 profile error: 1.88915631942e-15
- Linf profile error: 3.10862446895e-15
- Analytic flow rate per unit width: 0.666666666667
- Numerical flow rate per unit width: 0.666015625
- Flow-rate relative error: 0.000976562500002

## Refinement

| n_nodes | dy | l1_error | l2_error | linf_error | observed_l2_error_order | flow_rate_relative_error | observed_flow_order |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.125 | 2.46716e-17 | 5.23364e-17 | 1.11022e-16 |  | 0.015625 |  |
| 17 | 0.0625 | 3.10209e-17 | 5.59178e-17 | 1.11022e-16 |  | 0.00390625 | 2 |
| 33 | 0.03125 | 1.60646e-15 | 1.88916e-15 | 3.10862e-15 |  | 0.000976563 | 2 |
| 65 | 0.015625 | 5.28135e-15 | 6.13854e-15 | 9.54792e-15 |  | 0.000244141 | 2 |

## Artifacts

- Profile CSV: `poiseuille_profile.csv`
- Convergence CSV: `poiseuille_convergence.csv`
- Profile figure: `poiseuille_profile.png`
- Convergence figure: `poiseuille_flow_convergence.png`
- Profile error figure: `poiseuille_profile_error_convergence.png`
- Profile observed-order figure: `poiseuille_profile_observed_orders.png`
