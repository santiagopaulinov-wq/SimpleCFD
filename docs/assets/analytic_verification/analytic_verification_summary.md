# SimpleCFD Analytic Verification

## Summary

| benchmark | n_nodes | l1_error | l2_error | linf_error | integral_metric | integral_relative_error | finest_observed_integral_order | profile_order_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| poiseuille | 33 | 1.60646e-15 | 1.88916e-15 | 3.10862e-15 | flow_rate | 0.000976563 | 2 | not_applicable_roundoff_profile_error |
| couette | 33 | 1.26151e-15 | 1.45834e-15 | 2.33147e-15 | kinetic_energy | 0.000488281 | 2 | not_applicable_roundoff_profile_error |

## Artifacts

- Summary CSV: `analytic_verification_summary.csv`
- Poiseuille artifacts: `poiseuille/`
- Couette artifacts: `couette/`

Profile order is reported as not applicable when nodal profile errors are at the roundoff floor.
