# Verification

SimpleCFD verification is based on two complementary layers:

1. Golden and invariant tests for the 1D SIMPLE-family solver.
2. Analytic benchmarks with exact solutions and mesh-refinement artifacts.

The project currently focuses on 1D verification only.

## Registered Solver Benchmark

The `versteeg_6_2` registered case declares a golden numerical benchmark in
`simplecfd.cases`. It checks final pressure, velocity, mass flow, residuals,
and iteration count for the SIMPLE/upwind configuration.

The broader solver test suite also checks:

- internal momentum coefficients against known values;
- inlet and outlet momentum boundary coefficients;
- pressure-correction coefficients and tridiagonal systems;
- TDMA behavior;
- mass conservation after pressure correction;
- constant-area analytic behavior;
- residual and stability metrics.

## Poiseuille Benchmark

`simplecfd.poiseuille` implements steady plane Poiseuille flow between fixed
parallel plates:

```text
u(y) = (Delta p / L) y (H - y) / (2 mu)
```

The numerical system is a second-order finite-difference discretization of the
1D transverse momentum balance. It is solved with the package TDMA solver.

Verified quantities include:

- no-slip boundary values;
- numerical profile against analytic profile;
- centerline velocity;
- flow rate per unit width;
- profile error norms `L1`, `L2`, and `Linf`;
- flow-rate convergence under mesh refinement.

Because the discrete operator exactly recovers the quadratic profile at grid
nodes for this problem, nodal profile errors are at roundoff. Observed profile
orders are intentionally left undefined when the errors are at the roundoff
floor. Flow-rate error is still meaningful because it is computed by trapezoid
integration of the numerical profile and shows second-order convergence.

## Couette Benchmark

`simplecfd.couette` implements steady plane Couette flow:

```text
u(y) = U_lower + (U_upper - U_lower) y / H
```

Verified quantities include:

- lower and upper no-slip moving-wall values;
- numerical profile against analytic profile;
- wall shear stress;
- flow rate per unit width;
- kinetic energy per unit width;
- profile error norms `L1`, `L2`, and `Linf`;
- kinetic-energy convergence under mesh refinement.

The linear profile is recovered at grid nodes to roundoff. As with Poiseuille,
profile observed orders are not inferred from roundoff-level errors.
Integrated kinetic energy provides a meaningful second-order refinement metric.

## Aggregate Analytic Verification

`simplecfd.verification.generate_analytic_verification_report(output_dir)` runs
all built-in analytic benchmarks and writes:

- `analytic_verification_summary.md`
- `analytic_verification_summary.csv`
- per-benchmark artifact directories

The CLI command is:

```bash
python -m simplecfd verify-analytic --output-dir outputs/analytic_verification
```

The summary table reports:

- benchmark name;
- grid size;
- `L1`, `L2`, and `Linf` profile errors;
- selected integral metric;
- integral relative error;
- finest observed integral order;
- whether profile order is measured or not applicable due to roundoff-level
  profile error.

## Artifact Types

Analytic benchmarks write:

- profile CSV with numerical, analytic, and pointwise error values;
- convergence CSV with error norms and observed-order columns;
- Markdown summary;
- profile PNG;
- integral convergence PNG;
- profile-error convergence PNG;
- profile observed-order PNG.

The figures are evidence artifacts, not interactive visualization tools.

## Interpretation

The current verification evidence supports:

- correct 1D tridiagonal solves;
- correct implementation of analytic 1D diffusion benchmark equations;
- second-order behavior for integrated quantities in Poiseuille and Couette;
- stable regression behavior for the Versteeg 6.2 SIMPLE/upwind case;
- internal consistency of 1D momentum and pressure-correction assembly.

It does not verify multidimensional CFD behavior, turbulence models, transient
schemes, or general-purpose Navier-Stokes solvers.
