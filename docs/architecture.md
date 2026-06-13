# SimpleCFD Architecture

SimpleCFD is a Python package for one-dimensional finite-volume pressure-
velocity coupling experiments. The core solver is a 1D staggered-grid solver
for nozzle/channel-style problems. Analytic verification benchmarks for plane
Poiseuille and Couette flow are implemented as separate 1D transverse
diffusion benchmarks that reuse the package linear-system and artifact
generation utilities.

The package does not contain a 2D flow solver, unstructured meshes, turbulence
models, compressible-flow models, or transient integration.

## Package Layout

- `simplecfd.geometry`: 1D staggered mesh geometry. Pressure nodes and
  staggered velocity nodes are represented by separate arrays.
- `simplecfd.fields`: 1D pressure, velocity, and pressure-correction fields.
- `simplecfd.coefficients`: coefficient containers for momentum,
  pressure-correction, and tridiagonal linear systems.
- `simplecfd.linalg`: Thomas algorithm (`tdma`) for tridiagonal systems.
- `simplecfd.schemes`: upwind and central-difference convection policies.
- `simplecfd.boundary`: inlet stagnation-pressure and outlet fixed-pressure
  boundary contributions for the 1D momentum equation.
- `simplecfd.assembly`: momentum, pressure-correction, and SIMPLER
  absolute-pressure assemblers.
- `simplecfd.simple_loop`: SIMPLE, SIMPLEC, and SIMPLER iteration strategies.
- `simplecfd.solver`: global nonlinear iteration controller and residual
  reporting.
- `simplecfd.cases`: declarative problem definitions, case registry, and
  builders that wire runnable solver cases.
- `simplecfd.comparison`: batch comparison of registered cases, schemes, and
  coupling methods.
- `simplecfd.reports`: Markdown, CSV, and PNG method-comparison artifacts.
- `simplecfd.mesh_refinement`: 1D mesh-refinement studies for registered cases.
- `simplecfd.telemetry`: per-iteration pressure, velocity, and residual
  histories.
- `simplecfd.poiseuille` and `simplecfd.couette`: analytic 1D benchmark
  problems with CSV, Markdown, and PNG artifacts.
- `simplecfd.verification`: aggregate analytic-verification report across the
  built-in analytic benchmarks.
- `simplecfd.cli`: command line entry point used by `simplecfd` and
  `python -m simplecfd`.

## Core Data Model

The primary solver path is built around `ProblemDefinition` in
`simplecfd.cases`. A problem definition contains:

- `Geometry`: pressure-node areas, staggered velocity-node areas, and scalar or
  array spacing `dx`.
- `Field`: pressure `p`, staggered velocity `u`, and pressure correction
  `p_prime`.
- `FlowProperties`: currently density only.
- `BoundaryConditions`: optional inlet and outlet boundary objects.
- `ConvectionScheme`: `Upwind` or `CentralDifference`.
- `SolverControls`: tolerance, maximum iterations, and pressure/velocity
  relaxation factors.
- `coupling_strategy`: SIMPLE, SIMPLEC, or SIMPLER strategy object.
- `momentum_terms`: optional 1D source/sink terms.

`build_case(problem)` copies the initial field and wires:

1. `MomentumAssembler`
2. `PressureCorrectionAssembler`
3. `PressureVelocityStepSolver`
4. `PressureVelocitySolver`

The resulting `SolverCase` owns both the immutable definition and the mutable
working field.

## Execution Flow

For registered nozzle/channel cases, the execution path is:

1. A case factory creates a `ProblemDefinition`.
2. `build_case()` constructs assemblers, step solver, and global solver.
3. `PressureVelocitySolver.solve()` iterates until convergence or
   `max_iterations`.
4. Each iteration calls `PressureVelocityStepSolver.run_single_iteration()`.
5. The selected strategy performs one or more internal pressure-velocity
   stages.
6. The global solver evaluates continuity and momentum residuals.

The convergence state is based on:

```text
residual = max(||continuity residual||inf, ||momentum residual||inf)
```

The solve is marked converged when this scalar is below the configured
tolerance.

## Internal Dependencies

The core 1D solver has a deliberately narrow dependency graph:

```text
Geometry + Field
      |
      v
MomentumAssembler ----> MomentumCoefficients
      |                         |
      v                         v
PressureCorrectionAssembler -> LinearSystem -> tdma
      |
      v
PressureVelocityStepSolver -> PressureVelocitySolver
```

The analytic benchmarks reuse `LinearSystem`, `tdma`, and artifact helpers from
`analytic_benchmarks`, but they do not use `ProblemDefinition` or the SIMPLE
case registry because they are direct scalar diffusion problems with known
analytic solutions.

## Registered Cases

Built-in registered solver cases are all 1D:

- `versteeg_6_2`
- `constant_area_1d`
- `linear_nozzle_1d`
- `smooth_linear_nozzle_1d`
- `aggressive_contraction_1d`
- `strong_contraction_1d`
- `expansion_1d`
- `nearly_constant_area_1d`
- `fine_mesh_nozzle_1d`
- `poor_initial_guess_1d`

Only `versteeg_6_2` currently declares a golden `NumericalBenchmark` entry in
the case registry.

## Command Line Integration

The CLI exposes:

- `list-cases`
- `list-methods`
- `run`
- `poiseuille`
- `couette`
- `verify-analytic`

`run` executes registered 1D solver cases. The analytic benchmark commands
generate standalone verification artifacts.
