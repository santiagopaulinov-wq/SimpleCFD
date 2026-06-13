# SimpleCFD Portfolio Positioning

This document summarizes SimpleCFD as a professional portfolio project. It is
based only on capabilities implemented in the repository: the packaged 1D
solver, analytic benchmarks, tests, CLI, generated artifacts, documentation,
and CI workflow.

## Elevator Pitch

SimpleCFD is a compact 1D finite-volume CFD package built to demonstrate
research software engineering practice around pressure-velocity coupling. It
implements SIMPLE, SIMPLEC, and SIMPLER on a staggered grid, includes analytic
Poiseuille and Couette verification benchmarks, exports reproducible CSV,
Markdown, and PNG artifacts, and runs a full pytest suite in GitHub Actions
across Python 3.10, 3.11, and 3.12.

## Two-Minute Technical Explanation

SimpleCFD focuses on one-dimensional finite-volume pressure-velocity coupling
for nozzle and channel-style problems. The core data model separates geometry,
fields, boundary conditions, solver controls, schemes, assemblers, coupling
strategies, and nonlinear iteration. The solver uses staggered pressure and
velocity locations, assembles tridiagonal momentum and pressure-correction
systems, and solves them with a TDMA implementation.

The project implements three pressure-velocity coupling strategies: SIMPLE,
SIMPLEC, and SIMPLER. Registered cases include the Versteeg and Malalasekera
example 6.2 nozzle problem plus several additional 1D area-variation cases.
The CLI can list cases and methods, run solver configurations, and export
residual histories, pressure profiles, velocity profiles, mass-flow profiles,
JSON summaries, and plots.

Verification is split into solver-regression evidence and analytic benchmark
evidence. The registered Versteeg case has a golden numerical benchmark for
the SIMPLE/upwind configuration. The analytic Poiseuille and Couette modules
solve 1D transverse diffusion problems with exact reference solutions, compute
L1, L2, and Linf profile errors, run mesh-refinement studies, estimate observed
orders for meaningful integral quantities, and generate reproducible CSV,
Markdown, and PNG artifacts. The aggregate analytic verification report shows
roundoff-level nodal profile errors and second-order convergence of integrated
flow-rate or kinetic-energy metrics.

From a software perspective, the repository is installable with `pip install
-e .[dev]`, exposes a console script, contains focused tests for numerical
contracts and artifact generation, documents architecture and limitations, and
uses GitHub Actions to validate installation, CLI workflows, benchmarks, and
the full pytest suite on supported Python versions.

## CV Description

Built SimpleCFD, a Python research software project for 1D finite-volume CFD
with staggered-grid pressure-velocity coupling. Implemented SIMPLE, SIMPLEC,
and SIMPLER strategies; reusable case registration; TDMA-based linear solves;
analytic Poiseuille and Couette verification benchmarks; mesh-refinement error
studies; reproducible CLI artifact generation; technical documentation; and
GitHub Actions CI across Python 3.10-3.12.

## LinkedIn Description

SimpleCFD is a focused research software engineering project around 1D CFD
verification. It combines finite-volume pressure-velocity coupling, analytic
benchmarks, convergence studies, a tested Python package, CLI-generated
reports, and CI. The emphasis is on numerical transparency: local coefficient
tests, boundary-condition checks, solver regression tests, analytic reference
solutions, and generated evidence that can be inspected from the command line.

## GitHub Description

Installable 1D finite-volume CFD package with SIMPLE, SIMPLEC, SIMPLER,
analytic Poiseuille and Couette benchmarks, mesh-refinement verification,
CLI-generated CSV/Markdown/PNG artifacts, documentation, a pytest suite, and
GitHub Actions CI.

## Technical Highlights

- 1D staggered-grid finite-volume solver with separate pressure and velocity
  fields.
- Momentum, pressure-correction, and SIMPLER absolute-pressure assemblers.
- SIMPLE, SIMPLEC, and SIMPLER pressure-velocity coupling strategies.
- Upwind and central-difference convection policies.
- TDMA implementation for tridiagonal linear systems.
- Registered case architecture for reproducible 1D nozzle/channel problems.
- Golden numerical benchmark for Versteeg and Malalasekera example 6.2.
- Analytic Poiseuille benchmark with exact parabolic velocity profile.
- Analytic Couette benchmark with exact linear velocity profile.
- Mesh-refinement utilities and observed-order calculations.
- Reusable benchmark helpers for error norms, Markdown tables, CSV writing,
  and convergence plots.
- CLI entry points for solver runs, analytic benchmark generation, and
  aggregate verification.
- Generated evidence artifacts under `docs/assets/` for README presentation.
- Documentation covering architecture, numerical methods, verification,
  testing, limitations, and developer workflow.
- CI workflow that installs the package in editable mode and runs full tests
  on Python 3.10, 3.11, and 3.12.

## Quantitative Achievements

- 28 pytest files covering package metadata, geometry, fields, assemblers,
  pressure correction, SIMPLE-family iterations, TDMA, benchmarks, reports,
  CLI behavior, dashboard exports, and verification utilities.
- 273 tests passed in the latest local run on Python 3.12.
- CI matrix targets 3 supported Python versions: 3.10, 3.11, and 3.12.
- 10 registered 1D solver cases are available through the case registry.
- 3 pressure-velocity coupling methods are exposed by the CLI: SIMPLE,
  SIMPLEC, and SIMPLER.
- 2 analytic benchmarks generate independent CSV, Markdown, and PNG artifacts.
- Poiseuille profile errors at 33 nodes: L1 `1.60646e-15`, L2
  `1.88916e-15`, Linf `3.10862e-15`.
- Couette profile errors at 33 nodes: L1 `1.26151e-15`, L2 `1.45834e-15`,
  Linf `2.33147e-15`.
- Poiseuille flow-rate relative error at 33 nodes: `9.76563e-4`, with finest
  observed integral order near 2.
- Couette kinetic-energy relative error at 33 nodes: `4.88281e-4`, with
  finest observed integral order near 2.

## Ten Most Important Technical Achievements

1. Converted a CFD script-style concept into an installable Python package.
2. Implemented a modular 1D staggered-grid pressure-velocity solver.
3. Added three pressure-velocity coupling strategies behind a common workflow.
4. Built local numerical tests for coefficients, boundary terms, pressure
   correction, mass conservation, and TDMA behavior.
5. Created a registered-case system for reproducible solver configurations.
6. Added a golden numerical benchmark for the Versteeg 6.2 case.
7. Implemented analytic Poiseuille verification with error norms and mesh
   refinement.
8. Implemented analytic Couette verification with no-slip validation and
   integral convergence.
9. Added CLI workflows that export CSV, Markdown, JSON, and PNG artifacts.
10. Added documentation and GitHub Actions CI that make quality checks
    repeatable outside the local machine.

## Lessons Learned

- Numerical credibility depends on testing internal contracts, not only final
  solver output.
- Exact analytic profiles are useful, but integrated quantities can provide a
  more meaningful convergence signal when nodal errors reach roundoff.
- A small solver becomes easier to trust when architecture, limitations, and
  unsupported features are documented explicitly.
- CLI-generated artifacts make verification easier to reproduce and easier to
  review than ad hoc notebook output.
- CI should validate installation and representative workflows, not only unit
  tests.
- Scope control is a technical strength: keeping the project 1D avoided a
  forced 2D architecture before the necessary mesh and field abstractions
  existed.

## Ten Future Improvements With Strong Professional Return

1. Add linting and formatting checks with a minimal tool such as Ruff.
2. Add static typing checks for public modules once annotations are consistent
   enough to be meaningful.
3. Add coverage reporting and publish a coverage badge after CI is connected
   to a GitHub remote.
4. Add a manufactured-solution benchmark for the registered 1D solver path,
   not only the standalone analytic diffusion benchmarks.
5. Add API reference documentation generated from docstrings.
6. Add release automation and versioned benchmark artifacts.
7. Add a small notebook or script that walks through one verification workflow
   from installation to generated report.
8. Add performance profiling for solver iterations and benchmark generation.
9. Add more robust comparison reports for SIMPLE vs SIMPLEC vs SIMPLER across
   registered cases.
10. Add a clearly separated Scientific ML exploration only after the numerical
    verification foundation is stable, for example surrogate modeling of 1D
    convergence or parameter sweeps.

## Role Fit Assessment

| Target | Score | Honest assessment |
| --- | ---: | --- |
| Internships | 8/10 | Strong because it shows packaging, tests, CLI, docs, and numerical thinking beyond class assignments. |
| Research internships | 8/10 | Strong for groups that value verification and reproducibility; weaker if the role expects novel physics or published research. |
| Junior CFD | 6.5/10 | Good evidence of finite-volume fundamentals and pressure-velocity coupling, but it does not yet demonstrate 2D/3D CFD, turbulence, meshing, or production solver use. |
| Junior Scientific Software Engineer | 8/10 | Strong evidence of maintainable Python, testing, CI, documentation, and reproducible scientific outputs. |
| Future Scientific ML route | 5.5/10 | Useful numerical foundation, but it does not yet include ML models, datasets, training pipelines, uncertainty analysis, or differentiable/physics-informed components. |

## Prioritized Professional Roadmap

1. Add Ruff formatting/linting and include it in CI.
2. Add coverage measurement and publish coverage once a GitHub remote exists.
3. Add one manufactured-solution verification case for the registered solver
   architecture.
4. Add a short reproducible walkthrough document or notebook for Poiseuille,
   Couette, and Versteeg outputs.
5. Add docstrings and generated API reference pages for public extension
   points.
6. Add release tags with archived benchmark artifacts.
7. Add performance telemetry for solver iterations and report generation.
8. Expand method-comparison reports across the registered case suite.
9. Add a parameter-sweep dataset generator for 1D cases.
10. Use that dataset as the first careful bridge toward Scientific ML, such as
    surrogate modeling or error prediction, without claiming general CFD ML.

## Future Work

Future work should preserve the current evidence-based style. The most useful
next steps are not larger claims, but stronger automation and broader
verification: linting, coverage, manufactured solutions for the registered
solver path, versioned reference artifacts, and clearer public API docs. A
Scientific ML direction is plausible later, but it should be built as a
separate layer over verified 1D data rather than presented as an existing
capability.
