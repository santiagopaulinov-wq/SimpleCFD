# Developer Guide

This guide describes how to work on SimpleCFD without breaking the current 1D
architecture.

## Installation

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Running The Project

List registered cases:

```bash
python -m simplecfd list-cases
```

Run one registered case:

```bash
python -m simplecfd run --case versteeg_6_2 --method simple --scheme upwind --output-dir outputs/run
```

Generate analytic benchmark artifacts:

```bash
python -m simplecfd poiseuille --output-dir outputs/poiseuille_benchmark
python -m simplecfd couette --output-dir outputs/couette_benchmark
python -m simplecfd verify-analytic --output-dir outputs/analytic_verification
```

## Running Tests

Run everything:

```bash
python -m pytest
```

The same full suite is executed by GitHub Actions on Python 3.10, 3.11, and
3.12 after an editable installation and representative CLI checks.

Run focused checks before changing numerical code:

```bash
python -m pytest tests/test_numerical_rigor.py tests/test_pressure_correction.py tests/test_tdma.py
```

Run focused checks before changing analytic benchmarks:

```bash
python -m pytest tests/test_analytic_benchmarks.py tests/test_poiseuille.py tests/test_couette.py tests/test_verification.py
```

## Adding A Registered 1D Solver Case

Add a factory in `simplecfd.cases` that returns `ProblemDefinition`. The case
should define:

- a 1D `Geometry`;
- an initial `Field`;
- `FlowProperties`;
- inlet/outlet `BoundaryConditions`;
- a default scheme;
- `SolverControls`;
- optional `momentum_terms`.

Register it with `register_case(name, factory)`.

Add tests that prove:

- the case appears in `list_available_cases()`;
- `build_problem_by_name()` returns a valid `ProblemDefinition`;
- `build_case_by_name()` wires the expected assemblers and controls;
- a short solve runs without non-finite values;
- any declared benchmark is matched by the solver result.

## Adding A Numerical Benchmark For A Registered Case

Use `NumericalBenchmark` and `NumericExpectation` in `simplecfd.benchmarks`.
Attach the benchmark at registration time or through `register_case_benchmark`.

Benchmarks can check:

- pressure;
- velocity;
- mass flow;
- residual;
- continuity residual;
- momentum residual;
- iteration count.

Prefer exact expected arrays only when the configuration is deterministic and
well understood. Use ranges for robustness or stress cases.

## Adding An Analytic 1D Benchmark

Use the Poiseuille and Couette modules as templates. A new analytic benchmark
should provide:

- a frozen problem dataclass with validated inputs;
- an analytic solution function;
- a `LinearSystem` assembly when a discrete solve is needed;
- a result dataclass containing numerical solution, analytic solution,
  pointwise error, `L1`, `L2`, and `Linf`;
- a refinement runner;
- artifact generation for CSV, Markdown, and PNG;
- tests for analytic values, discrete assembly, refinement, artifacts, and
  invalid inputs.

Reuse helpers from `simplecfd.analytic_benchmarks` for error norms, observed
orders, CSV writing, Markdown tables, and plots.

## Extending Numerical Methods

For new 1D convection schemes:

1. Add a class in `simplecfd/schemes/`.
2. Implement `ConvectionScheme`.
3. Export it from `simplecfd/schemes/__init__.py`.
4. Add it to `SCHEME_FACTORIES` if it should be available in comparisons.
5. Add tests for interpolation and coefficient signs.

For new SIMPLE-family coupling methods:

1. Add a strategy in `simplecfd.simple_loop` or a dedicated module if it grows.
2. Register a name in `COUPLING_STRATEGIES`.
3. Add tests for coefficients, correction equations, linear-solve counts, and
   convergence behavior.

Do not add multidimensional concepts to `Geometry`, `Field`, or the existing
assemblers unless the project explicitly changes scope. The current solver is
1D by design.

## Documentation Updates

When changing behavior, update the relevant document:

- architecture changes: `docs/architecture.md`;
- numerical method changes: `docs/numerical_methods.md`;
- benchmark or verification changes: `docs/verification.md`;
- test organization changes: `docs/testing.md`;
- scope changes: `docs/limitations.md`;
- contributor workflow changes: `docs/developer_guide.md`.
