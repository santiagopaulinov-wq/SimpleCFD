# Testing

SimpleCFD uses focused pytest tests rather than one large end-to-end test. The
tests are intended to protect numerical contracts, public APIs, CLI behavior,
and artifact generation.

## Running Tests

From the repository root:

```bash
python -m pytest
```

## Continuous Integration

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on every push
and pull request. It installs the package in editable mode on Python 3.10,
3.11, and 3.12, verifies the importable package version, checks the primary CLI
entry points, runs representative solver and analytic-benchmark commands, and
then executes the full pytest suite.

The analytic benchmark commands are included in CI because they are already
covered by fast tests and complete in a normal CI budget. They provide useful
coverage for generated CSV, Markdown, and PNG artifacts without requiring a
separate expensive benchmark stage.

Useful focused checks:

```bash
python -m pytest tests/test_poiseuille.py tests/test_couette.py
python -m pytest tests/test_analytic_benchmarks.py tests/test_verification.py
python -m pytest tests/test_momentum_internal_node.py tests/test_pressure_correction.py
python -m pytest tests/test_cli.py tests/test_distribution.py
```

## Test Groups

- `test_distribution.py`: package metadata, public exports, hygiene files, and
  example import behavior.
- `test_geometry_field.py`: 1D geometry and field validation.
- `test_momentum_internal_node.py`: internal momentum coefficients.
- `test_momentum_boundaries.py`: inlet and outlet momentum boundary assembly.
- `test_pressure_correction.py`: pressure-correction coefficients, boundary
  rows, RHS, and SIMPLEC coefficient usage.
- `test_pressure_absolute.py`: SIMPLER absolute-pressure assembly.
- `test_tdma.py`: tridiagonal solver correctness, shape validation, and input
  immutability.
- `test_simple_loop.py`: pressure-velocity step behavior, correction logic, and
  coupling strategies.
- `test_solver.py`: global solver residuals and iteration behavior.
- `test_numerical_rigor.py`: conservation and manufactured algebraic checks
  across momentum, pressure correction, and linear solves.
- `test_benchmarks.py`: registered numerical benchmark declarations and
  validation logic.
- `test_numerical_regression.py`: regression checks against stored expected
  values.
- `test_mesh_refinement.py`: 1D mesh-refinement study outputs.
- `test_case_comparison.py`: batch comparisons, stability metrics, benchmark
  errors, and failure rows.
- `test_reports.py`: method-comparison Markdown, CSV, and PNG artifacts.
- `test_poiseuille.py`: analytic Poiseuille profile, discretization,
  refinement, artifacts, and input validation.
- `test_couette.py`: analytic Couette profile, discretization, refinement,
  artifacts, and input validation.
- `test_analytic_benchmarks.py`: reusable error norms, observed order, and
  generic convergence plots.
- `test_verification.py`: aggregate analytic verification report.
- `test_cli.py`: command-line entry points and generated artifacts.
- `test_dashboard.py` and `test_convergence_telemetry.py`: convergence
  diagnostics and dashboard export helpers.

## Testing Philosophy

The test suite avoids relying only on final solver outputs. It also checks:

- local finite-volume coefficients;
- boundary contributions;
- linear-system assembly;
- residual definitions;
- conservation after correction;
- exact analytic solutions when available;
- artifact schemas and filenames;
- invalid input handling.

This makes failures easier to diagnose than a single black-box comparison.

## Known Test Environment Note

On the current Windows workspace, pytest may warn that it cannot write to an
existing `.pytest_cache` path due to permissions. This warning does not affect
the numerical test results.
