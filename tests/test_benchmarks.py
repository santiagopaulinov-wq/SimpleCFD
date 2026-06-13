import numpy as np
import pytest

from simplecfd.benchmarks import NumericalBenchmark, NumericExpectation
from simplecfd.cases import (
    CaseRegistry,
    get_case_benchmarks,
    list_registered_benchmarks,
    register_case_benchmark,
    versteeg_example_6_2_problem,
)
from simplecfd.schemes import Upwind


SCHEMES = {
    "upwind": Upwind,
}


def _build_benchmark_case(benchmark):
    configuration = benchmark.configuration
    scheme = SCHEMES[configuration["scheme"]]()
    from simplecfd.cases import build_case_by_name

    return build_case_by_name(
        benchmark.case_name,
        coupling=configuration["coupling"],
        scheme=scheme,
        tolerance=configuration["tolerance"],
        max_iterations=configuration["max_iterations"],
        pressure_relaxation=configuration["pressure_relaxation"],
        velocity_relaxation=configuration["velocity_relaxation"],
    )


def test_versteeg_6_2_registers_golden_numerical_benchmark():
    benchmarks = get_case_benchmarks("versteeg_6_2")

    assert [benchmark.variant for benchmark in benchmarks] == ["golden_simple_upwind"]
    benchmark = benchmarks[0]
    assert benchmark.configuration == {
        "coupling": "simple",
        "scheme": "upwind",
        "tolerance": 1e-5,
        "max_iterations": 100,
        "pressure_relaxation": 0.7,
        "velocity_relaxation": 0.7,
    }
    assert set(benchmark.expectations) == {
        "pressure",
        "velocity",
        "mass_flow",
        "residual",
        "continuity_residual",
        "momentum_residual",
        "iterations",
    }


@pytest.mark.parametrize(
    "benchmark",
    list_registered_benchmarks("versteeg_6_2"),
    ids=lambda benchmark: f"{benchmark.case_name}[{benchmark.variant}]",
)
def test_registered_numerical_benchmark_matches_solver_result(benchmark):
    case = _build_benchmark_case(benchmark)

    result = case.solver.solve()
    mass_flow = case.definition.properties.density * case.field.u * case.geometry.velocity_areas

    benchmark.assert_result(
        result,
        pressure=case.field.p,
        velocity=case.field.u,
        mass_flow=mass_flow,
    )


def test_numeric_expectation_accepts_ranges_for_future_case_benchmarks():
    expectation = NumericExpectation(lower=[0.9, 1.9], upper=[1.1, 2.1], atol=1e-12)

    assert expectation.failure_message("velocity", np.array([1.0, 2.0])) is None
    assert "expected <=" in expectation.failure_message("velocity", np.array([1.0, 2.2]))


def test_benchmark_reports_all_metric_failures():
    benchmark = NumericalBenchmark(
        case_name="demo",
        variant="loose",
        configuration={},
        pressure=NumericExpectation(expected=[1.0, 2.0]),
        iterations=NumericExpectation(lower=2, upper=5),
    )

    failures = benchmark.check_result(
        {
            "residual": 0.0,
            "continuity_residual": 0.0,
            "momentum_residual": 0.0,
            "iterations": 8,
        },
        pressure=[1.0, 3.0],
        velocity=[],
        mass_flow=[],
    )

    assert len(failures) == 2
    assert "pressure expected" in failures[0]
    assert "iterations expected <=" in failures[1]


def test_case_registry_allows_cases_to_declare_benchmarks_at_registration():
    benchmark = NumericalBenchmark(
        case_name="custom_nozzle",
        variant="acceptance",
        configuration={},
        residual=NumericExpectation(upper=1e-5),
    )
    registry = CaseRegistry(problem_factories={})

    registry.register("custom_nozzle", versteeg_example_6_2_problem, benchmarks=(benchmark,))

    assert registry.list_benchmarks("custom_nozzle") == (benchmark,)


def test_case_registry_rejects_benchmark_for_different_case_name():
    benchmark = NumericalBenchmark(
        case_name="other",
        variant="acceptance",
        configuration={},
        residual=NumericExpectation(upper=1e-5),
    )
    registry = CaseRegistry(problem_factories={})
    registry.register("custom_nozzle", versteeg_example_6_2_problem)

    with pytest.raises(ValueError, match="case_name"):
        registry.register_benchmark("custom_nozzle", benchmark)


def test_register_case_benchmark_rejects_duplicate_variant():
    benchmark = get_case_benchmarks("versteeg_6_2")[0]

    with pytest.raises(ValueError, match="already registered"):
        register_case_benchmark("versteeg_6_2", benchmark)
