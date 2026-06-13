from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from math import isfinite
from numbers import Real
from typing import Any, Callable

from simplecfd.assembly.momentum import MomentumAssembler
from simplecfd.assembly.pressure_correction import PressureCorrectionAssembler
from simplecfd.benchmarks import NumericalBenchmark, NumericExpectation
from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.momentum_terms import MomentumTerm
from simplecfd.simple_loop import (
    PressureVelocityCouplingStrategy,
    PressureVelocityStepSolver,
    SIMPLECouplingStrategy,
    SIMPLECCouplingStrategy,
    SIMPLERCouplingStrategy,
)
from simplecfd.schemes.base import ConvectionScheme
from simplecfd.schemes.upwind import Upwind
from simplecfd.solver import PressureVelocitySolver


@dataclass
class FlowProperties:
    density: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_finite("density", self.density)


@dataclass
class BoundaryConditions:
    inlet: InletStagnationPressure | None = None
    outlet: OutletFixedPressure | None = None


@dataclass
class SolverControls:
    tolerance: float = 1e-5
    max_iterations: int = 100
    pressure_relaxation: float = 0.7
    velocity_relaxation: float = 0.7

    def __post_init__(self) -> None:
        _validate_positive_finite("tolerance", self.tolerance)
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool):
            raise ValueError("max_iterations must be a positive integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        _validate_relaxation("pressure_relaxation", self.pressure_relaxation)
        _validate_relaxation("velocity_relaxation", self.velocity_relaxation)


@dataclass
class ProblemDefinition:
    geometry: Geometry
    initial_field: Field
    properties: FlowProperties
    boundaries: BoundaryConditions
    scheme: ConvectionScheme
    controls: SolverControls
    coupling_strategy: PressureVelocityCouplingStrategy = dataclass_field(
        default_factory=SIMPLECouplingStrategy
    )
    momentum_terms: tuple[MomentumTerm, ...] = ()

    def __post_init__(self) -> None:
        _validate_required("geometry", self.geometry)
        _validate_required("initial_field", self.initial_field)
        _validate_required("properties", self.properties)
        _validate_required("boundaries", self.boundaries)
        _validate_required("scheme", self.scheme)
        _validate_required("controls", self.controls)
        _validate_required("coupling_strategy", self.coupling_strategy)
        self.momentum_terms = tuple(self.momentum_terms)
        self.initial_field.validate_against(self.geometry)


@dataclass
class SolverCase:
    definition: ProblemDefinition
    geometry: Geometry
    field: Field
    momentum_asm: MomentumAssembler
    pressure_correction_asm: PressureCorrectionAssembler
    step_solver: PressureVelocityStepSolver
    solver: PressureVelocitySolver


VersteegExample62Case = SolverCase
ProblemFactory = Callable[..., ProblemDefinition]
COUPLING_STRATEGIES = {
    "simple": SIMPLECouplingStrategy,
    "simplec": SIMPLECCouplingStrategy,
    "simpler": SIMPLERCouplingStrategy,
}


@dataclass
class CaseRegistry:
    problem_factories: dict[str, ProblemFactory]
    case_benchmarks: dict[str, tuple[NumericalBenchmark, ...]] = dataclass_field(
        default_factory=dict
    )

    def register(
        self,
        name: str,
        factory: ProblemFactory,
        benchmarks: tuple[NumericalBenchmark, ...] = (),
    ) -> None:
        _validate_case_name(name)
        _validate_required("factory", factory)
        if not callable(factory):
            raise ValueError("factory must be callable")
        if name in self.problem_factories:
            raise ValueError(f"case '{name}' is already registered")
        self.problem_factories[name] = factory
        self.case_benchmarks[name] = ()
        for benchmark in benchmarks:
            self.register_benchmark(name, benchmark)

    def register_benchmark(self, name: str, benchmark: NumericalBenchmark) -> None:
        _validate_case_name(name)
        _validate_required("benchmark", benchmark)
        if name not in self.problem_factories:
            available = ", ".join(self.list_cases()) or "none"
            raise ValueError(f"unknown case '{name}'. Available cases: {available}")
        if not isinstance(benchmark, NumericalBenchmark):
            raise ValueError("benchmark must be a NumericalBenchmark")
        if benchmark.case_name != name:
            raise ValueError("benchmark case_name must match the registered case name")
        existing = self.case_benchmarks.get(name, ())
        if any(item.variant == benchmark.variant for item in existing):
            raise ValueError(
                f"benchmark '{benchmark.variant}' is already registered for case '{name}'"
            )
        self.case_benchmarks[name] = (*existing, benchmark)

    def list_cases(self) -> tuple[str, ...]:
        return tuple(sorted(self.problem_factories))

    def list_benchmarks(self, name: str | None = None) -> tuple[NumericalBenchmark, ...]:
        if name is not None:
            return self.benchmarks_for(name)
        benchmarks = []
        for case_name in self.list_cases():
            benchmarks.extend(self.case_benchmarks.get(case_name, ()))
        return tuple(benchmarks)

    def benchmarks_for(self, name: str) -> tuple[NumericalBenchmark, ...]:
        self._factory_for(name)
        return self.case_benchmarks.get(name, ())

    def build_problem(self, name: str, **kwargs: Any) -> ProblemDefinition:
        factory = self._factory_for(name)
        problem = factory(**kwargs)
        if not isinstance(problem, ProblemDefinition):
            raise ValueError(f"case '{name}' factory must return a ProblemDefinition")
        return problem

    def build_case(self, name: str, **kwargs: Any) -> SolverCase:
        return build_case(self.build_problem(name, **kwargs))

    def _factory_for(self, name: str) -> ProblemFactory:
        _validate_case_name(name)
        try:
            return self.problem_factories[name]
        except KeyError as exc:
            available = ", ".join(self.list_cases()) or "none"
            raise ValueError(f"unknown case '{name}'. Available cases: {available}") from exc


def _validate_required(name: str, value: object) -> None:
    if value is None:
        raise ValueError(f"{name} is required")


def _validate_case_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("case name must be a non-empty string")


def _validate_positive_finite(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_relaxation(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be in the interval (0, 1]")
    if not isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1]")


def build_case(problem: ProblemDefinition) -> SolverCase:
    _validate_required("problem", problem)
    field = problem.initial_field.copy()
    field.validate_against(problem.geometry)

    momentum_asm = MomentumAssembler(
        geometry=problem.geometry,
        density=problem.properties.density,
        scheme=problem.scheme,
        inlet=problem.boundaries.inlet,
        outlet=problem.boundaries.outlet,
        terms=problem.momentum_terms,
    )
    pressure_correction_asm = PressureCorrectionAssembler(
        geometry=problem.geometry,
        density=problem.properties.density,
    )
    step_solver = PressureVelocityStepSolver(
        geometry=problem.geometry,
        field=field,
        momentum_assembler=momentum_asm,
        pressure_correction_assembler=pressure_correction_asm,
        pressure_relaxation=problem.controls.pressure_relaxation,
        velocity_relaxation=problem.controls.velocity_relaxation,
        coupling_strategy=problem.coupling_strategy,
    )
    solver = PressureVelocitySolver(
        geometry=problem.geometry,
        field=field,
        step_solver=step_solver,
        density=problem.properties.density,
        tolerance=problem.controls.tolerance,
        max_iterations=problem.controls.max_iterations,
    )
    return SolverCase(
        definition=problem,
        geometry=problem.geometry,
        field=field,
        momentum_asm=momentum_asm,
        pressure_correction_asm=pressure_correction_asm,
        step_solver=step_solver,
        solver=solver,
    )


CASE_REGISTRY = CaseRegistry(problem_factories={})


def register_case(
    name: str,
    factory: ProblemFactory,
    benchmarks: tuple[NumericalBenchmark, ...] = (),
) -> None:
    CASE_REGISTRY.register(name, factory, benchmarks=benchmarks)


def register_case_benchmark(name: str, benchmark: NumericalBenchmark) -> None:
    CASE_REGISTRY.register_benchmark(name, benchmark)


def list_available_cases() -> tuple[str, ...]:
    return CASE_REGISTRY.list_cases()


def list_registered_benchmarks(name: str | None = None) -> tuple[NumericalBenchmark, ...]:
    return CASE_REGISTRY.list_benchmarks(name)


def get_case_benchmarks(name: str) -> tuple[NumericalBenchmark, ...]:
    return CASE_REGISTRY.benchmarks_for(name)


def build_problem_by_name(name: str, **kwargs: Any) -> ProblemDefinition:
    return CASE_REGISTRY.build_problem(name, **kwargs)


def build_case_by_name(name: str, **kwargs: Any) -> SolverCase:
    return CASE_REGISTRY.build_case(name, **kwargs)


def build_coupling_strategy(name: str) -> PressureVelocityCouplingStrategy:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("coupling must be a non-empty string")
    normalized = name.strip().lower()
    try:
        return COUPLING_STRATEGIES[normalized]()
    except KeyError as exc:
        available = ", ".join(sorted(COUPLING_STRATEGIES))
        raise ValueError(
            f"unknown pressure-velocity coupling '{name}'. Available couplings: {available}"
        ) from exc


def versteeg_example_6_2_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    selected_coupling = (
        build_coupling_strategy(coupling)
        if coupling_strategy is None
        else coupling_strategy
    )

    return ProblemDefinition(
        geometry=Geometry.versteeg_example_6_2(),
        initial_field=Field.versteeg_example_6_2_initial_guess(),
        properties=FlowProperties(density=density),
        boundaries=BoundaryConditions(
            inlet=InletStagnationPressure(stagnation_pressure=10.0),
            outlet=OutletFixedPressure(pressure=0.0),
        ),
        scheme=Upwind() if scheme is None else scheme,
        controls=SolverControls(
            tolerance=tolerance,
            max_iterations=max_iterations,
            pressure_relaxation=pressure_relaxation,
            velocity_relaxation=velocity_relaxation,
        ),
        coupling_strategy=selected_coupling,
        momentum_terms=momentum_terms,
    )


def linear_nozzle_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
    inlet_area: float = 1.0,
    outlet_area: float = 0.5,
    n_pressure: int = 5,
    dx: float | Any = 0.25,
    mass_flow_guess: float = 0.6,
    inlet_pressure: float = 5.0,
    outlet_pressure: float = 1.0,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=inlet_area,
        outlet_area=outlet_area,
        n_pressure=n_pressure,
        dx=dx,
        mass_flow_guess=mass_flow_guess,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def constant_area_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 200,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Constant-area variant of the Example 6.2 staggered 1D nozzle setup."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=1.0,
        outlet_area=1.0,
        n_pressure=6,
        dx=0.2,
        mass_flow_guess=1.0,
        inlet_pressure=5.0,
        outlet_pressure=1.0,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def smooth_linear_nozzle_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 200,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
    inlet_area: float = 1.0,
    outlet_area: float = 0.7,
    n_pressure: int = 6,
    dx: float | Any = 0.2,
    mass_flow_guess: float = 0.8,
    inlet_pressure: float = 5.0,
    outlet_pressure: float = 1.0,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Mild linear contraction using the same 1D coupling setup as Example 6.2."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=inlet_area,
        outlet_area=outlet_area,
        n_pressure=n_pressure,
        dx=dx,
        mass_flow_guess=mass_flow_guess,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def aggressive_contraction_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 300,
    pressure_relaxation: float = 0.5,
    velocity_relaxation: float = 0.5,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Stronger linear contraction with conservative relaxation for stability."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=1.0,
        outlet_area=0.25,
        n_pressure=6,
        dx=0.2,
        mass_flow_guess=0.6,
        inlet_pressure=8.0,
        outlet_pressure=1.0,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def strong_contraction_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 500,
    pressure_relaxation: float = 0.35,
    velocity_relaxation: float = 0.35,
    inlet_area: float = 1.0,
    outlet_area: float = 0.15,
    n_pressure: int = 8,
    dx: float | Any = 0.15,
    mass_flow_guess: float = 0.45,
    inlet_pressure: float = 10.0,
    outlet_pressure: float = 1.0,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Severe contraction used as a robustness stress case."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=inlet_area,
        outlet_area=outlet_area,
        n_pressure=n_pressure,
        dx=dx,
        mass_flow_guess=mass_flow_guess,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def expansion_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 400,
    pressure_relaxation: float = 0.45,
    velocity_relaxation: float = 0.45,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Area expansion with conservative relaxation for robustness comparisons."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=0.45,
        outlet_area=1.1,
        n_pressure=7,
        dx=0.2,
        mass_flow_guess=0.5,
        inlet_pressure=6.0,
        outlet_pressure=1.0,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def nearly_constant_area_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 250,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Almost constant area case with a small contraction."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=1.0,
        outlet_area=0.97,
        n_pressure=7,
        dx=0.2,
        mass_flow_guess=0.8,
        inlet_pressure=5.0,
        outlet_pressure=1.0,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def fine_mesh_nozzle_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 1000,
    pressure_relaxation: float = 0.3,
    velocity_relaxation: float = 0.3,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Finer 1D mesh with conservative relaxation."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=1.0,
        outlet_area=0.75,
        n_pressure=12,
        dx=0.08,
        mass_flow_guess=0.7,
        inlet_pressure=5.0,
        outlet_pressure=1.0,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def poor_initial_guess_1d_problem(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 500,
    pressure_relaxation: float = 0.45,
    velocity_relaxation: float = 0.45,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> ProblemDefinition:
    """Moderate nozzle started from a deliberately low mass-flow guess."""
    return _nozzle_1d_problem(
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        inlet_area=1.0,
        outlet_area=0.55,
        n_pressure=6,
        dx=0.2,
        mass_flow_guess=0.05,
        inlet_pressure=6.0,
        outlet_pressure=1.0,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


def _nozzle_1d_problem(
    density: float,
    tolerance: float,
    max_iterations: int,
    pressure_relaxation: float,
    velocity_relaxation: float,
    inlet_area: float,
    outlet_area: float,
    n_pressure: int,
    dx: float | Any,
    mass_flow_guess: float,
    inlet_pressure: float,
    outlet_pressure: float,
    scheme: ConvectionScheme | None,
    coupling: str,
    coupling_strategy: PressureVelocityCouplingStrategy | None,
    momentum_terms: tuple[MomentumTerm, ...],
) -> ProblemDefinition:
    selected_coupling = (
        build_coupling_strategy(coupling)
        if coupling_strategy is None
        else coupling_strategy
    )
    geometry = Geometry.linear_nozzle(
        inlet_area=inlet_area,
        outlet_area=outlet_area,
        n_pressure=n_pressure,
        dx=dx,
    )

    return ProblemDefinition(
        geometry=geometry,
        initial_field=Field.initial_nozzle_guess(
            geometry=geometry,
            density=density,
            mass_flow_guess=mass_flow_guess,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure,
        ),
        properties=FlowProperties(density=density),
        boundaries=BoundaryConditions(
            inlet=InletStagnationPressure(stagnation_pressure=inlet_pressure),
            outlet=OutletFixedPressure(pressure=outlet_pressure),
        ),
        scheme=Upwind() if scheme is None else scheme,
        controls=SolverControls(
            tolerance=tolerance,
            max_iterations=max_iterations,
            pressure_relaxation=pressure_relaxation,
            velocity_relaxation=velocity_relaxation,
        ),
        coupling_strategy=selected_coupling,
        momentum_terms=momentum_terms,
    )


def build_versteeg_example_6_2_case(
    density: float = 1.0,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
    pressure_relaxation: float = 0.7,
    velocity_relaxation: float = 0.7,
    scheme: ConvectionScheme | None = None,
    coupling: str = "simple",
    coupling_strategy: PressureVelocityCouplingStrategy | None = None,
    momentum_terms: tuple[MomentumTerm, ...] = (),
) -> VersteegExample62Case:
    return build_case_by_name(
        "versteeg_6_2",
        density=density,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pressure_relaxation=pressure_relaxation,
        velocity_relaxation=velocity_relaxation,
        scheme=scheme,
        coupling=coupling,
        coupling_strategy=coupling_strategy,
        momentum_terms=momentum_terms,
    )


VERSTEEG_6_2_GOLDEN_BENCHMARK = NumericalBenchmark(
    case_name="versteeg_6_2",
    variant="golden_simple_upwind",
    configuration={
        "coupling": "simple",
        "scheme": "upwind",
        "tolerance": 1e-5,
        "max_iterations": 100,
        "pressure_relaxation": 0.7,
        "velocity_relaxation": 0.7,
    },
    pressure=NumericExpectation(
        expected=[
            10.0,
            9.004194245993599,
            8.2506112979705,
            6.19430740551947,
            0.0,
        ],
        rtol=1e-8,
        atol=1e-10,
    ),
    velocity=NumericExpectation(
        expected=[
            1.3826797094190517,
            1.7777310549673522,
            2.488823476954293,
            4.148039128257155,
        ],
        rtol=1e-8,
        atol=1e-10,
    ),
    mass_flow=NumericExpectation(
        expected=[
            0.6222058692385732,
            0.6222058692385732,
            0.6222058692385732,
            0.6222058692385732,
        ],
        rtol=1e-8,
        atol=1e-10,
    ),
    residual=NumericExpectation(expected=9.765912092341011e-6, rtol=0.05, atol=1e-8),
    continuity_residual=NumericExpectation(expected=0.0, rtol=0.05, atol=1e-8),
    momentum_residual=NumericExpectation(expected=9.765912092341011e-6, rtol=0.05, atol=1e-8),
    iterations=NumericExpectation(expected=23, atol=0.0),
)


register_case(
    "versteeg_6_2",
    versteeg_example_6_2_problem,
    benchmarks=(VERSTEEG_6_2_GOLDEN_BENCHMARK,),
)
register_case("constant_area_1d", constant_area_1d_problem)
register_case("linear_nozzle_1d", linear_nozzle_1d_problem)
register_case("smooth_linear_nozzle_1d", smooth_linear_nozzle_1d_problem)
register_case("aggressive_contraction_1d", aggressive_contraction_1d_problem)
register_case("strong_contraction_1d", strong_contraction_1d_problem)
register_case("expansion_1d", expansion_1d_problem)
register_case("nearly_constant_area_1d", nearly_constant_area_1d_problem)
register_case("fine_mesh_nozzle_1d", fine_mesh_nozzle_1d_problem)
register_case("poor_initial_guess_1d", poor_initial_guess_1d_problem)
