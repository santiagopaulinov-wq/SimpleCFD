from simplecfd.boundary import InletStagnationPressure, OutletFixedPressure
from simplecfd.cases import (
    BoundaryConditions,
    FlowProperties,
    ProblemDefinition,
    SolverCase,
    SolverControls,
    build_case,
    build_case_by_name,
    build_problem_by_name,
    build_versteeg_example_6_2_case,
    get_case_benchmarks,
    list_available_cases,
    list_registered_benchmarks,
)
from simplecfd.comparison import (
    compare_case_variants,
    compare_registered_cases,
    compare_simple_family,
    compare_simple_vs_simplec,
    compare_upwind_vs_central_difference,
)
from simplecfd.couette import (
    CouetteProblem,
    CouetteResult,
    generate_couette_benchmark,
    run_couette_refinement,
    solve_couette,
)
from simplecfd.fields import Field
from simplecfd.geometry import Geometry
from simplecfd.mesh_refinement import run_mesh_refinement_study
from simplecfd.momentum_terms import (
    LinearFrictionLoss,
    LocalizedLoss,
    MomentumDiffusion,
)
from simplecfd.poiseuille import (
    PoiseuilleProblem,
    PoiseuilleResult,
    generate_poiseuille_benchmark,
    run_poiseuille_refinement,
    solve_poiseuille,
)
from simplecfd.reports import generate_method_comparison_report
from simplecfd.schemes import CentralDifference, Upwind
from simplecfd.verification import generate_analytic_verification_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BoundaryConditions",
    "CentralDifference",
    "CouetteProblem",
    "CouetteResult",
    "Field",
    "FlowProperties",
    "Geometry",
    "InletStagnationPressure",
    "LinearFrictionLoss",
    "LocalizedLoss",
    "MomentumDiffusion",
    "OutletFixedPressure",
    "PoiseuilleProblem",
    "PoiseuilleResult",
    "ProblemDefinition",
    "SolverCase",
    "SolverControls",
    "Upwind",
    "build_case",
    "build_case_by_name",
    "build_problem_by_name",
    "build_versteeg_example_6_2_case",
    "compare_case_variants",
    "compare_registered_cases",
    "compare_simple_family",
    "compare_simple_vs_simplec",
    "compare_upwind_vs_central_difference",
    "generate_method_comparison_report",
    "generate_analytic_verification_report",
    "generate_couette_benchmark",
    "generate_poiseuille_benchmark",
    "get_case_benchmarks",
    "list_available_cases",
    "list_registered_benchmarks",
    "run_mesh_refinement_study",
    "run_couette_refinement",
    "run_poiseuille_refinement",
    "solve_couette",
    "solve_poiseuille",
]
