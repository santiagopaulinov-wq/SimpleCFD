from pathlib import Path

import numpy as np

from simplecfd.analytic_benchmarks import (
    ROUND_OFF_ERROR_FLOOR,
    error_norms,
    observed_order,
    plot_error_convergence,
    plot_observed_orders,
)


def test_error_norms_report_l1_l2_and_linf():
    norms = error_norms(np.array([1.0, -2.0, 2.0, -5.0]))

    assert norms.l1 == 2.5
    np.testing.assert_allclose(norms.l2, np.sqrt(8.5))
    assert norms.linf == 5.0


def test_observed_order_suppresses_roundoff_level_errors():
    assert observed_order(1.0, 0.25, 0.5, 0.25) == 2.0
    assert observed_order(ROUND_OFF_ERROR_FLOOR / 10.0, 0.0, 0.5, 0.25) == ""


def test_error_and_order_plots_are_written_for_refinement_rows():
    rows = [
        {"dy": 0.5, "l2_error": 0.25, "observed_l2_error_order": ""},
        {"dy": 0.25, "l2_error": 0.0625, "observed_l2_error_order": 2.0},
    ]
    output_dir = Path("outputs") / "test_analytic_benchmark_utils"
    error_path = output_dir / "error.png"
    order_path = output_dir / "order.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_error_convergence(
        rows,
        error_path,
        error_columns=("l2_error",),
        labels=("L2",),
        title="Error convergence",
    )
    plot_observed_orders(
        rows,
        order_path,
        order_columns=("observed_l2_error_order",),
        labels=("L2",),
        title="Observed order",
    )

    assert error_path.exists() and error_path.stat().st_size > 0
    assert order_path.exists() and order_path.stat().st_size > 0
