from __future__ import annotations

from pathlib import Path

import streamlit as st

from simplecfd.dashboard import (
    DASHBOARD_CASES,
    DASHBOARD_METHODS,
    NOZZLE_CASES,
    OUTPUT_DIR,
    create_dashboard_figures,
    export_dashboard_artifacts,
    run_dashboard_case,
)


DEFAULTS = {
    "versteeg_6_2": {
        "tolerance": 1e-5,
        "max_iterations": 100,
        "pressure_relaxation": 0.7,
        "velocity_relaxation": 0.7,
        "density": 1.0,
    },
    "linear_nozzle_1d": {
        "tolerance": 1e-5,
        "max_iterations": 100,
        "pressure_relaxation": 0.7,
        "velocity_relaxation": 0.7,
        "density": 1.0,
        "inlet_area": 1.0,
        "outlet_area": 0.5,
        "n_pressure": 5,
        "dx": 0.25,
        "mass_flow_guess": 0.6,
        "inlet_pressure": 5.0,
        "outlet_pressure": 1.0,
    },
    "smooth_linear_nozzle_1d": {
        "tolerance": 1e-5,
        "max_iterations": 200,
        "pressure_relaxation": 0.7,
        "velocity_relaxation": 0.7,
        "density": 1.0,
        "inlet_area": 1.0,
        "outlet_area": 0.7,
        "n_pressure": 6,
        "dx": 0.2,
        "mass_flow_guess": 0.8,
        "inlet_pressure": 5.0,
        "outlet_pressure": 1.0,
    },
    "strong_contraction_1d": {
        "tolerance": 1e-5,
        "max_iterations": 500,
        "pressure_relaxation": 0.35,
        "velocity_relaxation": 0.35,
        "density": 1.0,
        "inlet_area": 1.0,
        "outlet_area": 0.15,
        "n_pressure": 8,
        "dx": 0.15,
        "mass_flow_guess": 0.45,
        "inlet_pressure": 10.0,
        "outlet_pressure": 1.0,
    },
}


st.set_page_config(page_title="SimpleCFD", layout="wide")
st.title("SimpleCFD")

with st.sidebar:
    case_name = st.selectbox("Caso", DASHBOARD_CASES, index=0)
    method_label = st.selectbox("Metodo", ("SIMPLE", "SIMPLEC", "SIMPLER"), index=0)
    method = method_label.lower()
    defaults = DEFAULTS[case_name]

    st.divider()
    tolerance = st.number_input(
        "tolerance",
        min_value=1e-12,
        max_value=1.0,
        value=float(defaults["tolerance"]),
        format="%.2e",
    )
    max_iterations = st.number_input(
        "max_iterations",
        min_value=1,
        max_value=5000,
        value=int(defaults["max_iterations"]),
        step=1,
    )
    pressure_relaxation = st.slider(
        "pressure_relaxation",
        min_value=0.01,
        max_value=1.0,
        value=float(defaults["pressure_relaxation"]),
        step=0.01,
    )
    velocity_relaxation = st.slider(
        "velocity_relaxation",
        min_value=0.01,
        max_value=1.0,
        value=float(defaults["velocity_relaxation"]),
        step=0.01,
    )
    density = st.number_input(
        "density",
        min_value=1e-12,
        value=float(defaults["density"]),
        format="%.6g",
    )

    nozzle_configuration = {}
    if case_name in NOZZLE_CASES:
        st.divider()
        nozzle_configuration = {
            "inlet_area": st.number_input(
                "inlet_area",
                min_value=1e-12,
                value=float(defaults["inlet_area"]),
                format="%.6g",
            ),
            "outlet_area": st.number_input(
                "outlet_area",
                min_value=1e-12,
                value=float(defaults["outlet_area"]),
                format="%.6g",
            ),
            "n_pressure": st.number_input(
                "n_pressure",
                min_value=2,
                max_value=200,
                value=int(defaults["n_pressure"]),
                step=1,
            ),
            "dx": st.number_input(
                "dx",
                min_value=1e-12,
                value=float(defaults["dx"]),
                format="%.6g",
            ),
            "mass_flow_guess": st.number_input(
                "mass_flow_guess",
                min_value=1e-12,
                value=float(defaults["mass_flow_guess"]),
                format="%.6g",
            ),
            "inlet_pressure": st.number_input(
                "inlet_pressure",
                value=float(defaults["inlet_pressure"]),
                format="%.6g",
            ),
            "outlet_pressure": st.number_input(
                "outlet_pressure",
                value=float(defaults["outlet_pressure"]),
                format="%.6g",
            ),
        }

    run = st.button("Ejecutar", type="primary", use_container_width=True)

configuration = {
    "tolerance": tolerance,
    "max_iterations": int(max_iterations),
    "pressure_relaxation": pressure_relaxation,
    "velocity_relaxation": velocity_relaxation,
    "density": density,
    **nozzle_configuration,
}

if run:
    with st.spinner("Ejecutando simulacion..."):
        case, history, summary = run_dashboard_case(case_name, method, configuration)
        figures = create_dashboard_figures(history, tolerance=tolerance)

    metric_columns = st.columns(5)
    metric_columns[0].metric("converged", str(summary["converged"]))
    metric_columns[1].metric("iterations", summary["iterations"])
    metric_columns[2].metric("final_residual", f"{summary['final_residual']:.3e}")
    metric_columns[3].metric(
        "continuity_residual",
        f"{summary['continuity_residual']:.3e}",
    )
    metric_columns[4].metric("momentum_residual", f"{summary['momentum_residual']:.3e}")

    st.plotly_chart(figures["velocity"], use_container_width=True)
    st.plotly_chart(figures["pressure"], use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(figures["momentum_residual"], use_container_width=True)
    right.plotly_chart(figures["continuity_residual"], use_container_width=True)
    st.plotly_chart(figures["global_residual"], use_container_width=True)

    try:
        paths = export_dashboard_artifacts(
            OUTPUT_DIR,
            case_name,
            method,
            case,
            history,
            summary,
            figures,
        )
        st.success(f"Artefactos exportados en {Path(OUTPUT_DIR).as_posix()}")
        st.caption(
            f"CSV: {paths['summary_csv']} | {paths['history_csv']}"
        )
    except Exception as exc:
        st.warning(f"No se pudieron exportar todos los artefactos: {exc}")
else:
    st.info("Configura el caso y pulsa Ejecutar.")
