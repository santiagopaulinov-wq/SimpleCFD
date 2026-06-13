from __future__ import annotations

import numpy as np

from simplecfd import run_mesh_refinement_study


def main() -> None:
    study = run_mesh_refinement_study(
        "linear_nozzle_1d",
        node_counts=(5, 7, 9, 11),
        length=1.0,
        inlet_area=1.0,
        outlet_area=0.6,
        max_iterations=300,
        pressure_relaxation=0.45,
        velocity_relaxation=0.45,
    )

    print("node_counts:", study["node_counts"])
    print(
        "mass_flow_means:",
        np.array2string(study["mass_flow"]["final_means"], precision=6),
    )
    print(
        "velocity_order:",
        np.array2string(
            study["spatial_convergence"]["velocity"]["observed_order"],
            precision=3,
        ),
    )


if __name__ == "__main__":
    main()
