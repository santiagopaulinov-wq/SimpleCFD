from __future__ import annotations

import numpy as np

from simplecfd import build_case_by_name


def main() -> None:
    case = build_case_by_name("versteeg_6_2")
    result = case.solver.solve()
    mass_flow = case.definition.properties.density * case.field.u * case.geometry.velocity_areas

    print(f"converged: {result['converged']}")
    print(f"iterations: {result['iterations']}")
    print(f"residual: {result['residual']:.6e}")
    print("pressure:", np.array2string(case.field.p, precision=6))
    print("velocity:", np.array2string(case.field.u, precision=6))
    print("mass_flow:", np.array2string(mass_flow, precision=6))


if __name__ == "__main__":
    main()
