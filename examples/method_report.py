from __future__ import annotations

from pathlib import Path

from simplecfd import generate_method_comparison_report


def main() -> None:
    report = generate_method_comparison_report(
        Path("outputs") / "example_method_report",
        case_names=(
            "versteeg_6_2",
            "linear_nozzle_1d",
            "smooth_linear_nozzle_1d",
            "strong_contraction_1d",
        ),
        methods=("simple", "simplec", "simpler"),
        schemes=("upwind",),
        max_iterations=25,
    )
    print(report["paths"]["markdown"])


if __name__ == "__main__":
    main()
