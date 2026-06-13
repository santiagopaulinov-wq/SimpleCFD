import runpy
import tomllib
from pathlib import Path

import simplecfd


def test_pyproject_declares_local_distribution_metadata():
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "simplecfd"
    assert metadata["project"]["requires-python"] == ">=3.10"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    assert "numpy>=1.24" in metadata["project"]["dependencies"]
    assert metadata["project"]["scripts"]["simplecfd"] == "simplecfd.cli:main"
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["simplecfd*"]
    assert metadata["tool"]["setuptools"]["package-data"]["simplecfd"] == ["py.typed"]
    assert metadata["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_repository_hygiene_files_are_present():
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    gitignore_patterns = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert "MIT License" in license_text
    assert "outputs/" in gitignore_patterns
    assert "*.egg-info/" in gitignore_patterns
    assert "__pycache__/" in gitignore_patterns


def test_top_level_public_exports_are_available():
    expected_exports = {
        "Geometry",
        "Field",
        "build_case_by_name",
        "list_available_cases",
        "Upwind",
        "CentralDifference",
        "MomentumDiffusion",
        "CouetteProblem",
        "PoiseuilleProblem",
        "generate_analytic_verification_report",
        "solve_couette",
        "solve_poiseuille",
        "run_mesh_refinement_study",
        "generate_method_comparison_report",
    }

    assert expected_exports.issubset(set(simplecfd.__all__))
    assert simplecfd.__version__ == "0.1.0"
    assert "versteeg_6_2" in simplecfd.list_available_cases()


def test_reproducible_examples_run_without_import_side_effects(capsys):
    for example in (
        "examples/run_versteeg.py",
        "examples/mesh_refinement.py",
    ):
        runpy.run_path(example, run_name="__main__")

    captured = capsys.readouterr()
    assert "converged: True" in captured.out
    assert "node_counts:" in captured.out
