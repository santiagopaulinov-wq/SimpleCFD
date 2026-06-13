from __future__ import annotations

import pytest

from simplecfd.cases import VersteegExample62Case, build_versteeg_example_6_2_case


@pytest.fixture
def versteeg_example_6_2_case() -> VersteegExample62Case:
    return build_versteeg_example_6_2_case()
