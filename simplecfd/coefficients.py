from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MomentumCoefficients:
    a_w: np.ndarray
    a_e: np.ndarray
    a_p: np.ndarray
    source: np.ndarray
    d: np.ndarray
    f_w: np.ndarray
    f_e: np.ndarray


@dataclass
class PressureCorrectionCoefficients:
    a_w: np.ndarray
    a_e: np.ndarray
    a_p: np.ndarray
    source: np.ndarray


@dataclass
class LinearSystem:
    lower: np.ndarray
    diagonal: np.ndarray
    upper: np.ndarray
    rhs: np.ndarray
