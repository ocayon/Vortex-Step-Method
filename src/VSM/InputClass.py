import dataclasses
import numpy as np


@dataclasses
class Wing:
    n_panels: int
    spanwise_panel_distribution: str
    Section: list  # child-class


@dataclasses
class Section:
    LE_point: np.array
    TE_point: np.array
    CL_alpha: np.array
    CD_alpha: np.array
    CM_alpha: np.array
