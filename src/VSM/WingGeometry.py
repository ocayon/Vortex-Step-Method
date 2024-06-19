import dataclasses
import numpy as np
from typing import List

@dataclasses
class Wing:
    n_panels: int
    spanwise_panel_distribution: str = "linear"
    Section: List[Section]  # child-class

    def refine_aerodynamic_mesh(self):
        #3 user defined Section
        #to 25
        pass
        # return refined_sections #list of updated sections

@dataclasses
class Section:
    LE_point: np.array
    TE_point: np.array
    CL_alpha: np.array
    CD_alpha: np.array
    CM_alpha: np.array




######################
#### IDEATION ON FUNCTIONS THAT WILL BE USEFULL FOR THE REFINE_AERODYNAMIC_MESH FUNCTION
######################

def calculate_panel_distribution(self):
    """
    Input is the wing_instance of the Wing object
    1. transform the wing geometry into n_panels,
        using the defined spanwise_panel_distribution
        and finds the corner points of each.
    2. Using the provided aerodynamic properties
        for each section, calculates the aerodynamic
        properties for each panel by integrating
    """
    wing_panel_corner_points = self.__compute_wing_panel_distribution()
    return self.__compute_panel_aerodynamic_coefficients(wing_panel_corner_points)

def __compute_wing_panel_corner_points(self):
    """
    Computes the chordwise section of the wing
    Using
    - self.n_panels
    - self.spanwise_panel_distribution
    - self.sections
    """
    wing_panel_corner_points = ["panel_corner_points_1","panel_corner_points_2"]
    return wing_panel_corner_points

def __compute_panel_aerodynamic_coefficients(self,sections):
    """
    Computes the aerodynamic properties of each panel
    Using
    - self.sections
    """
    wing_panel_corner_points = np.concatenate([[section.LE_point, section.TE_point] for section in sections])
    panel_aerodynamic_properties = ["CL-alpha", "CD-alpha", "CM-alpha"]
    return np.append(wing_panel_corner_points, panel_aerodynamic_properties, axis=1)