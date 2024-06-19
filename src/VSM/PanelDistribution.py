import numpy as np

class PanelDistribution:
    """
    This class is responsible for creating a list of Panel objects
    Args:
        wing_instance (Wing) containing:
        - n_panels
        - spanwise_panel_distribution
        - sections
            - LE point
            - TE point
            - CL-alpha
            - CD-alpha
            - CM-alpha

    Returns:
        panels (list of Panel objects)

    """

    def __init__(self) -> None:
        pass

    def main(self, wing_properties):
        """
        Input is the wing_properties
        1. transform the wing geometry into n_panels,
            using the defined spanwise_panel_distribution
            and finds the corner points of each.
        2. Using the provided aerodynamic properties
            for each section, calculates the aerodynamic
            properties for each panel by integrating
        """
        wing_panel_corner_points = self.__compute_wing_panel_distribution()
        return self.__compute_panel_aerodynamic_properties(wing_panel_corner_points

    def __compute_wing_panel_corner_points(self):
        """
        Computes the chordwise section of the wing
        Using
        - self.n_panels
        - self.spanwise_panel_distribution
        - self.sections
        """
        wing_panel_corner_points = ["panel_corner_points_1","panel_corner_points_2"]]
        return wing_panel_corner_points
    
    def __compute_panel_aerodynamic_properties(self,wing_panel_corner_points):
        """
        Computes the aerodynamic properties of each panel
        Using
        - self.sections
        """
        panel_aerodynamic_properties = ["CL-alpha", "CD-alpha", "CM-alpha"]

        return np.append(wing_panel_corner_points, panel_aerodynamic_properties, axis=1)
