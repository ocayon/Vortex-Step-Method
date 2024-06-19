class Panels(Panel):
    """
    This class is responsible for creating a list of Panel objects
    Args:
        wing_properties (wing_properties):
        a dataclass consistent of the following attributes:
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

        panels = []  # a list of Panel objects

        return panels


class Panel:
    pass
