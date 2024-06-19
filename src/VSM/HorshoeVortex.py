class HorshoeVortex:
    """
    A class to represent a horshoe vortex.

    input:
    a single panel object
    containing all the corner points and aerodynamic properties

    output:
    a horshoe vortex object

    """

    def __init__(self):
        self.horshoe_vortex: list = []

    def update_for_va(self, va: list):
        """
        Updates the horshoe vortex for the given va.

        Args:
            va (list): List of va values.

        Returns:
            None
        """
        return self.horshoe_vortex
