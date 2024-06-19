import numpy as np
from VSM.functions_needed_for_solve import *
from VSM.panels import Panels
from VSM.horshoe_vortex import HorshoeVortex


class VortexStepMethod:
    def __init__(
        self,
        wings: list,
        initial_gamma_distribution: str = "elliptic",
        ring_geometry: str = "5fil",
        aerodynamic_model_type: str = "VSM",
        density: float = 1.225,
        max_iterations: int = 1000,
        allowed_error: float = 1e-5,
        relaxation_factor: float = 0.03,
        artificial_damping: dict = {"k2": 0.0, "k4": 0.0},
    ):
        """Constructor for the Vortex Step Method.

        Args:
            wings (list): List of WingProperties objects, that are each data-classes
            initial_gamma_distribution (str, optional): Initial gamma distribution. Defaults to "elliptic".
            ring_geometry (str, optional): Ring geometry. Defaults to "5fil".
            aerodynamic_model_type (str, optional): Aerodynamic model type. Defaults to "VSM".
            density (float, optional): Air density. Defaults to 1.225.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
            allowed_error (float, optional): Allowed error. Defaults to 1e-5.
            relaxation_factor (float, optional): Relaxation factor. Defaults to 0.03.
            artificial_damping (dict, optional): Artificial damping. Defaults to {"k2": 0.0, "k4": 0.0}.

        Returns:
            None
        """
        self.wings = wings
        self.initial_gamma_distribution = initial_gamma_distribution
        self.ring_geometry = ring_geometry
        self.aerodynamic_model_type = aerodynamic_model_type
        self.density = density
        self.max_iterations = max_iterations
        self.allowed_error = allowed_error
        self.relaxation_factor = relaxation_factor
        self.artificial_damping = artificial_damping
        self.__horshoe_vortices = []

    def instantiate_horshoe_vortices(self, wing_properties: list):
        """Instantiates the horshoe vortices for the given wing properties.

        Args:
            wing_properties (list): List of WingProperties objects.

        Returns:
            None
        """
        self.__horshoe_vortices = []
        for wing_properties in self.wings:
            panels = Panels(wing_properties)
            for panel_properties in panels:
                self.__horshoe_vortices.append(HorshoeVortex(panel_properties))

            self.__horshoe_vortices.update_horshoe_vortices_for_gamma_distribution(
                self.__initial_gamma_distribution
            )

    def update_horshoe_vortices_for_gamma_distribution(
        self, gamma_distribution: np.ndarray
    ):
        """Updates the horshoe vortices for the given gamma distribution.

        Args:
            gamma_distribution (np.ndarray): The gamma distribution to be updated.

        Returns:
            None
        """
        if gamma_distribution == "elliptic":
            self.__update_horshoe_vortices_for_eliptic_distribution()
        else:
            self.__update_horshoe_vortices_for_given_distribution(gamma_distribution)

    def update_horshoe_vortices_for_va(self, va: list):
        """Updates the horshoe vortices for the given va.

        Args:
            va (list): List of va values.

        Returns:
            None
        """
        if len(va) == 1:
            for i, horshoe in enumerate(self.__horshoe_vortices):
                self.__horshoe_vortices[i].update_for_va(va[0])
        elif len(va) == len(self.__horshoe_vortices):
            for i, horshoe in enumerate(self.__horshoe_vortices):
                self.__horshoe_vortices[i].update_for_va(va[i])
        else:
            raise ValueError(
                "The number of va values should be either 1 or equal to the number of horshoe vortices."
            )

    def __calculate_AIC_matrix(self):
        """Calculates the AIC matrix for the given horshoe vortices."""
        pass

    def solve(self, va: list, yaw_rate: float = 0):
        """Solves the Vortex Step Method."""
        # 1. Update the horshoe vortices for the given va
        self.update_horshoe_vortices_for_va(va)

        # 2. Calculate the AIC matrix
        self.__calculate_AIC_matrix()

        # 3. Iteratively find the gamma distribution
        # TODO: parse these as seperate values, or as self, rather than as a dictionary
        convergence_criteria = {
            "Niterations": self.max_iterations,
            "error": self.allowed_error,
            "Relax_factor": self.relaxation_factor,
        }
        # TODO: Implement optionality for va_distribution, rather a single va value
        Uinf = va[0]
        Fmag, Gamma, aero_coeffs = solve_lifting_line_system_matrix_approach_art_visc(
            ringvec,
            controlpoints,
            rings,
            Uinf,
            Gamma0,
            data_airf,
            convergence_criteria,
            self.aerodynamic_model_type,
            self.density,
        )
        # 4. Calculate the global output

        pass
