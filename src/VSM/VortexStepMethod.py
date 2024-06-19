import numpy as np

from VSM.functions_needed_for_solve import *
from VSM.PanelDistribution import PanelDistribution
from VSM.HorshoeVortex import HorshoeVortex

class VortexStepMethod:
    def __init__(
        self,
        wings: list, # List of Wing object instances
        # Below are all settings, with a default value, that can but don't have to be changed
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


    # va,yaw_rate are input here such that one can easily run loops over different va values
    def simulate(self, va: list, yaw_rate: float = 0):
        """Runs the Vortex Step Method.

        Args:
            va (list): List of va values.
            yaw_rate (float, optional): Yaw rate. Defaults to 0.
            
        Returns:
            None
        """
        # 0. Instantiate the horshoe vortices
        self.instantiate_horshoe_vortices(self.wings)

        # 2. Update the horshoe vortices for the given va
        self.update_horshoe_vortices_for_va(va)

        # 3. Calculate the AIC matrix
        self.__calculate_AIC_matrix()

        # 4. Iteratively find the gamma distribution
        self.solve_lifting_line_system_matrix_approach_art_visc(va)

        # 5. Calculate the global output
        self.calculate_global_output()


    # 1. Instantiate the horshoe vortices
    def instantiate_horshoe_vortices(self):
        """Instantiates the horshoe vortices for the given wing properties.

        Args:
            wing_properties (list): List of WingProperties objects.

        Returns:
            None
        """
        # Loop through the list of wings
        for wing_instance in self.wings:

            # Distribute the panels of the wing_instance

            # Panel Object
                # 4 corner_points
                # aerodynamic properties
                # tangential, normal and the other vector
                # horshoe vortex
                # control point
                # aerodynamic center

            sections = wing_instance.refine_aerodynamic_mesh()
            for i in range(len(sections)-1):
                self.panels.append(Panel(section[i],section[i+1]))

            # Update the horshoe vortices for the initial gamma distribution
            self.update_horshoe_vortices_for_gamma_distribution(
                self.initial_gamma_distribution
            )

    def update_horshoe_vortices_for_gamma_distribution(
        self, gamma_distribution: np.ndarray = "elliptic"
    ):
        """Updates the horshoe vortices for the given gamma distribution.

        
            This is a seperate function as it might needing an update when not converging

        Args:
            gamma_distribution (np.ndarray): The gamma distribution to be updated.

        Returns:
            None
        """
        if gamma_distribution == "elliptic":
            #update the horshoe vortices for elliptic distribution
        else:
            #update the horshoe vortices for the given distribution

    # 2. Update the horshoe vortices for the given va
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

    # 3. Calculate the AIC matrix
    def __calculate_AIC_matrix(self):
        """Calculates the AIC matrix for the given horshoe vortices."""
        pass

    # 4. Iteratively find the gamma distribution
    def solve_lifting_line_system_matrix_approach_art_visc(self, va: list, yaw_rate: float = 0):
        """
        Solves the lifting line system using the matrix approach with artificial viscosity.

        Args:
            va (list): List of va values.
            yaw_rate (float, optional): Yaw rate. Defaults to 0.
        
        Returns:
            None
        """

        # TODO: parse these as seperate values, or as self, rather than as a dictionary
        convergence_criteria = {
            "Niterations": self.max_iterations,
            "error": self.allowed_error,
            "Relax_factor": self.relaxation_factor,
        }
        # TODO: Implement optionality for va_distribution, rather a single va value
        Uinf = va[0]
        self.Fmag, self.Gamma, self.aero_coeffs = solve_lifting_line_system_matrix_approach_art_visc(
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

        pass
    
    # 5. Calculate the global output
    def calculate_global_output(self):
        """Calculates the global output."""

        # Define the parameters of interest
        self.CL_per_panel = "CL_per_panel"
        self.CD_per_panel = "CD_per_panel"
        self.CL = "CL"
        self.CD = "CD"
        
        pass