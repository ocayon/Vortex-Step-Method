import numpy as np
from VSM.Panel import Panel
from dataclasses import dataclass, field


# TODO: should change name to deal with multiple wings
@dataclass
class WingAerodynamics:
    def __init__(
        self,
        wings: list,  # List of Wing object instances
        initial_gamma_distribution: str = "elliptic",
    ):
        """
        A class to represent a vortex system.
        """
        panels = []
        n_panels_per_wing = np.empty(len(wings))
        n_panels = 0
        for i, wing_instance in enumerate(wings):
            sections = wing_instance.refine_aerodynamic_mesh()
            for j in range(len(sections) - 1):
                panels.append(Panel(sections[j], sections[j + 1]))
            # adding the number of panels of each wing
            n_panels += wing_instance.n_panels
            # calculating the number of panels per wing
            n_panels_per_wing[i] = len(sections)

        self.wings = wings
        self._panels = panels
        self._n_panels_per_wing = n_panels_per_wing
        self.n_panels = n_panels
        self.va = None
        self.alpha_aerodynamic_center = np.empty(n_panels)
        self.alpha_control_point = np.empty(n_panels)
        self.cl = np.empty(n_panels)
        self.cd = np.empty(n_panels)
        self.cm = np.empty(n_panels)
        self.initial_gamma_distribution = initial_gamma_distribution

    ###########################
    ## CALCULATE FUNCTIONS
    ###########################

    # TODO: this method should be properly tested against the old code and analytics
    def calculate_AIC_matrices(self, model: str = "VSM"):
        """Calculates the AIC matrices for the given aerodynamic model

        Args:
            model (str): The aerodynamic model to be used, either VSM or LLT

        Returns:
            MatrixU (np.array): The x-component of the AIC matrix
            MatrixV (np.array): The y-component of the AIC matrix
            MatrixW (np.array): The z-component of the AIC matrix
            U_2D (np.array): The 2D velocity induced by a bound vortex
        """

        n_panels = self.n_panels
        U_2D = U_2D = np.array([0, 0, 0])
        MatrixU = np.empty((n_panels, n_panels))
        MatrixV = np.empty((n_panels, n_panels))
        MatrixW = np.empty((n_panels, n_panels))

        if model == "VSM":
            evaluation_point = "control_point"
        elif model == "LLT":
            evaluation_point = "aerodynamic_center"
        else:
            raise ValueError("Invalid aerodynamic model type, should be VSM or LLT")

        for icp, panel_icp in enumerate(self.panels):

            for jring, panel_jring in enumerate(self.panels):
                velocity_induced = panel_jring.calculate_velocity_induced(
                    getattr(panel_icp, evaluation_point), gamma_mag=1
                )
                if icp == jring:
                    U_2D = panel_jring.calculate_velocity_induced_bound_2D(
                        getattr(panel_icp, evaluation_point), gamma_mag=1
                    )
                # AIC Matrix
                MatrixU[icp, jring] = velocity_induced[0] + U_2D[0]
                MatrixV[icp, jring] = velocity_induced[1] + U_2D[1]
                MatrixW[icp, jring] = velocity_induced[2] + U_2D[2]

        return MatrixU, MatrixV, MatrixW, U_2D

    def calculate_circulation_distribution_elliptical_wing(self, gamma_0: float = 1):
        """
        Calculates the circulation distribution for an elliptical wing.

        Args:
            gamma_0 (float): The circulation at the wing root

        Returns:
            np.array: The circulation distribution
        """
        print(f"gamma_O: {gamma_0},type: {type(gamma_0)}")
        gamma_i = np.array([])
        # Calculating the wing_span from the panels
        for i, (wing_instance, n_panels) in enumerate(
            zip(self.wings, self._n_panels_per_wing)
        ):
            # calculating the wing-span of each wing
            wing_span = wing_instance.calculate_wing_span()

            y = np.linspace(-wing_span / 2, wing_span / 2, int(n_panels - 1))
            gamma_i_wing = gamma_0 * np.sqrt(1 - (2 * y / wing_span) ** 2)
            gamma_i = np.append(gamma_i, gamma_i_wing)

        return gamma_i

    def calculate_gamma_distribution(self, gamma_distribution=None):
        """Calculates the circulation distribution for the wing

        Args:
            gamma_distribution (np.array): The circulation distribution to be used

        Returns:
            np.array: The circulation distribution
        """
        if self.initial_gamma_distribution == "elliptic":
            return self.calculate_circulation_distribution_elliptical_wing()
        elif len(gamma_distribution) == self.n_panels:
            return gamma_distribution
        else:
            raise ValueError(
                f"Invalid gamma distribution, len(gamma_distribution) :{len(gamma_distribution)} != self.n_panels:{self.n_panels}"
            )

    def calculate_relative_alpha_and_relative_velocity(
        self, induced_velocity: np.array
    ):
        return self.panels.calculate_relative_alpha_and_relative_velocity(
            self, induced_velocity
        )

    ###########################
    ## GETTER FUNCTIONS
    ###########################

    @property
    def panels(self):
        return self._panels

    @property
    def spanwise_panel_distribution(self):
        return {
            "alpha_aerodynamic_center": self.alpha_aerodynamic_center,
            "alpha_control_point": self.alpha_control_point,
            "cl": self.cl,
            "cd": self.cd,
            "cm": self.cm,
        }

    ###########################
    ## SETTER FUNCTIONS
    ###########################

    @panels.setter
    def panels(self, value):
        self._panels = value

    # TODO: needs work
    def set_va(self, va, yaw_rate):
        self.va = va
        self.yaw_rate = yaw_rate
        # Make n panels, 3 array of the list va
        va_distribution = np.repeat([va], len(self.panels), axis=0)
        for i, panel in enumerate(self.panels):
            panel.va = va_distribution[i]
        self.update_wake(va_distribution)  # Define the trailing wake filaments

    def update_effective_angle_of_attack(self):
        """Updates the angle of attack at the aerodynamic center of each panel,
            Calculated at the AERODYNAMIC CENTER

        Args:
            None

        Returns:
            None
        """
        for i, panel_i in enumerate(self.panels):
            induced_velocity = self.wing_induced_velocity(panel_i.aerodynamic_center)
            self.alpha_aerodynamic_center[i], _ = (
                panel_i.calculate_relative_alpha_and_relative_velocity(induced_velocity)
            )

    def update_aerodynamic_coefficients_and_alpha(self):
        """Updates the aerodynamic coefficients of each panel,
            Calculated at the CONTROL POINT

        Args:
            None

        Returns:
            None
        """
        for i, panel_i in enumerate(self.panels):
            induced_velocity = self.wing_induced_velocity(panel_i.control_point)
            alpha_i, _ = panel_i.calculate_relative_alpha_and_relative_velocity(
                induced_velocity
            )
            self.cl[i], self.cd[i], self.cm[i] = (
                panel_i.calculate_aerodynamic_coefficients(alpha_i)
            )
            self.alpha_control_point[i] = alpha_i

    # TODO: Needs Work
    def wing_induced_velocity(self, point):
        # Placeholder for actual implementation
        induced_velocity = np.array([0, 0, 0])
        return induced_velocity

    # TODO: Needs Work
    def update_global_aerodynamics(self):
        pass

    # TODO: Needs Work
    def update_wake(self, va_distribution):
        # Placeholder for actual implementation
        pass

    ###########################
    ## below is not yet needed?
    ###########################

    def calculate_aerodynamic_forces(self, alpha, cl, cd, cm):
        # Placeholder for actual implementation
        pass

    def get_wing_coefficients(self):
        # Placeholder for actual implementation
        return {"CL": 0.0, "CD": 0.0, "CS": 0.0, "CM": 0.0}
