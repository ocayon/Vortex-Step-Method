import numpy as np
from VSM.Panel import Panel


# TODO: should change name to deal with multiple wings
class WingAerodynamics:
    def __init__(
        self,
        wings: list,  # List of Wing object instances
        initial_gamma_distribution: str = "elliptic",
    ):
        """
        A class to represent a vortex system.
        """
        self.panels = np.array([])
        n_panels = 0
        for wing_instance in wings:
            sections = wing_instance.refine_aerodynamic_mesh()
            for i in range(len(sections) - 1):
                np.append(self.panels, Panel(sections[i], sections[i + 1]))
            # adding the number of panels of each wing
            n_panels += wing_instance.get_n_panels()

        self.n_panels = n_panels
        self.va = None
        self.alpha_aerodynamic_center = np.empty(n_panels)
        self.alpha_control_point = np.empty(n_panels)
        self.cl = np.empty(n_panels)
        self.cd = np.empty(n_panels)
        self.cm = np.empty(n_panels)
        self.set_gamma_distribution(initial_gamma_distribution)

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
        U_2D = np.empty((n_panels, 3))
        MatrixU = np.empty((n_panels, n_panels))
        MatrixV = np.empty((n_panels, n_panels))
        MatrixW = np.empty((n_panels, n_panels))

        # TODO: Should be able to include U_2D inside the AIC matrices
        for icp, panel_icp in enumerate(self.panels):

            if model == "VSM":
                # Velocity induced by a infinite bound vortex with Gamma = 1
                U_2D[icp] = panel_icp.calculate_velocity_induced_bound_2D()

            elif model != "LLT":
                raise ValueError("Invalid aerodynamic model type, should be LLT or VSM")

            for jring, panel_jring in enumerate(self.panels):
                # TODO: verify that calculate_velocity_induced contains CORE correction
                velocity_induced = panel_jring.calculate_velocity_induced(
                    panel_icp.control_point, strength=1
                )
                # AIC Matrix
                MatrixU[icp, jring] = velocity_induced[0]
                MatrixV[icp, jring] = velocity_induced[1]
                MatrixW[icp, jring] = velocity_induced[2]

        return MatrixU, MatrixV, MatrixW, U_2D

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

    # TODO: needs work
    def calculate_aerodynamic_coefficients(self, alpha):
        # Placeholder for actual implementation
        pass

    def update_wake(self, va_distribution):
        # Placeholder for actual implementation
        pass

    def get_panels(self):
        pass

    def get_gamma_distribution(self):
        pass

    def set_gamma_distribution(self, gamma):
        pass

    def calculate_aerodynamic_forces(self, alpha, cl, cd, cm):
        # Placeholder for actual implementation
        pass

    def get_wing_coefficients(self):
        # Placeholder for actual implementation
        return {"CL": 0.0, "CD": 0.0, "CS": 0.0, "CM": 0.0}

    ###########################
    ## GETTER FUNCTIONS
    ###########################

    def get_spanwise_panel_distribution(self):
        return {
            "alpha_aerodynamic_center": self.alpha_aerodynamic_center,
            "alpha_control_point": self.alpha_control_point,
            "cl": self.cl,
            "cd": self.cd,
            "cm": self.cm,
        }
