import numpy as np
from VSM.Panel import Panel


# TODO: should change name to deal with multiple wings
class WingAerodynamics:
    def __init__(
        self,
        wings: list,  # List of Wing object instances
        initial_gamma_distribution: str = "elliptic",
        ring_geometry: str = "5fil",
    ):
        """
        A class to represent a vortex system.
        """
        self.panels = np.array([])
        for wing_instance in wings:
            sections = wing_instance.refine_aerodynamic_mesh()
            for i in range(len(sections) - 1):
                np.append(self.panels, Panel(sections[i], sections[i + 1]))
        self.va = None

    def calculate_AIC_matrices(self, model: str = "VSM"):

        N = len(self.panels)
        U_2D = np.empty((N, 3))
        MatrixU = np.empty((N, N))
        MatrixV = np.empty((N, N))
        MatrixW = np.empty((N, N))

        for icp, panel_icp in enumerate(self.panels):

            if model == "VSM":
                # Velocity induced by a infinte bound vortex with Gamma = 1
                # TODO: verify that get_ringvec is defined within Panel

                U_2D[icp] = self.calculate_U2D_matrix(
                    panel_icp, panel_icp.get_ringvec()
                )

            elif model != "LLT":
                raise ValueError("Invalid aerodynamic model type, should be LTT or VSM")

            for jring, panel_jring in enumerate(self.panels):

                # TODO: verify that calculate_induced velocity contains CORE correction
                # TODO: verify that get_control_point is defined within Panel
                velocity_induced = panel_jring.calculate_induced_velocity(
                    panel_icp.get_control_point(), strength=1
                )
                # AIC Matrix
                MatrixU[icp, jring] = velocity_induced[0]
                MatrixV[icp, jring] = velocity_induced[1]
                MatrixW[icp, jring] = velocity_induced[2]

        return MatrixU, MatrixV, MatrixW, U_2D

    def calculate_U2D_matrix(ring_vec):
        # TODO: verify that velocity_induced_bound_2D is defined within Panel
        r0 = ringvec["r0"]
        r3 = ringvec["r3"]

        cross = [
            r0[1] * r3[2] - r0[2] * r3[1],
            r0[2] * r3[0] - r0[0] * r3[2],
            r0[0] * r3[1] - r0[1] * r3[0],
        ]

        ind_vel = (
            cross
            / (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
            / 2
            / np.pi
            * np.linalg.norm(r0)
        )

        return ind_vel

    def get_panels(self):
        pass

    def get_gamma_distribution(self):
        pass

    def set_gamma_distribution(self, gamma):
        pass

    def calculate_effective_angle_of_attack(self, aerodynamic_center):
        # Placeholder for actual implementation
        pass

    def calculate_aerodynamic_coefficients(self, alpha):
        # Placeholder for actual implementation
        pass

    def calculate_aerodynamic_forces(self, alpha, cl, cd, cm):
        # Placeholder for actual implementation
        pass

    def get_wing_coefficients(self):
        # Placeholder for actual implementation
        return {"CL": 0.0, "CD": 0.0, "CS": 0.0, "CM": 0.0}

    def update_wake(self, va_distribution):
        # Placeholder for actual implementation
        pass

    def set_va(self, va, yaw_rate):
        self.va = va
        self.yaw_rate = yaw_rate
        # Make n panels, 3 array of the list va
        va_distribution = np.repeat([va], len(self.panels), axis=0)
        for i, panel in enumerate(self.panels):
            panel.update_va(va_distribution[i])

        self.update_wake(va_distribution)  # Define the trailing wake filaments
