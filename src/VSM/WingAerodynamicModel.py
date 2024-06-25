import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use("TkAgg")  # Replace 'TkAgg' with your preferred backend
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

        self._wings = wings
        self._panels = panels
        self._n_panels_per_wing = n_panels_per_wing
        self._n_panels = n_panels
        self._va = None
        self._initial_gamma_distribution = initial_gamma_distribution
        self._gamma_distribution = None

        # TODO: Can you not delete these attributes and only give them back to the user
        # TODO: As a result at the end of the calculation, and leave the WingAerodynamics Object Clean
        # arrays per panel
        self._alpha_aerodynamic_center = np.zeros(n_panels)
        self._alpha_control_point = np.zeros(n_panels)
        self._cl = np.zeros(n_panels)
        self._cd = np.zeros(n_panels)
        self._cm = np.zeros(n_panels)

    ###########################
    ## GETTER FUNCTIONS
    ###########################

    @property
    def panels(self):
        return self._panels

    @property
    def n_panels(self):
        return self._n_panels

    @property
    def va(self):
        return self._va

    @property
    def gamma_distribution(self):
        return self._gamma_distribution

    ###########################
    ## SETTER FUNCTIONS
    ###########################

    @panels.setter
    def panels(self, value):
        self._panels = value

    @va.setter
    def va(self, va, yaw_rate: float = 0.0):
        self._va = np.array(va)
        self._yaw_rate = yaw_rate

        if len(va) == 3:
            va_distribution = np.repeat([va], len(self.panels), axis=0)
        elif len(va) == len(self.panels):
            va_distribution = va
        else:
            raise ValueError(
                f"Invalid va distribution, len(va) :{len(va)} != len(self.panels):{len(self.panels)}"
            )
        # Update the va attribute of each panel
        for i, panel in enumerate(self.panels):
            panel.va = va_distribution[i]

        # TODO: Populate the wake class and this function
        self.update_wake(va_distribution)  # Define the trailing wake filaments

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

        n_panels = self._n_panels
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
        gamma_i = np.array([])
        # Calculating the wing_span from the panels
        for _, (wing_instance, n_panels) in enumerate(
            zip(self._wings, self._n_panels_per_wing)
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

        if (
            gamma_distribution is None
            and self._initial_gamma_distribution == "elliptic"
        ):
            gamma_distribution = (
                self.calculate_circulation_distribution_elliptical_wing()
            )
        elif (
            gamma_distribution is not None and len(gamma_distribution) != self._n_panels
        ):
            raise ValueError(
                f"Invalid gamma distribution, len(gamma_distribution) :{len(gamma_distribution)} != self._n_panels:{self._n_panels}"
            )
        self._gamma_distribution = gamma_distribution
        return gamma_distribution

    # TODO: Needs Work
    def calculate_results(self):
        """Update spanwise local values and calculate the aerodynamics of the wing

        Args:
            None

        Returns:
            results_dict (dict): Dictionary containing the aerodynamic results of the wing
            wing_aero (WingAerodynamics): The updated WingAerodynamics object
        """
        results_dict = {}

        results_dict.update(
            [("alpha_aerodynamic_center", self._alpha_aerodynamic_center)]
        )
        results_dict.update([("alpha_control_point", self._alpha_control_point)])
        results_dict.update([("cl", self._cl)])
        results_dict.update([("cd", self._cd)])
        results_dict.update([("cm", self._cm)])

        # Calculate global aerodynamics
        results_dict.update([("cl_wing", 0.0)])
        results_dict.update([("cd_wing", 0.0)])
        results_dict.update([("cs_wing", 0.0)])
        results_dict.update([("cmx_wing", 0.0)])
        results_dict.update([("cmy_wing", 0.0)])
        results_dict.update([("cmz_wing", 0.0)])
        results_dict.update([("lift_wing", 0.0)])
        results_dict.update([("drag_wing", 0.0)])
        results_dict.update([("side_wing", 0.0)])
        results_dict.update([("mx_wing", 0.0)])
        results_dict.update([("my_wing", 0.0)])
        results_dict.update([("mz_wing", 0.0)])

        return results_dict, self

    ###########################
    ## UPDATE FUNCTIONS
    ###########################

    # TODO: Needs Work
    def update_wake(self, va_distribution):
        # Placeholder for actual implementation
        pass

    def update_effective_angle_of_attack(self):
        """Updates the angle of attack at the aerodynamic center of each panel,
            Calculated at the AERODYNAMIC CENTER

        Args:
            None

        Returns:
            None
        """
        for i, panel_i in enumerate(self.panels):
            induced_velocity = self.calculate_wing_induced_velocity(
                panel_i.aerodynamic_center
            )
            self._alpha_aerodynamic_center[i], _ = (
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
            induced_velocity = self.calculate_wing_induced_velocity(
                panel_i.control_point
            )
            alpha_i, _ = panel_i.calculate_relative_alpha_and_relative_velocity(
                induced_velocity
            )
            self._cl[i], self._cd[i], self._cm[i] = panel_i.calculate_cl_cd_cm(alpha_i)
            self._alpha_control_point[i] = alpha_i

    def plot(self):
        """
        Plots the wing panels in 3D.

        Args:
            None

        Returns:
            None
        """
        # Extract corner points, control points, and aerodynamic centers from panels
        corner_points = np.array([panel.corner_points for panel in self.panels])
        control_points = np.array([panel.control_point for panel in self.panels])
        aerodynamic_centers = np.array(
            [panel.aerodynamic_center for panel in self.panels]
        )

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot each panel
        for i in range(len(self.panels)):
            # Get the corner points of the current panel and close the loop by adding the first point again
            x_corners = np.append(corner_points[i, :, 0], corner_points[i, 0, 0])
            y_corners = np.append(corner_points[i, :, 1], corner_points[i, 0, 1])
            z_corners = np.append(corner_points[i, :, 2], corner_points[i, 0, 2])

            # Plot the panel edges
            ax.plot(
                x_corners,
                y_corners,
                z_corners,
                color="k",
                label="Panel Edges" if i == 0 else "",
            )

            # Plot the control point
            ax.scatter(
                control_points[i, 0],
                control_points[i, 1],
                control_points[i, 2],
                color="r",
                label="Control Points" if i == 0 else "",
            )

            # Plot the aerodynamic center
            ax.scatter(
                aerodynamic_centers[i, 0],
                aerodynamic_centers[i, 1],
                aerodynamic_centers[i, 2],
                color="b",
                label="Aerodynamic Centers" if i == 0 else "",
            )

        # Add legends for the first occurrence of each label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # Display the plot
        plt.show()
