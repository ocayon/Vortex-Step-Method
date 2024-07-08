import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
import logging

from VSM.Panel import Panel
import VSM.Wake as Wake

logging.basicConfig(level=logging.INFO)


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

        # TODO: later Wake should be a class
        # Add the frozen wake elements based on the va distribution
        self.panels = Wake.frozen_wake(va_distribution, self.panels)

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
                velocity_induced = panel_jring.calculate_velocity_induced_horseshoe(
                    getattr(panel_icp, evaluation_point), gamma=1
                )
                # AIC Matrix
                MatrixU[icp, jring] = velocity_induced[0]
                MatrixV[icp, jring] = velocity_induced[1]
                MatrixW[icp, jring] = velocity_induced[2]

                if icp == jring:
                    if evaluation_point != "aerodynamic_center":
                        U_2D = panel_jring.calculate_velocity_induced_bound_2D(
                            getattr(panel_icp, evaluation_point), gamma=1
                        )
                        MatrixU[icp, jring] -= U_2D[0]
                        MatrixV[icp, jring] -= U_2D[1]
                        MatrixW[icp, jring] -= U_2D[2]

        return MatrixU, MatrixV, MatrixW

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

        return gamma_i * 0

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

    def calculate_results(self, density):
        """Update spanwise local values and calculate the aerodynamics of the wing

        Args:
            None

        Returns:
            results_dict (dict): Dictionary containing the aerodynamic results of the wing
            wing_aero (WingAerodynamics): The updated WingAerodynamics object
        """

        # Update the aerodynamic coefficients of each panel
        cl_array, cd_array, cm_array = (
            np.zeros(len(self.panels)),
            np.zeros(len(self.panels)),
            np.zeros(len(self.panels)),
        )
        for i, panel_i in enumerate(self.panels):
            cl = panel_i.calculate_cl(self._alpha_aerodynamic_center[i])
            cd, cm = panel_i.calculate_cd_cm(self._alpha_aerodynamic_center[i])
            cl_array[i] = cl
            cd_array[i] = cd
            cm_array[i] = cm

        # Calculate the global aerodynamics of the wing
        F_rel = []
        F_gl = []
        Fmag_gl = []
        SideF = []
        Ltot = 0
        Dtot = 0
        SFtot = 0

        Atot = 0
        if len(self._va) == 3:
            Uinf = self._va
        else:
            raise ValueError("Calc.results not ready for va_distributed input")

        for i, (alpha_i, panel_i) in enumerate(
            zip(self._alpha_aerodynamic_center, self.panels)
        ):

            # Defining directions wrt airfoils
            into_plane = panel_i.z_airf  # along the span
            tangential = panel_i.y_airf  # along the chord
            normal = panel_i.x_airf  # normal to the chord

            r_0 = into_plane  # bound["x2"] - bound["x1"]

            # Relative wind speed direction
            tangential = panel_i.y_airf
            normal = panel_i.x_airf
            dir_urel = np.cos(alpha_i) * tangential + np.sin(alpha_i) * normal
            dir_urel = dir_urel / np.linalg.norm(dir_urel)

            # Lift direction relative to urel
            dir_L = np.cross(dir_urel, r_0)
            dir_L = dir_L / np.linalg.norm(dir_L)

            # TODO: why is the [0,1,0] hardcode needed?
            # Drag direction relative to urel
            dir_D = np.cross([0, 1, 0], dir_L)
            dir_D = dir_D / np.linalg.norm(dir_D)

            # TODO: old code the L,D,M was calculated with non-corrected alpha, why?
            # TODO: change would be to swap self._alpha to self._alpha_aerodynamic_center
            # Lift and drag relative to urel
            q_inf = 0.5 * density * np.linalg.norm(Uinf) ** 2 * panel_i.chord
            L_rel = dir_L * panel_i.calculate_cl(self._alpha_uncorrected[i]) * q_inf
            D_rel = (
                dir_D * panel_i.calculate_cd_cm(self._alpha_uncorrected[i])[0] * q_inf
            )
            F_rel.append([L_rel + D_rel])

            # Lift direction relative to the wind speed
            dir_L_gl = np.cross(Uinf, [0, 1, 0])
            dir_L_gl = dir_L_gl / np.linalg.norm(dir_L_gl)

            def vector_projection(v, u):
                # Inputs:
                #     u = direction vector
                #     v = vector to be projected
                unit_u = u / np.linalg.norm(u)
                proj = np.dot(v, unit_u) * unit_u
                return proj

            # Lift and drag relative to the windspeed

            L_gl = vector_projection(L_rel, dir_L_gl) + vector_projection(
                D_rel, dir_L_gl
            )
            D_gl = vector_projection(L_rel, Uinf) + vector_projection(D_rel, Uinf)
            F_gl.append([L_gl, D_gl])
            Fmag_gl.append(
                [
                    np.dot(L_rel, dir_L_gl) + np.dot(D_rel, dir_L_gl),
                    np.dot(L_rel, Uinf / np.linalg.norm(Uinf))
                    + np.dot(D_rel, Uinf / np.linalg.norm(Uinf)),
                ]
            )
            SideF.append(np.dot(L_rel, [0, 1, 0]) + np.dot(D_rel, [0, 1, 0]))

            # Calculate Area of the panel
            Atot += panel_i.chord * np.linalg.norm(panel_i.z_airf)

        # Calculate total aerodynamic forces
        for i, Fmag_gl_i in enumerate(Fmag_gl):
            z_airf_i = self.panels[i].z_airf
            r0_length = np.linalg.norm(z_airf_i)

            Ltot += Fmag_gl_i[0] * r0_length
            Dtot += Fmag_gl_i[1] * r0_length
            SFtot += SideF[i] * r0_length

        Umag = np.linalg.norm(Uinf)
        CL = Ltot / (0.5 * Umag**2 * Atot * density)
        CD = Dtot / (0.5 * Umag**2 * Atot * density)
        CS = SFtot / (0.5 * Umag**2 * Atot * density)

        results_dict = {}
        results_dict.update([("cl", cl_array)])
        results_dict.update([("cd", cd_array)])
        results_dict.update([("cm", cm_array)])

        # Calculate global aerodynamics
        results_dict.update([("cl_wing", -CL)])
        results_dict.update([("cd_wing", -CD)])
        results_dict.update([("cs_wing", -CS)])
        results_dict.update([("cmx_wing", 0.0)])
        results_dict.update([("cmy_wing", 0.0)])
        results_dict.update([("cmz_wing", 0.0)])
        results_dict.update([("lift_wing", Ltot)])
        results_dict.update([("drag_wing", Dtot)])
        results_dict.update([("side_wing", SFtot)])
        results_dict.update([("mx_wing", 0.0)])
        results_dict.update([("my_wing", 0.0)])
        results_dict.update([("mz_wing", 0.0)])

        # Additional info
        results_dict.update([("alpha_at_ac", self._alpha_aerodynamic_center)])
        results_dict.update([("alpha_uncorrected", self._alpha_uncorrected)])
        results_dict.update([("gamma_distribution", self._gamma_distribution)])

        return results_dict

    ###########################
    ## UPDATE FUNCTIONS
    ###########################

    def update_effective_angle_of_attack(self, alpha, aerodynamic_model_type):
        """Updates the angle of attack at the aerodynamic center of each panel,
            Calculated at the AERODYNAMIC CENTER, which needs an update for VSM
            And can just use the old value for the LLT

        Args:
            None

        Returns:
            None
        """
        self._alpha_uncorrected = alpha
        # If VSM
        if aerodynamic_model_type == "VSM":
            # Initialize the matrices, WITHOUT U2D CORRECTION
            # The correction is done by calculating the alpha at the aerodynamic center,
            # where as before the control_point was used in the VSM method
            n_panels = self._n_panels
            MatrixU = np.empty((n_panels, n_panels))
            MatrixV = np.empty((n_panels, n_panels))
            MatrixW = np.empty((n_panels, n_panels))

            evaluation_point = "aerodynamic_center"
            for icp, panel_icp in enumerate(self.panels):

                for jring, panel_jring in enumerate(self.panels):
                    velocity_induced = panel_jring.calculate_velocity_induced_horseshoe(
                        getattr(panel_icp, evaluation_point), gamma=1
                    )
                    # AIC Matrix,WITHOUT U2D CORRECTION
                    MatrixU[icp, jring] = velocity_induced[0]
                    MatrixV[icp, jring] = velocity_induced[1]
                    MatrixW[icp, jring] = velocity_induced[2]

            gamma = self._gamma_distribution
            alpha = np.zeros(len(self.panels))
            for icp, panel in enumerate(self.panels):
                # Initialize induced velocity to 0
                u = 0
                v = 0
                w = 0
                # Compute induced velocities with previous gamma distribution
                for jring, gamma_jring in enumerate(gamma):
                    u = u + MatrixU[icp][jring] * gamma_jring
                    # x-component of velocity
                    v = v + MatrixV[icp][jring] * gamma_jring
                    # y-component of velocity
                    w = w + MatrixW[icp][jring] * gamma_jring
                    # z-component of velocity

                induced_velocity = np.array([u, v, w])
                alpha[icp], _ = panel.calculate_relative_alpha_and_relative_velocity(
                    induced_velocity
                )

            self._alpha_aerodynamic_center = alpha

        # if LTT: no updating is required
        elif aerodynamic_model_type == "LLT":
            self._alpha_aerodynamic_center = alpha
        else:
            raise ValueError("Invalid aerodynamic model type, should be VSM or LLT")

    ###########################
    ## PLOTTING FUNCTIONS
    ###########################

    def plot_line_segment(self, ax, segment, color, label, width: float = 3):
        ax.plot(
            [segment[0][0], segment[1][0]],
            [segment[0][1], segment[1][1]],
            [segment[0][2], segment[1][2]],
            color=color,
            label=label,
            linewidth=width,
        )
        dir = segment[1] - segment[0]
        ax.quiver(
            segment[0][0],
            segment[0][1],
            segment[0][2],
            dir[0],
            dir[1],
            dir[2],
            color=color,
        )

    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max([x_range, y_range, z_range])

        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)

        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

    def plot(self):
        """
        Plots the wing panels and filaments in 3D.

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
        for i, panel in enumerate(self.panels):
            # Get the corner points of the current panel and close the loop by adding the first point again
            x_corners = np.append(corner_points[i, :, 0], corner_points[i, 0, 0])
            y_corners = np.append(corner_points[i, :, 1], corner_points[i, 0, 1])
            z_corners = np.append(corner_points[i, :, 2], corner_points[i, 0, 2])

            # Plot the panel edges
            ax.plot(
                x_corners,
                y_corners,
                z_corners,
                color="grey",
                label="Panel Edges" if i == 0 else "",
                linewidth=1,
            )

            # Create a list of tuples representing the vertices of the polygon
            verts = [list(zip(x_corners, y_corners, z_corners))]
            poly = Poly3DCollection(verts, color="grey", alpha=0.1)
            ax.add_collection3d(poly)

            # Plot the control point
            ax.scatter(
                control_points[i, 0],
                control_points[i, 1],
                control_points[i, 2],
                color="green",
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

            # Plot the filaments
            filaments = panel.calculate_filaments_for_plotting()
            legends = ["Bound Vortex", "side1", "side2", "wake_1", "wake_2"]

            for filament, legend in zip(filaments, legends):
                x1, x2, color = filament
                logging.info("Legend: %s", legend)
                self.plot_line_segment(ax, [x1, x2], color, legend)

        # Add legends for the first occurrence of each label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # Set equal axis limits
        self.set_axes_equal(ax)

        # Flip the z-axis (to stick to body reference frame)
        ax.invert_zaxis()

        # Display the plot
        plt.show()
