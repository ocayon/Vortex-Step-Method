import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import logging

from VSM.Panel import Panel
import VSM.Wake as Wake


# TODO: should change name to deal with multiple wings
class WingAerodynamics:

    def __init__(
        self,
        wings: list,  # List of Wing object instances
    ):
        """
        A class to represent a vortex system.
        """
        panels = []
        for i, wing_instance in enumerate(wings):
            section_list = wing_instance.refine_aerodynamic_mesh()
            n_panels_per_wing = len(section_list) - 1
            logging.info(f"Number of panels: {n_panels_per_wing}")
            logging.info(f"Number of sections: {len(section_list)}")
            (
                aerodynamic_center_list,
                control_point_list,
                bound_point_1_list,
                bound_point_2_list,
                x_airf_list,
                y_airf_list,
                z_airf_list,
            ) = self.calculate_panel_properties(
                section_list,
                n_panels_per_wing,
                aerodynamic_center_location=0.25,
                control_point_location=0.75,
            )
            for j in range(n_panels_per_wing):
                panels.append(
                    Panel(
                        section_list[j],
                        section_list[j + 1],
                        aerodynamic_center_list[j],
                        control_point_list[j],
                        bound_point_1_list[j],
                        bound_point_2_list[j],
                        x_airf_list[j],
                        y_airf_list[j],
                        z_airf_list[j],
                    )
                )

        self._wings = wings
        self._panels = panels
        self._n_panels = len(panels)
        self._va = None
        self._gamma_distribution = None
        self._alpha_uncorrected = None
        self._alpha_corrected = None

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

    @property
    def wings(self):
        return self._wings

    ###########################
    ## SETTER FUNCTIONS
    ###########################

    @panels.setter
    def panels(self, value):
        self._panels = value

    @va.setter
    def va(self, va, yaw_rate: float = 0.0):

        # # Removing old wake filaments
        # self.panels = Wake.remove_frozen_wake(self.panels)

        self._va = np.array(va)
        self._yaw_rate = yaw_rate

        if yaw_rate != 0.0:
            raise ValueError("Yaw rate not yet implemented")

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

    # TODO: implement usaged of the .25 and .75 variables
    # TODO: could be CPU optimized
    def calculate_panel_properties(
        self,
        section_list,
        n_panels,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    ):
        # Initialize lists
        aerodynamic_center_list = []
        control_point_list = []
        bound_point_1_list = []
        bound_point_2_list = []
        x_airf_list = []
        y_airf_list = []
        z_airf_list = []

        # defining coordinates
        coordinates = np.zeros((2 * (n_panels + 1), 3))
        logging.debug(f"shape of coordinates: {coordinates.shape}")
        for i in range(n_panels):
            logging.debug(f"i: {i}")
            coordinates[2 * i] = section_list[i].LE_point
            coordinates[2 * i + 1] = section_list[i].TE_point
            coordinates[2 * i + 2] = section_list[i + 1].LE_point
            coordinates[2 * i + 3] = section_list[i + 1].TE_point

        logging.debug(f"coordinates: {coordinates}")

        for i in range(n_panels):
            # Identify points defining the panel
            section = {
                "p1": coordinates[2 * i, :],
                "p2": coordinates[2 * i + 2, :],
                "p3": coordinates[2 * i + 3, :],
                "p4": coordinates[2 * i + 1, :],
            }

            di = np.linalg.norm(
                coordinates[2 * i, :] * 0.75
                + coordinates[2 * i + 1, :] * 0.25
                - (coordinates[2 * i + 2, :] * 0.75 + coordinates[2 * i + 3, :] * 0.25)
            )
            if i == 0:
                diplus = np.linalg.norm(
                    coordinates[2 * (i + 1), :] * 0.75
                    + coordinates[2 * (i + 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i + 1) + 2, :] * 0.75
                        + coordinates[2 * (i + 1) + 3, :] * 0.25
                    )
                )
                ncp = di / (di + diplus)
            elif i == n_panels - 1:
                dimin = np.linalg.norm(
                    coordinates[2 * (i - 1), :] * 0.75
                    + coordinates[2 * (i - 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i - 1) + 2, :] * 0.75
                        + coordinates[2 * (i - 1) + 3, :] * 0.25
                    )
                )
                ncp = dimin / (dimin + di)
            else:
                dimin = np.linalg.norm(
                    coordinates[2 * (i - 1), :] * 0.75
                    + coordinates[2 * (i - 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i - 1) + 2, :] * 0.75
                        + coordinates[2 * (i - 1) + 3, :] * 0.25
                    )
                )
                diplus = np.linalg.norm(
                    coordinates[2 * (i + 1), :] * 0.75
                    + coordinates[2 * (i + 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i + 1) + 2, :] * 0.75
                        + coordinates[2 * (i + 1) + 3, :] * 0.25
                    )
                )
                ncp = 0.25 * (dimin / (dimin + di) + di / (di + diplus) + 1)

            ncp = 1 - ncp

            # aerodynamic center at 1/4c
            LLpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * 3 / 4 + (
                section["p3"] * (1 - ncp) + section["p4"] * ncp
            ) * 1 / 4
            # control point at 3/4c
            VSMpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * 1 / 4 + (
                section["p3"] * (1 - ncp) + section["p4"] * ncp
            ) * 3 / 4

            # Calculating the bound
            bound_1 = section["p1"] * 3 / 4 + section["p4"] * 1 / 4
            bound_2 = section["p2"] * 3 / 4 + section["p3"] * 1 / 4

            ### Calculate the local reference frame, below are all unit_vectors
            # NORMAL x_airf defined upwards from the chord-line, perpendicular to the panel
            x_airf = np.cross(VSMpoint - LLpoint, section["p2"] - section["p1"])
            x_airf = x_airf / np.linalg.norm(x_airf)

            # TANGENTIAL y_airf defined parallel to the chord-line, from LE-to-TE
            y_airf = VSMpoint - LLpoint
            y_airf = y_airf / np.linalg.norm(y_airf)

            # SPAN z_airf along the LE, in plane (towards left tip, along span) from the airfoil perspective
            z_airf = bound_2 - bound_1
            z_airf = z_airf / np.linalg.norm(z_airf)

            # Appending
            aerodynamic_center_list.append(LLpoint)
            control_point_list.append(VSMpoint)
            bound_point_1_list.append(bound_1)
            bound_point_2_list.append(bound_2)
            x_airf_list.append(x_airf)
            y_airf_list.append(y_airf)
            z_airf_list.append(z_airf)

        return (
            aerodynamic_center_list,
            control_point_list,
            bound_point_1_list,
            bound_point_2_list,
            x_airf_list,
            y_airf_list,
            z_airf_list,
        )

    # TODO: this method should be properly tested against the old code and analytics
    def calculate_AIC_matrices(self, model, core_radius_fraction):
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
                # velocity_induced = panel_jring.calculate_velocity_induced_horseshoe(
                #     getattr(panel_icp, evaluation_point),
                #     gamma=1,
                #     core_radius_fraction=core_radius_fraction,
                #     model=model,
                # )

                ####################
                import os
                import sys

                root_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                )
                sys.path.insert(0, root_path)
                from tests.WingAerodynamics.test_wing_aero_object_against_create_geometry_general import (
                    create_ring_from_wing_object,
                )
                from tests.WingAerodynamics.thesis_functions import (
                    velocity_induced_single_ring_semiinfinite,
                )

                rings = create_ring_from_wing_object(self, gamma_data=1)
                evaluation_point_list = [
                    getattr(panel_index, evaluation_point)
                    for panel_index in self.panels
                ]
                va_norm = np.linalg.norm(self.va)
                velocity_induced = velocity_induced_single_ring_semiinfinite(
                    rings[jring], evaluation_point_list[jring], model, va_norm
                )
                ##################

                # AIC Matrix
                MatrixU[icp, jring] = velocity_induced[0]
                MatrixV[icp, jring] = velocity_induced[1]
                MatrixW[icp, jring] = velocity_induced[2]

                # Only apply correction term when dealing with same horshoe vortex (see p.27 Uri Thesis)
                a = False
                if icp == jring and a:
                    if evaluation_point != "aerodynamic_center":  # if VSM and not LLT
                        # CORRECTION TERM (S.T.Piszkin and E.S.Levinsky,1976)
                        # Not present in classic LLT, added to allow for "arbitrary" (3/4c) control point location [37].
                        U_2D = panel_jring.calculate_velocity_induced_bound_2D(
                            getattr(panel_icp, evaluation_point),
                            gamma=1,
                            core_radius_fraction=core_radius_fraction,
                        )
                        MatrixU[icp, jring] -= U_2D[0]
                        MatrixV[icp, jring] -= U_2D[1]
                        MatrixW[icp, jring] -= U_2D[2]

        return MatrixU, MatrixV, MatrixW

    # TODO: be aware that gamma_0 is NEGATIVE, to accompany the weird reference frame
    def calculate_circulation_distribution_elliptical_wing(self, gamma_0=-1):
        """
        Calculates the circulation distribution for an elliptical wing.

        Args:
            wings (list): List of wing instances
            gamma_0 (float): The circulation at the wing root

        Returns:
            np.array: The circulation distribution
        """
        gamma_i = np.array([])
        if len(self.wings) > 1:
            raise NotImplementedError("Multiple wings not yet implemented")

        wing_span = self.wings[0].calculate_wing_span()

        logging.debug(f"wing_span: {wing_span}")

        y = np.array([panel.control_point[1] for panel in self.panels])
        gamma_i_wing = gamma_0 * np.sqrt(1 - (2 * y / wing_span) ** 2)
        gamma_i = np.append(gamma_i, gamma_i_wing)

        return gamma_i

    def calculate_gamma_distribution(
        self, gamma_distribution, type_initial_gamma_distribution
    ):
        """Calculates the circulation distribution for the wing

        Args:
            gamma_distribution (np.array): The circulation distribution to be used

        Returns:
            np.array: The circulation distribution
        """

        if gamma_distribution is None and type_initial_gamma_distribution == "elliptic":
            self._gamma_distribution = (
                self.calculate_circulation_distribution_elliptical_wing()
            )
            return self._gamma_distribution
        elif (
            gamma_distribution is not None and len(gamma_distribution) == self._n_panels
        ):
            return self._gamma_distribution

    def calculate_results(self, density):

        # Checking that va is not distributed input
        if len(self._va) != 3:
            raise ValueError("Calc.results not ready for va_distributed input")

        # Initializing variables
        cl_prescribed_va_list = []
        cd_prescribed_va_list = []
        cs_prescribed_va_list = []
        lift_prescribed_va_list = []
        drag_prescribed_va_list = []
        side_prescribed_va_list = []
        ftotal_prescribed_va_list = []
        fx_global_list = []
        fy_global_list = []
        fz_global_list = []
        area_all_panels = 0
        lift_wing = 0
        drag_wing = 0
        side_wing = 0
        ftotal_prescribed_va = 0
        fx_global = 0
        fy_global = 0
        fz_global = 0

        spanwise_direction = self.wings[0].spanwise_direction
        va_mag = np.linalg.norm(self._va)
        va = self._va
        va_unit = va / va_mag
        q_inf = 0.5 * density * va_mag**2

        for i, panel_i in enumerate(self.panels):

            ### Defining panel_variables
            # Defining directions of airfoil that defines current panel_i
            z_airf_span = panel_i.z_airf  # along the span
            y_airf_chord = panel_i.y_airf  # along the chord
            x_airf_normal_to_chord = panel_i.x_airf  # normal to the chord
            # TODO: implement these
            alpha_corrected = self._alpha_corrected[i]
            alpha_uncorrected = self._alpha_uncorrected[i]
            panel_chord = panel_i.chord
            panel_width = panel_i.width
            panel_area = panel_chord * panel_width
            area_all_panels += panel_area

            ### Calculate the direction of the induced apparent wind speed to the airfoil orientation
            # this is done using the CORRECTED CALCULATED (comes from gamma distribution) angle of attack
            # For VSM the correction is applied, and it is the angle of attack, from calculating induced velocities at the 1/4c aerodynamic center location
            # For LTT the correction is NOT applied, and it is the angle of attack, from calculating induced velocities at the 3/4c control point
            induced_va_airfoil = (
                np.cos(alpha_corrected) * y_airf_chord
                + np.sin(alpha_corrected) * x_airf_normal_to_chord
            )
            dir_induced_va_airfoil = induced_va_airfoil / np.linalg.norm(
                induced_va_airfoil
            )
            ### Calculate the direction of the lift and drag vectors
            # lift is perpendical/normal to induced apparent wind speed
            # drag is parallel/tangential to induced apparent wind speed
            dir_lift_induced_va = np.cross(dir_induced_va_airfoil, z_airf_span)
            dir_lift_induced_va = dir_lift_induced_va / np.linalg.norm(
                dir_lift_induced_va
            )
            dir_drag_induced_va = np.cross(spanwise_direction, dir_lift_induced_va)
            dir_drag_induced_va = dir_drag_induced_va / np.linalg.norm(
                dir_drag_induced_va
            )

            ### Calculating the MAGNITUDE of the lift and drag
            # The VSM and LTT methods do NOT differ here, both use the uncorrected angle of attack
            # i.e. evaluate the magnitude at the (3/4c) control point

            # panel/airfoil 2D C_l NORMAL to CALCULATED induced velocity
            cl_induced_va = panel_i.calculate_cl(alpha_uncorrected)
            # panel/airfoil 2D C_d, C_m TANGENTIAL to CALCULATED induced velocity
            cd_induced_va, cm_local_va = panel_i.calculate_cd_cm(alpha_uncorrected)
            # 2D AIRFOIL aerodynamic forces, so multiplied by chord
            lift_induced_va_mag = cl_induced_va * q_inf * panel_chord
            drag_induced_va_mag = cd_induced_va * q_inf * panel_chord

            # panel force VECTOR NORMAL to CALCULATED induced velocity
            lift_induced_va = lift_induced_va_mag * dir_lift_induced_va
            # panel force VECTOR TANGENTIAL to CALCULATED induced velocity
            drag_induced_va = drag_induced_va_mag * dir_drag_induced_va
            ftotal_induced_va = lift_induced_va + drag_induced_va

            ### Converting forces to prescribed wing va
            dir_lift_prescribed_va = np.cross(va, spanwise_direction)
            dir_lift_prescribed_va = dir_lift_prescribed_va / np.linalg.norm(
                dir_lift_prescribed_va
            )

            lift_prescribed_va = np.dot(
                lift_induced_va, dir_lift_prescribed_va
            ) + np.dot(drag_induced_va, dir_lift_prescribed_va)
            drag_prescribed_va = np.dot(lift_induced_va, va_unit) + np.dot(
                drag_induced_va, va_unit
            )
            side_prescribed_va = np.dot(lift_induced_va, spanwise_direction) + np.dot(
                drag_induced_va, spanwise_direction
            )

            ftotal_prescribed_va = (
                lift_prescribed_va + drag_prescribed_va + side_prescribed_va
            )

            # TODO: you can check: ftotal_prescribed_va = ftotal_induced_va
            # if not np.allclose(ftotal_prescribed_va, ftotal_induced_va):
            #     raise ValueError(
            #         "Conversion of forces from induced_va to prescribed_va failed"
            #     )
            # The above conversion is merely one of references frames

            ### Converting forces to the global reference frame
            fx_global = np.dot(ftotal_prescribed_va, np.array([1, 0, 0]))
            fy_global = np.dot(ftotal_prescribed_va, np.array([0, 1, 0]))
            fz_global = np.dot(ftotal_prescribed_va, np.array([0, 0, 1]))

            ### Storing results that are useful
            lift_prescribed_va_list.append(lift_prescribed_va)
            drag_prescribed_va_list.append(drag_prescribed_va)
            side_prescribed_va_list.append(side_prescribed_va)
            cl_prescribed_va_list.append(lift_prescribed_va / (q_inf * panel_chord))
            cd_prescribed_va_list.append(drag_prescribed_va / (q_inf * panel_chord))
            cs_prescribed_va_list.append(side_prescribed_va / (q_inf * panel_chord))
            ftotal_prescribed_va_list.append(ftotal_prescribed_va)
            fx_global_list.append(fx_global)
            fy_global_list.append(fy_global)
            fz_global_list.append(fz_global)

            ### Logging
            logging.debug("----calculate_results_new----- icp: %d", i)
            logging.debug(f"dir urel: {dir_induced_va_airfoil}")
            logging.debug(f"dir_L: {dir_lift_induced_va}")
            logging.debug(f"dir_D: {dir_drag_induced_va}")
            logging.debug(
                "lift_induced_va_2d (=L_rel): %s",
                lift_induced_va,
            )
            logging.debug(
                "lift_induced_va_2d (=D_rel): %s",
                drag_induced_va,
            )
            logging.debug(f"Fmag_0: {lift_prescribed_va}")
            logging.debug(f"Fmag_1: {drag_prescribed_va}")
            logging.debug(f"Fmag_2: {side_prescribed_va}")

            # 3D
            lift_wing += lift_prescribed_va * panel_width
            drag_wing += drag_prescribed_va * panel_width
            side_wing += side_prescribed_va * panel_width
            ftotal_prescribed_va += ftotal_prescribed_va * panel_width
            fx_global += fx_global * panel_width
            fy_global += fy_global * panel_width
            fz_global += fz_global * panel_width

        ### Storing results in a dictionary
        results_dict = {}
        # Global wing aerodynamics
        results_dict.update([("Fx", fx_global)])
        results_dict.update([("Fy", fy_global)])
        results_dict.update([("Fz", fz_global)])
        results_dict.update([("lift", lift_wing)])
        results_dict.update([("drag", drag_wing)])
        results_dict.update([("side", side_wing)])
        results_dict.update([("cl", lift_wing / (q_inf * area_all_panels))])
        results_dict.update([("cd", drag_wing / (q_inf * area_all_panels))])
        results_dict.update([("cs", side_wing / (q_inf * area_all_panels))])
        # Local panel aerodynamics
        results_dict.update([("cl_distribution", cl_prescribed_va_list)])
        results_dict.update([("cd_distribution", cd_prescribed_va_list)])
        results_dict.update([("cs_distribution", cs_prescribed_va_list)])

        results_dict.update([("Ftotal_distribution", ftotal_prescribed_va_list)])

        # Additional info
        results_dict.update([("cfz", fz_global / (q_inf * area_all_panels))])
        results_dict.update([("cfx", fx_global / (q_inf * area_all_panels))])
        results_dict.update([("cfy", fy_global / (q_inf * area_all_panels))])
        results_dict.update([("alpha_at_ac", self._alpha_corrected)])
        results_dict.update([("alpha_uncorrected", self._alpha_uncorrected)])
        results_dict.update([("gamma_distribution", self._gamma_distribution)])

        ### Logging
        logging.debug(f"cl:{results_dict['cl']}")
        logging.debug(f"cd:{results_dict['cd']}")
        logging.debug(f"cs:{results_dict['cs']}")
        logging.debug(f"lift:{lift_wing}")
        logging.debug(f"drag:{drag_wing}")
        logging.debug(f"side:{side_wing}")
        logging.debug(f"Area: {area_all_panels}")

        return results_dict

    ###########################
    ## UPDATE FUNCTIONS
    ###########################

    def update_effective_angle_of_attack(
        self, alpha, aerodynamic_model_type, core_radius_fraction
    ):
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
                        getattr(panel_icp, evaluation_point),
                        gamma=1,
                        core_radius_fraction=core_radius_fraction,
                    )
                    # AIC Matrix,WITHOUT U2D CORRECTION
                    MatrixU[icp, jring] = velocity_induced[0]
                    MatrixV[icp, jring] = velocity_induced[1]
                    MatrixW[icp, jring] = velocity_induced[2]

            gamma = self._gamma_distribution
            alpha_corrected_to_aerodynamic_center = np.zeros(len(self.panels))
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
                alpha_corrected_to_aerodynamic_center[icp], _ = (
                    panel.calculate_relative_alpha_and_relative_velocity(
                        induced_velocity
                    )
                )

            self._alpha_corrected = alpha_corrected_to_aerodynamic_center

        # if LTT: no updating is required
        elif aerodynamic_model_type == "LLT":
            self._alpha_corrected = alpha
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
                logging.debug("Legend: %s", legend)
                self.plot_line_segment(ax, [x1, x2], color, legend)

        # Plot the va_vector using the plot_segment
        max_chord = np.max([panel.chord for panel in self.panels])
        va_vector_begin = -2 * max_chord * self.va / np.linalg.norm(self.va)
        va_vector_end = va_vector_begin + 1.5 * self.va / np.linalg.norm(self.va)
        self.plot_line_segment(ax, [va_vector_begin, va_vector_end], "lightblue", "va")

        # Add legends for the first occurrence of each label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # Set equal axis limits
        self.set_axes_equal(ax)

        # Flip the z-axis (to stick to body reference frame)
        # ax.invert_zaxis()

        # Display the plot
        plt.show()
