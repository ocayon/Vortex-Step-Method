import numpy as np
import logging
from VSM.Panel import Panel
import VSM.Wake as Wake
from . import jit_cross, jit_norm, jit_dot


# TODO: should change name to deal with multiple wings
class WingAerodynamics:
    """WingAerodynamics class

    This class is used to calculate the aerodynamic properties of a wing.

    init inputs:
        wings (list): List of Wing object instances
        aerodynamic_center_location (float): The location of the aerodynamic center (default is 0.25)
        control_point_location (float): The location of the control point (default is 0.75)

    Properties:
        panels (list): List of Panel object instances
        n_panels (int): Number of Panel object instances
        va (np.array): The velocity vector of the air
        gamma_distribution (np.array): The circulation distribution
        wings (list): List of Wing object instances

    Methods:
        calculate_panel_properties: Calculate the properties of the panels
        calculate_AIC_matrices: Calculate the AIC matrices
        calculate_circulation_distribution_elliptical_wing: Calculate the circulation distribution for an elliptical wing
        calculate_results: Calculate the results
        update_effective_angle_of_attack_if_VSM: Update the effective angle of attack if VSM
        plot_line_segment: Plot a line segment
    """

    def __init__(
        self,
        wings: list,  # List of Wing object instances
        aerodynamic_center_location: float = 0.25,
        control_point_location: float = 0.75,
    ):
        self._wings = wings
        # Defining the panels by refining the aerodynamic mesh
        panels = []
        for i, wing_instance in enumerate(wings):
            section_list = wing_instance.refine_aerodynamic_mesh()
            n_panels_per_wing = len(section_list) - 1
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
                aerodynamic_center_location,
                control_point_location,
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
        self._panels = panels
        self._n_panels = len(panels)
        self._va = None
        self._gamma_distribution = None
        self._alpha_uncorrected = None
        self._alpha_corrected = None
        self.stall_angle_list = self.calculate_stall_angle_list()

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

    @gamma_distribution.setter
    def gamma_distribution(self, value):
        self._gamma_distribution = value

    @panels.setter
    def panels(self, value):
        self._panels = value

    @va.setter
    def va(self, va, yaw_rate: float = 0.0):

        # # Removing old wake filaments
        # self.panels = Wake.remove_frozen_wake(self.panels)
        if isinstance(va, tuple) and len(va) == 2:
            va, yaw_rate = va

        self._va = np.array(va)

        if len(va) == 3 and yaw_rate == 0.0:
            va_distribution = np.repeat([va], len(self.panels), axis=0)
        elif len(va) == len(self.panels):
            va_distribution = va
        elif yaw_rate != 0.0 and len(va) == 3:
            va_distribution = []

            for wing in self.wings:
                # Create the spanwise positions array
                spanwise_positions = np.array(
                    [panel.control_point[1] for panel in self.panels]
                )

                for i in range(wing.n_panels):
                    yaw_rate_apparent_velocity = np.array(
                        [-yaw_rate * spanwise_positions[i], 0, 0]
                    )

                    # Append the current wing's velocities to the overall distribution
                    va_distribution.append(yaw_rate_apparent_velocity + va)

            # Concatenate all wings' distributions into a single array
            va_distribution = np.vstack(va_distribution)

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

    # TODO: could be CPU optimized
    def calculate_panel_properties(
        self,
        section_list,
        n_panels,
        aerodynamic_center_location,
        control_point_location,
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
            coordinates[2 * i] = section_list[i].LE_point
            coordinates[2 * i + 1] = section_list[i].TE_point
            coordinates[2 * i + 2] = section_list[i + 1].LE_point
            coordinates[2 * i + 3] = section_list[i + 1].TE_point

        logging.debug(f"coordinates: {coordinates}")

        for i in range(n_panels):
            # Identify points defining the panel
            section = {
                "p1": coordinates[2 * i, :],  # p1 = LE_1
                "p2": coordinates[2 * i + 2, :],  # p2 = LE_2
                "p3": coordinates[2 * i + 3, :],  # p3 = TE_2
                "p4": coordinates[2 * i + 1, :],  # p4 = TE_1
            }

            di = jit_norm(
                coordinates[2 * i, :] * 0.75
                + coordinates[2 * i + 1, :] * 0.25
                - (coordinates[2 * i + 2, :] * 0.75 + coordinates[2 * i + 3, :] * 0.25)
            )
            if i == 0:
                diplus = jit_norm(
                    coordinates[2 * (i + 1), :] * 0.75
                    + coordinates[2 * (i + 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i + 1) + 2, :] * 0.75
                        + coordinates[2 * (i + 1) + 3, :] * 0.25
                    )
                )
                ncp = di / (di + diplus)
            elif i == n_panels - 1:
                dimin = jit_norm(
                    coordinates[2 * (i - 1), :] * 0.75
                    + coordinates[2 * (i - 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i - 1) + 2, :] * 0.75
                        + coordinates[2 * (i - 1) + 3, :] * 0.25
                    )
                )
                ncp = dimin / (dimin + di)
            else:
                dimin = jit_norm(
                    coordinates[2 * (i - 1), :] * 0.75
                    + coordinates[2 * (i - 1) + 1, :] * 0.25
                    - (
                        coordinates[2 * (i - 1) + 2, :] * 0.75
                        + coordinates[2 * (i - 1) + 3, :] * 0.25
                    )
                )
                diplus = jit_norm(
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
            # used to be: p2 - p1
            x_airf = jit_cross(VSMpoint - LLpoint, section["p1"] - section["p2"])
            x_airf = x_airf / jit_norm(x_airf)

            # TANGENTIAL y_airf defined parallel to the chord-line, from LE-to-TE
            y_airf = VSMpoint - LLpoint
            y_airf = y_airf / jit_norm(y_airf)

            # SPAN z_airf along the LE, in plane (towards left tip, along span) from the airfoil perspective
            # used to be bound_2 - bound_1
            z_airf = bound_1 - bound_2
            z_airf = z_airf / jit_norm(z_airf)

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

    def calculate_AIC_matrices(
        self, model, core_radius_fraction, va_norm_array, va_unit_array
    ):
        """Calculates the AIC matrices for the given aerodynamic model

        Args:
            model (str): The aerodynamic model to be used, either VSM or LLT
            core_radius_fraction (float): The core radius fraction for the vortex model

        Returns:
            Tuple[np.array, np.array, np.array]: The x, y, and z components of the AIC matrix
        """
        if model not in ["VSM", "LLT"]:
            raise ValueError("Invalid aerodynamic model type, should be VSM or LLT")

        evaluation_point = "control_point" if model == "VSM" else "aerodynamic_center"
        evaluation_point_on_bound = model == "LLT"

        AIC = np.empty((3, self.n_panels, self.n_panels))

        for icp, panel_icp in enumerate(self.panels):
            ep = getattr(panel_icp, evaluation_point)
            for jring, panel_jring in enumerate(self.panels):
                velocity_induced = (
                    panel_jring.calculate_velocity_induced_single_ring_semiinfinite(
                        ep,
                        evaluation_point_on_bound,
                        va_norm_array[jring],
                        va_unit_array[jring],
                        gamma=1,
                        core_radius_fraction=core_radius_fraction,
                    )
                )
                AIC[:, icp, jring] = velocity_induced

                if icp == jring and model == "VSM":
                    U_2D = panel_jring.calculate_velocity_induced_bound_2D(ep)
                    AIC[:, icp, jring] -= U_2D

        return AIC[0], AIC[1], AIC[2]

    def calculate_circulation_distribution_elliptical_wing(self, gamma_0=1):
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

        wing_span = self.wings[0].span

        logging.debug(f"wing_span: {wing_span}")

        y = np.array([panel.control_point[1] for panel in self.panels])
        gamma_i_wing = gamma_0 * np.sqrt(1 - (2 * y / wing_span) ** 2)
        gamma_i = np.append(gamma_i, gamma_i_wing)

        logging.debug(
            f"inside calculate_circulation_distribution_elliptical_wing, gamma_i: {gamma_i}"
        )

        return gamma_i

    def calculate_stall_angle_list(
        self,
        begin_aoa: float = 9,
        end_aoa: float = 22,
        step_aoa: float = 1,
        stall_angle_if_none_detected: float = 50,
        cl_initial: float = -10,
    ):
        """Calculates the stall angle list for each panel

        Args:
            begin_aoa (float): The beginning angle of attack
            end_aoa (float): The end angle of attack
            step_aoa (float): The step angle of attack
            stall_angle_if_none_detected (float): The stall angle if none is detected
            cl_initial (float): The initial lift coefficient

        Returns:
            np.array: The stall angle list"""

        aoa_range_over_which_stall_is_expected = np.deg2rad(
            np.arange(
                begin_aoa,
                end_aoa,
                step_aoa,
            )
        )
        stall_angle_list = []
        for panel in self.panels:
            # initialising a value, for when no stall is found
            panel_aoa_stall = stall_angle_if_none_detected

            # starting with a very small cl value
            cl_old = cl_initial
            for aoa in aoa_range_over_which_stall_is_expected:
                cl = panel.calculate_cl(aoa)
                if cl < cl_old:
                    panel_aoa_stall = aoa
                    break
                cl_old = cl
            stall_angle_list.append(panel_aoa_stall)
        return np.array(stall_angle_list)

    def calculate_results(
        self,
        gamma_new,
        density,
        aerodynamic_model_type,
        core_radius_fraction,
        mu,
        alpha_array,
        Umag_array,
        chord_array,
        x_airf_array,
        y_airf_array,
        z_airf_array,
        va_array,
        va_norm_array,
        va_unit_array,
        panels,
        is_only_f_and_gamma_output,
    ):

        cl_array, cd_array, cm_array = (
            np.zeros(len(panels)),
            np.zeros(len(panels)),
            np.zeros(len(panels)),
        )
        panel_width_array = np.zeros(len(panels))
        for icp, panel_i in enumerate(panels):
            cl_array[icp] = panel_i.calculate_cl(alpha_array[icp])
            cd_array[icp], cm_array[icp] = panel_i.calculate_cd_cm(alpha_array[icp])
            panel_width_array[icp] = panel_i.width
        lift = (cl_array * 0.5 * density * Umag_array**2 * chord_array)[:, np.newaxis]
        drag = (cd_array * 0.5 * density * Umag_array**2 * chord_array)[:, np.newaxis]
        moment = (cm_array * 0.5 * density * Umag_array**2 * chord_array)[:, np.newaxis]

        if aerodynamic_model_type == "VSM":
            alpha_corrected = self.update_effective_angle_of_attack_if_VSM(
                gamma_new,
                core_radius_fraction,
                x_airf_array,
                y_airf_array,
                va_array,
                va_norm_array,
                va_unit_array,
            )
            alpha_uncorrected = alpha_array[:, np.newaxis]

        elif aerodynamic_model_type == "LLT":
            alpha_corrected = alpha_array[:, np.newaxis]
            alpha_uncorrected = alpha_array[:, np.newaxis]
        else:
            raise ValueError("Unknown aerodynamic model type, should be LLT or VSM")
        # Checking that va is not distributed input
        if len(self._va) != 3:
            raise ValueError("Calc.results not ready for va_distributed input")

        # Initializing variables
        cl_prescribed_va_list = []
        cd_prescribed_va_list = []
        cs_prescribed_va_list = []
        f_global_3D_list = []
        fx_global_3D_list = []
        fy_global_3D_list = []
        fz_global_3D_list = []
        area_all_panels = 0
        lift_wing_3D_sum = 0
        drag_wing_3D_sum = 0
        side_wing_3D_sum = 0
        fx_global_3D_sum = 0
        fy_global_3D_sum = 0
        fz_global_3D_sum = 0

        spanwise_direction = self.wings[0].spanwise_direction
        va_mag = jit_norm(self._va)
        va = self._va
        va_unit = va / va_mag
        q_inf = 0.5 * density * va_mag**2

        # induced_va_airfoil_array = (
        #     np.cos(alpha_corrected) * y_airf_array
        #     + np.sin(alpha_corrected) * x_airf_array
        # )
        # dir_induced_va_airfoil_array = induced_va_airfoil_array / jit_norm(
        #     induced_va_airfoil_array
        # )

        # # z_airf_array = np.array([panel.z_airf for panel in self.panels])
        # dir_lift_induced_va_array = jit_cross(dir_induced_va_airfoil_array, z_airf_array)
        # dir_lift_induced_va_array = dir_lift_induced_va_array / jit_norm(
        #     dir_lift_induced_va_array
        # )
        # logging.info(f"induced_va_airfoil_array: {induced_va_airfoil_array.shape}")
        # logging.info(
        #     f"dir_induced_va_airfoil_array: {dir_induced_va_airfoil_array.shape}"
        # )
        # logging.info(f"dir_lift_induced_va_array: {dir_lift_induced_va_array.shape}")

        # breakpoint()
        for i, panel_i in enumerate(self.panels):

            ### Defining panel_variables
            # Defining directions of airfoil that defines current panel_i
            z_airf_span = panel_i.z_airf  # along the span
            y_airf_chord = panel_i.y_airf  # along the chord
            x_airf_normal_to_chord = panel_i.x_airf  # normal to the chord
            # TODO: implement these
            alpha_corrected_i = alpha_corrected[i]
            alpha_uncorrected_i = alpha_uncorrected[i]
            panel_chord = panel_i.chord
            panel_width = panel_i.width
            panel_area = panel_chord * panel_width
            area_all_panels += panel_area

            ### Calculate the direction of the induced apparent wind speed to the airfoil orientation
            # this is done using the CORRECTED CALCULATED (comes from gamma distribution) angle of attack
            # For VSM the correction is applied, and it is the angle of attack, from calculating induced velocities at the 1/4c aerodynamic center location
            # For LTT the correction is NOT applied, and it is the angle of attack, from calculating induced velocities at the 3/4c control point
            induced_va_airfoil = (
                np.cos(alpha_corrected_i) * y_airf_chord
                + np.sin(alpha_corrected_i) * x_airf_normal_to_chord
            )
            dir_induced_va_airfoil = induced_va_airfoil / jit_norm(induced_va_airfoil)
            ### Calculate the direction of the lift and drag vectors
            # lift is perpendical/normal to induced apparent wind speed
            # drag is parallel/tangential to induced apparent wind speed
            dir_lift_induced_va = jit_cross(dir_induced_va_airfoil, z_airf_span)
            dir_lift_induced_va = dir_lift_induced_va / jit_norm(dir_lift_induced_va)
            dir_drag_induced_va = jit_cross(spanwise_direction, dir_lift_induced_va)
            dir_drag_induced_va = dir_drag_induced_va / jit_norm(dir_drag_induced_va)

            # logging.info(f"----before")
            # logging.info(f"shape induced_va_airfoil: {induced_va_airfoil.shape}")
            # logging.info(
            #     f"shape dir_induced_va_airfoil: {dir_induced_va_airfoil.shape}"
            # )
            # logging.info(f"shape dir_lift_induced_va: {dir_lift_induced_va.shape}")
            # logging.info(f"shape dir_drag_induced_va: {dir_drag_induced_va.shape}")

            # logging.info(f"---shape of arrays")
            # logging.info(f"induced_va_airfoil_array: {induced_va_airfoil_array.shape}")
            # # logging.info(
            # #     f"dir_induced_va_airfoil_array: {dir_induced_va_airfoil_array.shape}"
            # # )
            # # logging.info(
            # #     f"dir_lift_induced_va_array: {dir_lift_induced_va_array.shape}"
            # # )
            # # logging.info(
            # #     f"dir_drag_induced_va_array: {dir_drag_induced_va_array.shape}"
            # # )

            # induced_va_airfoil = induced_va_airfoil_array[i, :]
            # dir_induced_va_airfoil = dir_induced_va_airfoil_array[i, :]
            # # dir_lift_induced_va = dir_lift_induced_va_array[i, :]
            # # dir_drag_induced_va = dir_drag_induced_va_array[i, :]

            # logging.info(f"----after")
            # logging.info(f"shape induced_va_airfoil: {induced_va_airfoil.shape}")
            # logging.info(
            #     f"shape dir_induced_va_airfoil: {dir_induced_va_airfoil.shape}"
            # )
            # logging.info(f"shape dir_lift_induced_va: {dir_lift_induced_va.shape}")
            # logging.info(f"shape dir_drag_induced_va: {dir_drag_induced_va.shape}")

            ### Calculating the MAGNITUDE of the lift and drag
            # The VSM and LTT methods do NOT differ here, both use the uncorrected angle of attack
            # i.e. evaluate the magnitude at the (3/4c) control point
            # 2D AIRFOIL aerodynamic forces, so multiplied by chord
            lift_induced_va_mag = lift[i]
            drag_induced_va_mag = drag[i]

            # panel force VECTOR NORMAL to CALCULATED induced velocity
            lift_induced_va = lift_induced_va_mag * dir_lift_induced_va
            # panel force VECTOR TANGENTIAL to CALCULATED induced velocity
            drag_induced_va = drag_induced_va_mag * dir_drag_induced_va
            ftotal_induced_va = lift_induced_va + drag_induced_va
            logging.debug(f"ftotal_induced_va: {ftotal_induced_va}")

            ### Converting forces to prescribed wing va
            dir_lift_prescribed_va = jit_cross(va, spanwise_direction)
            dir_lift_prescribed_va = dir_lift_prescribed_va / jit_norm(
                dir_lift_prescribed_va
            )
            lift_prescribed_va = jit_dot(
                lift_induced_va, dir_lift_prescribed_va
            ) + jit_dot(drag_induced_va, dir_lift_prescribed_va)
            drag_prescribed_va = jit_dot(lift_induced_va, va_unit) + jit_dot(
                drag_induced_va, va_unit
            )
            side_prescribed_va = jit_dot(lift_induced_va, spanwise_direction) + jit_dot(
                drag_induced_va, spanwise_direction
            )

            # TODO: you can check: ftotal_prescribed_va = ftotal_induced_va
            # if not np.allclose(ftotal_prescribed_va, ftotal_induced_va):
            #     raise ValueError(
            #         "Conversion of forces from induced_va to prescribed_va failed"
            #     )
            # The above conversion is merely one of references frames

            ### Converting forces to the global reference frame
            fx_global_2D = jit_dot(ftotal_induced_va, np.array([1, 0, 0]))
            fy_global_2D = jit_dot(ftotal_induced_va, np.array([0, 1, 0]))
            fz_global_2D = jit_dot(ftotal_induced_va, np.array([0, 0, 1]))

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

            # 3D, by multiplying with the panel width
            lift_wing_3D = lift_prescribed_va * panel_width
            drag_wing_3D = drag_prescribed_va * panel_width
            side_wing_3D = side_prescribed_va * panel_width
            fx_global_3D = fx_global_2D * panel_width
            fy_global_3D = fy_global_2D * panel_width
            fz_global_3D = fz_global_2D * panel_width

            # summing it up for totals
            lift_wing_3D_sum += lift_wing_3D
            drag_wing_3D_sum += drag_wing_3D
            side_wing_3D_sum += side_wing_3D
            fx_global_3D_sum += fx_global_3D
            fy_global_3D_sum += fy_global_3D
            fz_global_3D_sum += fz_global_3D

            # Storing results that are useful
            cl_prescribed_va_list.append(lift_prescribed_va / (q_inf * panel_chord))
            cd_prescribed_va_list.append(drag_prescribed_va / (q_inf * panel_chord))
            cs_prescribed_va_list.append(side_prescribed_va / (q_inf * panel_chord))
            fx_global_3D_list.append(fx_global_3D)
            fy_global_3D_list.append(fy_global_3D)
            fz_global_3D_list.append(fz_global_3D)
            f_global_3D_list.append(
                np.array([fx_global_3D, fy_global_3D, fz_global_3D])
            )

        if is_only_f_and_gamma_output:
            return {
                "F_distribution": f_global_3D_list,
                "gamma_distribution": gamma_new,
            }

        # Calculating projected_area, wing_span, aspect_ratio
        projected_area = 0
        for i, wing in enumerate(self.wings):
            projected_area += wing.calculate_projected_area()
        wing_span = wing.span
        aspect_ratio_projected = wing_span**2 / projected_area

        # Calculate geometric angle of attack wrt horizontal at mid-span
        horizontal_direction = np.array([1, 0, 0])
        alpha_geometric = np.array(
            [
                np.rad2deg(
                    np.arccos(jit_dot(panel_i.y_airf, horizontal_direction))
                    / (jit_norm(panel_i.y_airf) * jit_norm(horizontal_direction))
                )
                for panel_i in self.panels
            ]
        )
        # Calculating Reynolds Number
        max_chord = max(np.array([panel.chord for panel in self.panels]))
        reynolds_number = density * va_mag * max_chord / mu

        ### Storing results in a dictionary
        results_dict = {}
        # Global wing aerodynamics
        results_dict.update([("Fx", fx_global_3D_sum)])
        results_dict.update([("Fy", fy_global_3D_sum)])
        results_dict.update([("Fz", fz_global_3D_sum)])
        results_dict.update([("lift", lift_wing_3D_sum)])
        results_dict.update([("drag", drag_wing_3D_sum)])
        results_dict.update([("side", side_wing_3D_sum)])
        results_dict.update([("cl", lift_wing_3D_sum / (q_inf * projected_area))])
        results_dict.update([("cd", drag_wing_3D_sum / (q_inf * projected_area))])
        results_dict.update([("cs", side_wing_3D_sum / (q_inf * projected_area))])
        # Local panel aerodynamics
        results_dict.update([("cl_distribution", cl_prescribed_va_list)])
        results_dict.update([("cd_distribution", cd_prescribed_va_list)])
        results_dict.update([("cs_distribution", cs_prescribed_va_list)])
        results_dict.update([("F_distribution", f_global_3D_list)])

        # Additional info
        results_dict.update(
            [("cfx", np.array(fx_global_3D_list) / (q_inf * projected_area))]
        )
        results_dict.update(
            [("cfy", np.array(fy_global_3D_list) / (q_inf * projected_area))]
        )
        results_dict.update(
            [("cfz", np.array(fz_global_3D_list) / (q_inf * projected_area))]
        )
        results_dict.update([("alpha_at_ac", alpha_corrected)])
        results_dict.update([("alpha_uncorrected", alpha_uncorrected)])
        results_dict.update([("alpha_geometric", alpha_geometric)])
        results_dict.update([("gamma_distribution", gamma_new)])
        results_dict.update([("area_all_panels", area_all_panels)])
        results_dict.update([("projected_area", projected_area)])
        results_dict.update([("wing_span", wing_span)])
        results_dict.update([("aspect_ratio_projected", aspect_ratio_projected)])
        results_dict.update([("Rey", reynolds_number)])

        ### Logging
        logging.debug(f"cl:{results_dict['cl']}")
        logging.debug(f"cd:{results_dict['cd']}")
        logging.debug(f"cs:{results_dict['cs']}")
        logging.debug(f"lift:{lift_wing_3D_sum}")
        logging.debug(f"drag:{drag_wing_3D_sum}")
        logging.debug(f"side:{side_wing_3D_sum}")
        logging.debug(f"area: {area_all_panels}")
        logging.debug(f"Projected Area: {projected_area}")
        logging.debug(f"Aspect Ratio Projected: {aspect_ratio_projected}")

        return results_dict

    ###########################
    ## UPDATE FUNCTIONS
    ###########################

    # def update_effective_angle_of_attack_if_VSM(
    #     self, gamma, core_radius_fraction, va_norm_array, va_unit_array
    # ):
    #     """Updates the angle of attack at the aerodynamic center of each panel,
    #         Calculated at the AERODYNAMIC CENTER, which needs an update for VSM
    #         And can just use the old value for the LLT

    #     Args:
    #         None

    #     Returns:
    #         None
    #     """
    #     # The correction is done by calculating the alpha at the aerodynamic center,
    #     # where as before the control_point was used in the VSM method
    #     aerodynamic_model_type = "LLT"
    #     AIC_x, AIC_y, AIC_z = self.calculate_AIC_matrices(
    #         aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array
    #     )
    #     panels = self.panels
    #     alpha_corrected = np.zeros(len(panels))
    #     for icp, panel in enumerate(panels):
    #         # Initialize induced velocity to 0
    #         u = 0
    #         v = 0
    #         w = 0
    #         # Compute induced velocities with previous gamma distribution
    #         for jring, gamma_jring in enumerate(gamma):
    #             u = u + AIC_x[icp][jring] * gamma_jring
    #             # x-component of velocity
    #             v = v + AIC_y[icp][jring] * gamma_jring
    #             # y-component of velocity
    #             w = w + AIC_z[icp][jring] * gamma_jring
    #             # z-component of velocity

    #         # TODO: shouldn't grab from different classes inside the solver for CPU-efficiency
    #         induced_velocity = np.array([u, v, w])

    #         # This is double checked
    #         alpha_corrected[icp], _ = (
    #             panel.calculate_relative_alpha_and_relative_velocity(induced_velocity)
    #         )

    #     return alpha_corrected

    def update_effective_angle_of_attack_if_VSM(
        self,
        gamma,
        core_radius_fraction,
        x_airf_array,
        y_airf_array,
        va_array,
        va_norm_array,
        va_unit_array,
    ):
        """Updates the angle of attack at the aerodynamic center of each panel,
            Calculated at the AERODYNAMIC CENTER, which needs an update for VSM
            And can just use the old value for the LLT

        Args:
            None

        Returns:
            None
        """
        # The correction is done by calculating the alpha at the aerodynamic center,
        # where as before the control_point was used in the VSM method
        aerodynamic_model_type = "LLT"
        AIC_x, AIC_y, AIC_z = self.calculate_AIC_matrices(
            aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array
        )
        induced_velocity_all = np.array(
            [
                np.matmul(AIC_x, gamma),
                np.matmul(AIC_y, gamma),
                np.matmul(AIC_z, gamma),
            ]
        ).T
        relative_velocity_array = va_array + induced_velocity_all
        v_normal_array = np.sum(x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan(v_normal_array / v_tangential_array)

        return alpha_array[:, np.newaxis]
