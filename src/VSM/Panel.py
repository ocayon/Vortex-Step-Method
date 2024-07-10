import numpy as np
from VSM.Filament import BoundFilament, Infinite2DFilament


class Panel:
    def __init__(
        self,
        section_1,
        section_2,
        aerodynamic_center_location=0.25,
        control_point_location=0.75,
    ):

        self._TE_point_1 = section_1.TE_point
        self._LE_point_1 = section_1.LE_point
        self._TE_point_2 = section_2.TE_point
        self._LE_point_2 = section_2.LE_point
        self._chord = np.average(
            [
                np.linalg.norm(section_1.TE_point - section_1.LE_point),
                np.linalg.norm(section_2.TE_point - section_2.LE_point),
            ]
        )
        self._va = None

        # Defining panel_aero_model
        if section_1.aero_input[0] != section_2.aero_input[0]:
            raise ValueError(
                "Both sections should have the same aero_input, got"
                + section_1.aero_input[0]
                + " and "
                + section_2.aero_input[0]
            )
        self._panel_aero_model = section_1.aero_input[0]

        # Initializing the panel aerodynamic data dependent on the aero_model
        if self._panel_aero_model == "lei_airfoil_breukels":
            self.calculate_lei_airfoil_breukels_cl_cd_cm_coefficients(
                section_1, section_2
            )
        elif self._panel_aero_model == "inviscid":
            pass
        elif self._panel_aero_model == "polar_data":
            # Average the polar_data of the two sections
            aero_1 = section_1.aero_input[1]
            aero_2 = section_2.aero_input[1]
            if len(aero_1) != len(aero_2) or aero_1.shape != aero_2.shape:
                raise ValueError(
                    "The polar data of the two sections should have the same shape & length"
                )
            self._panel_polar_data = (aero_1 + aero_2) / 2
        else:
            raise NotImplementedError

        # Calculate the control point and aerodynamic center
        # Be wary that x is defined positive into the wind, so from TE to LE
        mid_LE_point = self.LE_point_1 + 0.5 * (self._LE_point_2 - self._LE_point_1)
        mid_TE_point = self.TE_point_1 + 0.5 * (self._TE_point_2 - self._TE_point_1)
        vec_LE_to_TE = mid_TE_point - mid_LE_point
        self._aerodynamic_center = (
            mid_LE_point + aerodynamic_center_location * vec_LE_to_TE
        )
        self._control_point = mid_LE_point + control_point_location * vec_LE_to_TE

        # Setting up the filaments
        self.filaments = []
        self.bound_point_1 = self._LE_point_1 + aerodynamic_center_location * (
            self._TE_point_1 - self._LE_point_1
        )
        self.bound_point_2 = self._LE_point_2 + aerodynamic_center_location * (
            self._TE_point_2 - self._LE_point_2
        )
        self.filaments.append(
            BoundFilament(x1=self.bound_point_1, x2=self.bound_point_2)
        )
        self.filaments.append(BoundFilament(x1=self.TE_point_1, x2=self.bound_point_1))
        self.filaments.append(BoundFilament(x1=self.bound_point_2, x2=self.TE_point_2))
        self._filament_2d = Infinite2DFilament(self.bound_point_1, self.bound_point_2)
        self._gamma = None  # Initialize the gamma attribute

        # Calculate the local reference frame, below are all unit_vectors
        # x_airf defined upwards from the chord-line, perpendicular to the panel
        # y_airf defined parallel to the chord-line, from LE-to-TE
        # z_airf along the LE, in plane (towards left tip, along span) from the airfoil perspective
        self._y_airf = vec_LE_to_TE / np.linalg.norm(vec_LE_to_TE)
        self._z_airf = (self.bound_point_2 - self.bound_point_1) / np.linalg.norm(
            self.bound_point_2 - self.bound_point_1
        )
        self._x_airf = np.cross(self._y_airf, self._z_airf)

        # Calculate the width of the panel
        self._average_width = np.average(
            [
                np.linalg.norm(self._TE_point_1 - self._TE_point_2),
                np.linalg.norm(self._LE_point_1 - self._LE_point_2),
            ]
        )

    ###########################
    ## GETTER FUNCTIONS
    ###########################
    @property
    def control_point(self):
        return self._control_point

    @property
    def z_airf(self):
        """Unit vector pointing in the airfoil plane, so that is towards left-tip in spanwise direction"""
        return self._z_airf

    @property
    def x_airf(self):
        """Unit vector pointing upwards from the chord-line, perpendicular to the panel"""
        return self._x_airf

    @property
    def y_airf(self):
        """Unit vector pointing parallel to the chord-line, from LE-to-TE"""
        return self._y_airf

    @property
    def va(self):
        return self._va

    @property
    def aerodynamic_center(self):
        return self._aerodynamic_center

    @property
    def corner_points(self):
        return np.array(
            [self._LE_point_1, self._TE_point_1, self._TE_point_2, self._LE_point_2]
        )

    @property
    def width(self):
        return self._average_width

    @property
    def chord(self):
        return self._chord

    @property
    def TE_point_1(self):
        return self._TE_point_1

    @property
    def TE_point_2(self):
        return self._TE_point_2

    @property
    def LE_point_1(self):
        return self._LE_point_1

    @property
    def LE_point_2(self):
        return self._LE_point_2

    ###########################
    ## SETTER FUNCTIONS
    ###########################
    @va.setter
    def va(self, value):
        self._va = value

    ###########################
    ## CALCULATE FUNCTIONS      # All this return something
    ###########################

    def calculate_relative_alpha_and_relative_velocity(
        self, induced_velocity: np.array
    ):
        """Calculates the relative angle of attack and relative velocity of the panel

        Args:
            induced_velocity (np.array): Induced velocity at the control point

        Returns:
            alpha (float): Relative angle of attack of the panel
            relative_velocity (np.array): Relative velocity of the panel
        """
        # Calculate relative velocity and angle of attack
        relative_velocity = self.va + induced_velocity
        v_normal = np.dot(self.x_airf, relative_velocity)
        v_tangential = np.dot(self.y_airf, relative_velocity)
        alpha = np.arctan(v_normal / v_tangential)
        return alpha, relative_velocity

    def calculate_lei_airfoil_breukels_cl_cd_cm_coefficients(
        self, section_1, section_2
    ):
        t1, k1 = section_1.aero_input[1]
        t2, k2 = section_2.aero_input[1]
        t_avg = (t1 + t2) / 2
        k_avg = (k1 + k2) / 2
        # non-dimensionalized average tube_diameter
        t = t_avg / self._chord
        # non-dimensionalized average max-chamber
        k = k_avg / self._chord

        # cl_coefficients
        C20 = -0.008011
        C21 = -0.000336
        C22 = 0.000992
        C23 = 0.013936
        C24 = -0.003838
        C25 = -0.000161
        C26 = 0.001243
        C27 = -0.009288
        C28 = -0.002124
        C29 = 0.012267
        C30 = -0.002398
        C31 = -0.000274
        C32 = 0
        C33 = 0
        C34 = 0
        C35 = -3.371000
        C36 = 0.858039
        C37 = 0.141600
        C38 = 7.201140
        C39 = -0.676007
        C40 = 0.806629
        C41 = 0.170454
        C42 = -0.390563
        C43 = 0.101966

        S9 = C20 * t**2 + C21 * t + C22
        S10 = C23 * t**2 + C24 * t + C25
        S11 = C26 * t**2 + C27 * t + C28
        S12 = C29 * t**2 + C30 * t + C31
        S13 = C32 * t**2 + C33 * t + C34
        S14 = C35 * t**2 + C36 * t + C37
        S15 = C38 * t**2 + C39 * t + C40
        S16 = C41 * t**2 + C42 * t + C43

        lambda5 = S9 * k + S10
        lambda6 = S11 * k + S12
        lambda7 = S13 * k + S14
        lambda8 = S15 * k + S16

        # # TODO: for CPU efficiency
        # # converting the coefficients to handle alpha input in radians
        # cl_3_rad = lambda5 * (np.pi / 180) ** 3
        # cl_2_rad = lambda6 * (np.pi / 180) ** 2
        # cl_1_rad = lambda7 * (np.pi / 180)
        # cl_0_rad = lambda8
        # self._cl_coefficients = [cl_3_rad, cl_2_rad, cl_1_rad, cl_0_rad]

        self._cl_coefficients = [lambda5, lambda6, lambda7, lambda8]

        # cd_coefficients
        C44 = 0.546094
        C45 = 0.022247
        C46 = -0.071462
        C47 = -0.006527
        C48 = 0.002733
        C49 = 0.000686
        C50 = 0.123685
        C51 = 0.143755
        C52 = 0.495159
        C53 = -0.105362
        C54 = 0.033468

        cd_2_deg = (C44 * t + C45) * k**2 + (C46 * t + C47) * k + (C48 * t + C49)
        cd_1_deg = 0
        cd_0_deg = (C50 * t + C51) * k + (C52 * t**2 + C53 * t + C54)

        # # TODO: for CPU efficiency
        # # converting the coefficients to handle alpha input in radians
        # cd_2_rad = cd_2_deg * (np.pi / 180) ** 2
        # cd_1_rad = cd_1_deg * (np.pi / 180)
        # cd_0_rad = cd_0_deg
        # self._cd_coefficients = [cd_2_rad, cd_1_rad, cd_0_rad]

        self._cd_coefficients = [cd_2_deg, cd_1_deg, cd_0_deg]

        # cm_coefficients
        C55 = -0.284793
        C56 = -0.026199
        C57 = -0.024060
        C58 = 0.000559
        C59 = -1.787703
        C60 = 0.352443
        C61 = -0.839323
        C62 = 0.137932

        cm_2_deg = (C55 * t + C56) * k + (C57 * t + C58)
        cm_1_deg = 0
        cm_0_deg = (C59 * t + C60) * k + (C61 * t + C62)

        # # TODO: for CPU efficiency
        # # converting the coefficients to handle alpha input in radians
        # cm_2_rad = cm_2_deg * (np.pi / 180) ** 2
        # cm_1_rad = cm_1_deg * (np.pi / 180)
        # cm_0_rad = cm_0_deg

        # self._cm_coefficients = [cm_2_rad, cm_1_rad, cm_0_rad]

        self._cm_coefficients = [cm_2_deg, cm_1_deg, cm_0_deg]

    def calculate_cl(self, alpha):
        """
        Get the lift coefficient (Cl) for a given angle of attack.

        Args:
            alpha (float): Angle of attack in radians.
            airfoil_data (np.array): Array containing airfoil data.

        Returns:
            float: Interpolated lift coefficient (Cl).
        """
        if self._panel_aero_model == "lei_airfoil_breukels":
            cl = np.polyval(self._cl_coefficients, np.rad2deg(alpha))
            # if outside of 20 degrees which in rad = np.pi/9
            if alpha > (np.pi / 9) or alpha < -(np.pi / 9):
                cl = 2 * np.cos(alpha) * np.sin(alpha) ** 2
            return cl
        elif self._panel_aero_model == "inviscid":
            return 2 * np.pi * alpha
        elif self._panel_aero_model == "polar_data":
            return np.interp(
                alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 1]
            )
        else:
            raise NotImplementedError

    def calculate_cd_cm(self, alpha):
        """
        Get the lift, drag, and moment coefficients (Cl, Cd, Cm) for a given angle of attack.

        Args:
            alpha (float): Angle of attack in radians.
            airfoil_data (np.array): Array containing airfoil data.

        Returns:
            tuple: Interpolated (Cl, Cd, Cm) coefficients.
        """
        if self._panel_aero_model == "lei_airfoil_breukels":
            cd = np.polyval(self._cd_coefficients, np.rad2deg(alpha))
            cm = np.polyval(self._cm_coefficients, np.rad2deg(alpha))
            # if outside of 20 degrees (np.pi/9)
            if alpha > (np.pi / 9) or alpha < -(np.pi / 9):
                cd = 2 * np.sin(alpha) ** 3
            return cd, cm
        elif self._panel_aero_model == "inviscid":
            cd = 0.0
            cm = 0.0
            return cd, cm
        elif self._panel_aero_model == "polar_data":
            cd = np.interp(
                alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 2]
            )
            cm = np.interp(
                alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 3]
            )
            return cd, cm
        else:
            raise NotImplementedError

    def calculate_velocity_induced_bound_2D(
        self, control_point, gamma, core_radius_fraction
    ):
        """ "
        This function calculates the 2D induced velocity at the control point due to the bound vortex filaments
        """
        if gamma is None:
            gamma = self._gamma

        return self._filament_2d.calculate_induced_velocity(
            control_point, gamma, core_radius_fraction
        )

    def calculate_velocity_induced_horseshoe(
        self, control_point, gamma, core_radius_fraction
    ):
        """ "
        This function calculates the induced velocity at the control point due to the bound vortex filaments
        """
        if gamma is None:
            gamma = self._gamma

        ind_vel = np.zeros(3)
        for filament in self.filaments:
            ind_vel += filament.calculate_induced_velocity(
                control_point, gamma, core_radius_fraction
            )

        return ind_vel

    def calculate_filaments_for_plotting(self):
        filaments = []
        for filament in self.filaments:
            x1 = filament.x1
            if hasattr(filament, "x2") and filament.x2 is not None:
                x2 = filament.x2
                color = "magenta"
            else:
                # For semi-infinite filaments
                x2 = x1 + 2 * self.chord * (self.va / np.linalg.norm(self.va))
                color = "orange"
                if filament.filament_direction == -1:
                    x1, x2 = x2, x1
                    color = "red"

            filaments.append([x1, x2, color])
        return filaments
