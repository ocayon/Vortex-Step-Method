import numpy as np
from VSM.HorshoeVortex import HorshoeVortex


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
        self._airfoil_data = self.calculate_airfoil_data(
            section_1.aero_input, section_2.aero_input
        )
        self._chord = np.average(
            [
                np.abs(section_1.LE_point - section_1.TE_point),
                np.abs(section_2.LE_point - section_2.TE_point),
            ]
        )
        self._horshoe_vortex = HorshoeVortex(
            self._LE_point_1,
            self._TE_point_1,
            self._LE_point_2,
            self._TE_point_2,
            aerodynamic_center_location,
        )
        self._va = None

        # Calculate the control point and aerodynamic center
        mid_LE_point = (self._LE_point_1 + self._LE_point_2) / 2
        mid_TE_point = (self._TE_point_1 + self._TE_point_2) / 2
        vec_LE_to_TE = mid_TE_point - mid_LE_point
        self._aerodynamic_center = (
            mid_LE_point + aerodynamic_center_location * vec_LE_to_TE
        )
        self._control_point = mid_LE_point + control_point_location * vec_LE_to_TE

        # TODO: CHECK Reference Frame Calculations
        # Calculate the local reference frame
        # x_airf defined upwards from the chord-line, perpendicular to the panel
        # y_airf defined parallel to the chord-line, from LE-to-TE
        # z_airf along the LE, out of plane from the airfoil perspective
        self._y_airf = vec_LE_to_TE / np.linalg.norm(vec_LE_to_TE)
        section_1_aerodynamic_center = (
            self._LE_point_1
            + aerodynamic_center_location * (self._LE_point_1 - self._TE_point_1)
        )
        section_2_aerodynamic_center = (
            self._LE_point_2
            + aerodynamic_center_location * (self._LE_point_2 - self._TE_point_2)
        )
        self._z_airf = (
            section_2_aerodynamic_center - section_1_aerodynamic_center
        ) / np.linalg.norm(section_2_aerodynamic_center - section_1_aerodynamic_center)
        self._x_airf = np.cross(self._y_airf, self._z_airf)

    ###########################
    ## GETTER FUNCTIONS
    ###########################
    @property
    def horshoe_vortex(self):
        return self._horshoe_vortex

    @property
    def control_point(self):
        return self._control_point

    @property
    def z_airf(self):
        return self._z_airf

    @property
    def x_airf(self):
        return self._x_airf
    
    @property
    def y_airf(self):
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
    def chord(self):
        return self._chord

    @property
    def TE_point_1(self):
        return self._TE_point_1

    @property
    def TE_point_2(self):
        return self._TE_point_2

    ###########################
    ## SETTER FUNCTIONS
    ###########################
    # TODO: remove this comment. To set va: panel.va = value (array)
    @va.setter
    def va(self, value):
        self._va = value

    ###########################
    ## CALCULATE FUNCTIONS      # All this return smthing
    ###########################

    # TODO: verify that calculate_velocity_induced contains CORE correction
    def calculate_velocity_induced_bound_2D(
        self, evaluation_point: np.array, gamma_mag: float
    ):
        """Calculates the induced velocity inside HorshoeVortex Class"""
        return self._horshoe_vortex.calculate_velocity_induced_bound_2D(
            evaluation_point, gamma_mag
        )

    def calculate_velocity_induced(self, evaluation_point: np.array, gamma_mag = None):
        return self._horshoe_vortex.calculate_velocity_induced_horseshoe(
            evaluation_point, gamma_mag
        )

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
        # Calculate terms of induced corresponding to the airfoil directions
        norm_airf = self._x_airf
        tan_airf = self._y_airf

        # Calculate relative velocity and angle of attack
        relative_velocity = self.va + induced_velocity
        vn = np.dot(norm_airf, relative_velocity)
        vtan = np.dot(tan_airf, relative_velocity)
        alpha = np.arctan(vn / vtan)
        return alpha, relative_velocity

    def _calculate_inviscid_polar_data(self):
        """Calculates the lift, drag and moment coefficients of the panel

        Args:
            None

        Returns:
            airfoil_data (np.array): Array containing the lift, drag and moment coefficients of the panel
        """
        aoa = np.arange(-20, 21, 1)
        airfoil_data = np.empty(
            (
                len(aoa),
                4,
            )
        )
        for j, alpha in enumerate(aoa):
            cl, cd, cm = 2 * np.pi * np.sin(alpha), 0.05, 0.01
            airfoil_data[j, 0] = alpha
            airfoil_data[j, 1] = cl
            airfoil_data[j, 2] = cd
            airfoil_data[j, 3] = cm

        return airfoil_data

    def calculate_airfoil_data(self, aero_input_1, aero_input_2):
        """Calculates the aerodynamic properties of the panel

        Args:
            aero_input_1 (tuple): Aerodynamic properties of the first section
            aero_input_2 (tuple): Aerodynamic properties of the second section

        Returns:
            airfoil_data (np.array): [alpha,cl,cd,cm] of size (n_alpha, 4)
        """
        if (aero_input_1[0] and aero_input_2[0]) == "inviscid":
            return self._calculate_inviscid_polar_data()

        elif (aero_input_1[0] and aero_input_2[0]) == "lei_airfoil_breukels":
            # TODO: 1. Average the Geometry, to find the mid-panel airfoil
            # TODO: 2. Calculate the aerodynamic properties of the mid-panel airfoil
            return NotImplementedError
        else:
            raise NotImplementedError

    def calculate_cl(self, alpha: float):
        """Calculates the lift coefficient of the panel

        Args:
            alpha (float): Angle of attack of the panel

        Returns:
            float: Lift coefficient of the panel
        """
        #TODO: Fix this
        return 2*np.pi*np.sin(alpha)
